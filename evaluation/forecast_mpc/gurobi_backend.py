"""Gurobi backend for the F-MPC optimisation, reusing the EXACT arrays built by evaluation.oracle.solver. Maintained.

evaluation.oracle.solver.solve() assembles the LP/MILP as objective c, sparse A with row bounds [rlo, rhi], variable
bounds [lb, ub] and an integrality vector, and hands them to scipy.optimize.milp (HiGHS). GurobiBackend.solve_day runs
the unchanged solver.solve_day (LP first; MILP with the charge/discharge exclusivity binaries only if the LP overlaps;
plan validation) while the module-level `milp` name is temporarily bound to GurobiBackend.milp, which receives those
same objects and solves them with Gurobi's matrix API. Nothing is re-formulated: same c, A, row bounds, variable
bounds, integrality columns and minimisation sense. HiGHS remains the default everywhere else (PF-MPC is untouched).

Status mapping onto the HiGHS codes solver.py already interprets (HIGHS_OUTCOMES):
    OPTIMAL                                   -> 0 'optimal'          (MILP: optimal within MIPGap)
    TIME_LIMIT / SUBOPTIMAL with an incumbent -> 1 'time_limit_feasible', x = incumbent
    TIME_LIMIT / SUBOPTIMAL without one       -> 1, x = None          (solver.py raises PlanError -> fallback)
    INFEASIBLE / INF_OR_UNBD                  -> 2 'infeasible', x = None
    UNBOUNDED                                 -> 3 'unbounded', x = None
    any other status (NUMERIC, INTERRUPTED..) -> 4 'failed', x = None
Whether a time-limited incumbent is used is decided by solve_day(allow_non_optimal) exactly as for HiGHS.
Installation, licence, environment, model-construction and API errors raise GurobiBackendError, which is NOT a
RuntimeError: the controller's PlanError -> 0 kW fallback never hides them; the run stops with a traceback.
The native Gurobi status, objective, best bound, MIP gap, runtime, solution count and parameters of every call are kept
in `last_calls` (the controller writes them to fmpc_solves.csv, column solver_native).
"""
from contextlib import contextmanager
import time
from types import SimpleNamespace

import numpy as np

STATUS_NAMES = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF',
                7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED',
                12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT', 16: 'WORK_LIMIT',
                17: 'MEM_LIMIT'}


class GurobiBackendError(Exception):
    """Installation, licence or API failure of the Gurobi backend (never an optimisation fallback)."""


def default_threads(default=1):
    """Gurobi Threads: the Slurm CPU allocation when defined, else `default` (never more than Slurm requested)."""
    import os
    value = os.environ.get('SLURM_CPUS_PER_TASK')
    if value:
        n = int(value)
        if n < 1:
            raise ValueError(f'SLURM_CPUS_PER_TASK={value!r}')
        return n
    return int(default)


def split_rows(A, rlo, rhi):
    """Rows of rlo <= A x <= rhi as Gurobi constraint blocks: (matrix, sense, rhs) for equalities, finite lower
    bounds (>=) and finite upper bounds (<=); a two-sided row gives one >= and one <= block; a free row none."""
    from scipy import sparse
    A = sparse.csr_matrix(A)
    rlo, rhi = np.asarray(rlo, dtype=float), np.asarray(rhi, dtype=float)
    eq = np.isfinite(rlo) & np.isfinite(rhi) & (rlo == rhi)
    ge = np.isfinite(rlo) & ~eq
    le = np.isfinite(rhi) & ~eq
    blocks = []
    for mask, sense, rhs in ((eq, '=', rlo), (ge, '>', rlo), (le, '<', rhi)):
        if mask.any():
            blocks.append((A[mask], sense, rhs[mask]))
    return blocks


class GurobiBackend:
    name = 'gurobi'

    def __init__(self, threads, seed, output_flag=0):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise GurobiBackendError(f'gurobipy is not importable: {exc}') from exc
        self.gp, self.GRB = gp, GRB
        self.threads, self.seed = int(threads), int(seed)
        try:
            self.env = gp.Env(empty=True)
            self.env.setParam('OutputFlag', int(output_flag))
            self.env.setParam('Threads', self.threads)
            self.env.setParam('Seed', self.seed)
            self.env.start()                                  # licence problems surface here
            self.version = '.'.join(str(v) for v in gp.gurobi.version())
        except Exception as exc:  # noqa: BLE001 - any API/licence failure is a backend error
            raise GurobiBackendError(f'Gurobi environment could not be started: {type(exc).__name__}: {exc}') from exc
        self.last_calls = []
        # Invocation evidence: every completed Gurobi optimize() gets a record with this process-unique session id and
        # a sequential call id; the runner reconciles these records with the solves and with calls_completed.
        import uuid
        self.session = uuid.uuid4().hex
        self.calls_completed = 0

    def describe(self, settings, allow_non_optimal):
        return dict(backend='gurobi', gurobi_version=self.version, threads=self.threads, seed=self.seed,
                    mip_rel_gap=float(settings.mip_rel_gap), time_limit_s=float(settings.time_limit_s),
                    accept_feasible_incumbent=bool(allow_non_optimal), objective_sense='minimize',
                    model='evaluation.oracle.solver arrays (c, A, row bounds, variable bounds, integrality), unchanged')

    def milp(self, c, integrality=None, bounds=None, constraints=None, options=None):
        """Drop-in for scipy.optimize.milp as called by evaluation.oracle.solver.solve()."""
        gp, GRB = self.gp, self.GRB
        options = dict(options or {})
        c = np.asarray(c, dtype=float)
        n = len(c)
        lb = np.asarray(bounds.lb, dtype=float) if bounds is not None else np.zeros(n)
        ub = np.asarray(bounds.ub, dtype=float) if bounds is not None else np.full(n, np.inf)
        integ = np.zeros(n) if integrality is None else np.asarray(integrality, dtype=float)
        stage = 'MILP' if integ.any() else 'LP'
        started = time.perf_counter()
        model = None
        try:
            model = gp.Model(env=self.env)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', self.threads)
            model.setParam('Seed', self.seed)
            if 'time_limit' in options:
                model.setParam('TimeLimit', float(options['time_limit']))
            if 'mip_rel_gap' in options:
                model.setParam('MIPGap', float(options['mip_rel_gap']))
            vtype = np.where(integ > 0, GRB.INTEGER, GRB.CONTINUOUS)
            x = model.addMVar(n, lb=np.where(np.isfinite(lb), lb, -GRB.INFINITY),
                              ub=np.where(np.isfinite(ub), ub, GRB.INFINITY), vtype=vtype)
            blocks = split_rows(constraints.A, constraints.lb, constraints.ub)
            for A, sense, rhs in blocks:
                model.addMConstr(A, x, sense, rhs)
            model.setObjective(c @ x, GRB.MINIMIZE)
            model.optimize()
            status, count = int(model.Status), int(model.SolCount)
            native = dict(stage=stage, status=status, status_name=STATUS_NAMES.get(status, str(status)),
                          sol_count=count, runtime_s=float(model.Runtime),
                          objective=float(model.ObjVal) if count else None,
                          best_bound=_attr(model, 'ObjBound') if stage == 'MILP' else None,
                          mip_gap=_attr(model, 'MIPGap') if stage == 'MILP' and count else None,
                          threads=self.threads, seed=self.seed, time_limit_s=options.get('time_limit'),
                          mip_rel_gap=options.get('mip_rel_gap'), variables=n,
                          constraint_rows=int(sum(b[0].shape[0] for b in blocks)))
            solution = np.asarray(x.X, dtype=float) if count else None
        except Exception as exc:  # noqa: BLE001 - API/model errors are backend failures, not optimisation outcomes
            # (BaseExceptions such as KeyboardInterrupt or the runner's SIGTERM signal pass through unchanged)
            raise GurobiBackendError(f'Gurobi {stage} call failed: {type(exc).__name__}: {exc}') from exc
        finally:
            if model is not None:
                try:
                    model.dispose()
                except Exception:  # noqa: BLE001 - freeing memory must not mask the result or the original error
                    pass
        code, keep = self.map_status(status, count)
        self.calls_completed += 1
        native.update(mapped_status=code, wall_s=time.perf_counter() - started, backend='gurobi',
                      gurobi_version=self.version, session=self.session, call_id=self.calls_completed)
        self.last_calls.append(native)
        message = (f"gurobi {native['status_name']} ({status}); sol_count {count}; objective {native['objective']}; "
                   f"bound {native['best_bound']}; gap {native['mip_gap']}")
        return SimpleNamespace(x=solution if keep else None, status=code, message=message,
                               fun=native['objective'] if keep else None)

    def map_status(self, status, count):
        """(HiGHS-equivalent status code, keep the solution?)"""
        GRB = self.GRB
        if status == GRB.OPTIMAL:
            return 0, count > 0
        if status in (GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return 1, count > 0
        if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            return 2, False
        if status == GRB.UNBOUNDED:
            return 3, False
        return 4, False

    @contextmanager
    def active(self):
        """Route evaluation.oracle.solver's milp() call to Gurobi for the duration of one solve_day."""
        from evaluation.oracle import solver
        original = solver.milp
        solver.milp = self.milp
        try:
            yield
        finally:
            solver.milp = original

    def solve_day(self, net, d, settings, allow_non_optimal):
        from evaluation.oracle import solver
        self.last_calls = []
        with self.active():
            return solver.solve_day(net, d, settings, allow_non_optimal)


def _attr(model, name):
    try:
        value = float(getattr(model, name))
    except Exception:  # noqa: BLE001 - attribute not available for this status
        return None
    return value if np.isfinite(value) else None
