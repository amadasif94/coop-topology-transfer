"""Small HiGHS vs Gurobi benchmark on REAL F-MPC instances (no 31-day run). Maintained.

    # 1) capture: run F-MPC (HiGHS, Gurobi F-MPC settings) on the first December day of one topology and save the exact
    #    optimisation instances (network, DayData, settings) of the listed decision steps; stops after the last one.
    python -u -m evaluation.forecast_mpc.gurobi_benchmark capture --feeder ieee69 --topology TP1 --scale 1.3 \
        --steps 0 1 48 --out /abs/path/fmpc_gurobi_benchmark
    # 2) compare: re-solve every captured instance with HiGHS (scipy.optimize.milp) and Gurobi (gurobi_backend) on the
    #    identical arrays: the two-stage pipeline (solve_day) and, separately, the forced LP and forced MILP stages.
    python -u -m evaluation.forecast_mpc.gurobi_benchmark compare --out /abs/path/fmpc_gurobi_benchmark [--repeats 3]

Per backend and stage: wall/solver runtime, mapped and native status, objective, max row / bound / integrality
residual of the returned x on the SAME (A, row bounds, variable bounds, integrality); per instance: first-step
dispatch (kW) of each backend and their difference. Writes benchmark.json and benchmark.csv in --out.
Nothing here claims a speed-up: read the numbers. Capture states before the captured step follow HiGHS dispatch.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np

from evaluation.forecast_mpc.gurobi_run import GUROBI_FMPC_CONFIG, require


class CaptureDone(BaseException):
    """Stops the capture run after the last requested step (not an Exception: never swallowed by the controller)."""


def check_out(out):
    out = Path(out).resolve()
    for part in out.parts:
        require(not part.lower().startswith('soc_feasible') and part != 'fmpc_ieee69_topology_tasks',
                f'{out} is inside a protected result tree')
    return out


def capture(args):
    from evaluation.forecast_mpc import controller as controller_module
    from evaluation.forecast_mpc import run
    from evaluation.forecast_mpc.gurobi_run import default_forecast_dir
    out = check_out(args.out)
    instances = out / 'instances'
    instances.mkdir(parents=True, exist_ok=True)
    wanted, saved = sorted(set(args.steps)), []
    original = controller_module.solve_day

    def recording_solve_day(net, d, settings, allow_non_optimal=True):
        step = 96 - len(d.price)
        if step in wanted:
            path = instances / f'{args.feeder}_{args.topology}_x{int(round(args.scale * 100))}_step{step:02d}.pkl'
            with path.open('xb') as stream:
                pickle.dump(dict(net=net, data=d, settings=settings, feeder=args.feeder, topology=args.topology,
                                 scale=args.scale, step=step, allow_non_optimal=allow_non_optimal), stream)
            saved.append(str(path))
            print('CAPTURED', path, flush=True)
            if step == wanted[-1]:
                raise CaptureDone()
        return original(net, d, settings, allow_non_optimal)

    dec_dir = Path(args.forecast_dir) if args.forecast_dir else default_forecast_dir(args.feeder)
    argv = ['--feeder', args.feeder, '--split', 'test', '--topologies', args.topology, '--scales', str(args.scale),
            '--max-days', '1', '--output-root', str(out / 'capture_run'), '--forecast-dirs', f'dec={dec_dir}',
            '--time-limit', repr(GUROBI_FMPC_CONFIG['time_limit_s']),
            '--mip-gap', repr(GUROBI_FMPC_CONFIG['mip_rel_gap']), '--allow-non-optimal']
    controller_module.solve_day = recording_solve_day
    try:
        run.main(argv)
    except CaptureDone:
        pass
    finally:
        controller_module.solve_day = original
    require(len(saved) == len(wanted), f'captured {len(saved)} of {len(wanted)} steps')
    return 0


def residuals(x, c, integrality, bounds, constraints):
    ax = constraints.A @ x
    row = float(np.max(np.concatenate([np.maximum(np.asarray(constraints.lb) - ax, 0.0),
                                       np.maximum(ax - np.asarray(constraints.ub), 0.0)]), initial=0.0))
    bnd = float(np.max(np.concatenate([np.maximum(np.asarray(bounds.lb) - x, 0.0),
                                       np.maximum(x - np.asarray(bounds.ub), 0.0)]), initial=0.0))
    integ = np.asarray(integrality) > 0
    intv = float(np.max(np.abs(x[integ] - np.round(x[integ])), initial=0.0)) if integ.any() else 0.0
    return dict(max_row_violation=row, max_bound_violation=bnd, max_integrality_violation=intv,
                objective_recomputed=float(c @ x))


def recorder(fn, calls):
    def milp(c, integrality=None, bounds=None, constraints=None, options=None):
        started = time.perf_counter()
        res = fn(c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
        entry = dict(stage='MILP' if np.any(np.asarray(integrality) > 0) else 'LP', wall_s=time.perf_counter() - started,
                     status=int(res.status), message=str(res.message), objective=None if res.x is None else float(res.fun),
                     variables=int(len(c)), rows=int(constraints.A.shape[0]))
        if res.x is not None:
            entry.update(residuals(np.asarray(res.x, dtype=float), np.asarray(c, dtype=float), integrality, bounds,
                                   constraints))
        calls.append(entry)
        return res
    return milp


def solve_with(backend_name, fn, inst, what):
    """what: 'pipeline' (solve_day), 'LP' or 'MILP' (forced single stage)."""
    from evaluation.oracle import solver
    calls, original = [], solver.milp
    solver.milp = recorder(fn, calls)
    started = time.perf_counter()
    try:
        if what == 'pipeline':
            plan = solver.solve_day(inst['net'], inst['data'], inst['settings'], True)
        else:
            plan = solver.solve(inst['net'], inst['data'], inst['settings'], binary=what == 'MILP')
        result = dict(ok=True, outcome=plan.solver_outcome, proven_optimal=bool(plan.proven_optimal),
                      milp_used=bool(plan.milp_used), objective=float(plan.objective),
                      first_step_kw=np.asarray(plan.net_power_kw[0], dtype=float).tolist())
    except RuntimeError as exc:                                   # PlanError: no usable solution (as in the controller)
        result = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        solver.milp = original
    result.update(backend=backend_name, what=what, total_wall_s=time.perf_counter() - started, calls=calls)
    return result


def compare(args):
    from evaluation.forecast_mpc.gurobi_backend import GurobiBackend, default_threads
    from evaluation.oracle import solver
    out = check_out(args.out)
    paths = sorted((out / 'instances').glob('*.pkl'))
    require(paths, f'no captured instances in {out / "instances"}')
    backend = GurobiBackend(default_threads(), GUROBI_FMPC_CONFIG['seed'])
    highs = solver.milp
    report = dict(gurobi_version=backend.version, threads=backend.threads, seed=backend.seed,
                  scipy=__import__('scipy').__version__, python=platform.python_version(),
                  host=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'), repeats=args.repeats,
                  settings=GUROBI_FMPC_CONFIG, instances=[])
    rows = []
    for path in paths:
        with path.open('rb') as stream:
            inst = pickle.load(stream)
        entry = dict(instance=path.name, step=inst['step'], feeder=inst['feeder'], topology=inst['topology'],
                     settings=inst['settings'].as_dict(), runs=[])
        for what in ('pipeline', 'LP', 'MILP'):
            for name, fn in (('highs', highs), ('gurobi', backend.milp)):
                for rep in range(args.repeats):
                    r = solve_with(name, fn, inst, what)
                    r['repeat'] = rep
                    entry['runs'].append(r)
                    for call in r['calls'] or [{}]:
                        rows.append(dict(instance=path.name, step=inst['step'], what=what, backend=name, repeat=rep,
                                         ok=r['ok'], outcome=r.get('outcome'), total_wall_s=r['total_wall_s'],
                                         **{f'call_{k}': v for k, v in call.items()}))
            first = {n: next(r for r in entry['runs'] if r['what'] == what and r['backend'] == n) for n in
                     ('highs', 'gurobi')}
            if first['highs']['ok'] and first['gurobi']['ok']:
                a, b = first['highs'], first['gurobi']
                entry[f'{what}_comparison'] = dict(
                    objective_abs_diff=abs(a['objective'] - b['objective']),
                    objective_rel_diff=abs(a['objective'] - b['objective']) / max(1.0, abs(a['objective'])),
                    first_step_max_abs_kw_diff=float(np.max(np.abs(np.subtract(a['first_step_kw'], b['first_step_kw'])))),
                    median_wall_s=dict(highs=float(np.median([r['total_wall_s'] for r in entry['runs']
                                                              if r['what'] == what and r['backend'] == 'highs'])),
                                       gurobi=float(np.median([r['total_wall_s'] for r in entry['runs']
                                                               if r['what'] == what and r['backend'] == 'gurobi']))))
        entry['gurobi_native_last'] = backend.last_calls[-1:] if backend.last_calls else []
        report['instances'].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k.endswith('_comparison') or k in ('instance', 'step')}),
              flush=True)
    import pandas as pd
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out / 'benchmark.csv', index=False)
    print('BENCHMARK_WRITTEN', out / 'benchmark.json', out / 'benchmark.csv', flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--feeder', required=True, choices=('ieee33', 'ieee69'))
    c.add_argument('--topology', default='TP1', type=str.upper)
    c.add_argument('--scale', type=float, default=1.3, choices=(1.0, 1.1, 1.2, 1.3))
    c.add_argument('--steps', type=int, nargs='+', default=[0])
    c.add_argument('--forecast-dir')
    c.add_argument('--out', required=True)
    m = sub.add_parser('compare')
    m.add_argument('--out', required=True)
    m.add_argument('--repeats', type=int, default=1)
    args = p.parse_args(argv)
    if args.command == 'capture':
        if not all(0 <= s < 96 for s in args.steps):
            p.error('--steps are decision steps 0..95')
        return capture(args)
    return compare(args)


if __name__ == '__main__':
    sys.exit(main())
