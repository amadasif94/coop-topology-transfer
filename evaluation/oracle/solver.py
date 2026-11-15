"""Perfect-foresight day (or rest-of-day) dispatch: LinDistFlow LP, MILP only when needed (SciPy HiGHS).

Decision variables per step t (N batteries): charge ch_ti, discharge dis_ti (kW, >= 0), end-of-step SoC s_ti, grid import
imp_t >= 0 and export exp_t in [0, export limit] (kW), feeder curtailment fraction c_t in [0, 1] (PV_b used =
PV_b (1 - c_t), the env's proportional rule), and nonnegative slacks for voltage (pu^2) and branch loading (pu).

    min  sum_t price_t imp_t dt/1000  -  p_day sum_i E_i s_Ti / 1000            (B4 adjusted cost up to a constant)
         + eps_curt * curtailed kWh + eps_cyc * throughput kWh + M * slacks       (tie-breakers / soft network limits)
    s.t. s_ti = s_(t-1)i + eta_ch dt ch_ti / E_i - dt dis_ti / (eta_dis E_i)      (environments.battery, exactly)
         imp_t - exp_t = sum_b (load_b - PV_b (1 - c_t)) - sum_i (dis_ti - ch_ti)  (lossless feeder balance)
         (V_min + m_v)^2 - slack <= w_bt <= (V_max - m_v)^2 + slack                 (LinDistFlow, every non-slack bus)
         |P_et| <= sqrt((m_b I_e V_min)^2 - Q_et^2) + slack                          (|I| = |S|/|V|; Q is fixed load kvar)
B4 adjusted cost = raw cost - p_day (E_T - E_0)/1000 (experiments.no_shared_kpi.protocol.inventory_adjusted_cost),
p_day = env.day_mean_price; E_0 is a constant here, so minimising the objective minimises the adjusted cost of the
lossless model. The LP allows simultaneous charge/discharge; any overlap is detected and the problem is re-solved as
an MILP with ch_ti <= Pch z_ti, dis_ti <= Pdis (1 - z_ti). The result is NOT an AC optimum: losses and voltage/current
linearisation errors are evaluated afterwards in the AC environment.

Solver outcome (Plan.solver_outcome; Plan.status keeps its historical 'optimal' / 'status_<code>' label):
    optimal               HiGHS status 0: proven optimal (MILP: within mip_rel_gap)          proven_optimal = True
    time_limit_feasible   status 1 with a solution: time/iteration limit, NOT proven optimal  proven_optimal = False
    infeasible / unbounded / failed: no usable solution -> PlanError (never replaced by a fallback dispatch)
solve_day(..., allow_non_optimal=False) also raises PlanError for time_limit_feasible; the historical default (True)
keeps the earlier behaviour. Network limits are soft: voltage (pu^2) and branch (pu) slacks are reported as totals and
maxima, and network_slack_used tells whether the plan needed them (a plan with slack does NOT satisfy every modelled
network constraint).
"""
from dataclasses import dataclass, field
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

OVERLAP_TOL_KW = 1e-6
SLACK_TOL = 1e-7                # slack values at or below this are solver round-off, not a relaxed constraint
HIGHS_OUTCOMES = {0: 'optimal', 1: 'time_limit_feasible', 2: 'infeasible', 3: 'unbounded'}


class PlanError(RuntimeError):
    """No acceptable optimisation result for a day (no solution, or a non-optimal one that was not allowed)."""


@dataclass
class OracleSettings:
    voltage_margin_pu: float = 0.005      # tightens [V_min, V_max] for the lossless voltage model
    branch_margin: float = 0.95           # fraction of the env line-current limit used by the lossless flow model
    slack_penalty: float = 1.0e4          # $ per pu (branch) / per pu^2 (voltage) per step
    eps_curtailment: float = 1.0e-4       # $/kWh, curtail only when needed (matches the env's minimal curtailment)
    eps_throughput: float = 1.0e-6        # $/kWh, tie-breaker only
    time_limit_s: float = 600.0
    mip_rel_gap: float = 1.0e-6

    def as_dict(self):
        return dict(vars(self))


@dataclass
class DayData:
    load_kw: np.ndarray        # [n, buses]
    load_kvar: np.ndarray      # [n, buses]
    pv_kw: np.ndarray          # [n, buses]
    price: np.ndarray          # [n] $/MWh
    soc0: np.ndarray           # [N] SoC at the start of the window
    day_mean_price: float      # $/MWh (B4 terminal valuation)
    day_start_energy_kwh: float
    dt_hours: float
    export_limit_kw: float
    battery_buses: np.ndarray  # [N]
    capacity_kwh: np.ndarray
    p_ch_max_kw: np.ndarray
    p_dis_max_kw: np.ndarray
    eta_ch: np.ndarray
    eta_dis: np.ndarray
    soc_min: np.ndarray
    soc_max: np.ndarray


@dataclass
class Plan:
    status: str
    milp_used: bool
    overlap_steps_lp: int
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    soc: np.ndarray            # [n + 1, N] including the window start
    import_kw: np.ndarray
    export_kw: np.ndarray
    curtail_fraction: np.ndarray
    voltage_slack: float
    branch_slack: float
    objective: float
    predicted_raw_cost: float
    predicted_adjusted_cost: float
    solve_time_s: float
    predicted_v_min: float = np.nan
    predicted_v_max: float = np.nan
    predicted_branch_loading_max: float = np.nan
    solver_status_code: int = -1
    solver_message: str = ''
    solver_outcome: str = ''
    proven_optimal: bool = False
    voltage_slack_max: float = 0.0
    branch_slack_max: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def network_slack_used(self):
        return bool(self.voltage_slack > SLACK_TOL or self.branch_slack > SLACK_TOL)

    @property
    def net_power_kw(self):
        return self.discharge_kw - self.charge_kw


def battery_arrays(env):
    cfg = [b.config for b in env.batteries]
    get = lambda name: np.array([getattr(c, name) for c in cfg], dtype=float)
    return dict(battery_buses=np.asarray(env.dess_buses, dtype=int), capacity_kwh=get('capacity_kwh'),
                p_ch_max_kw=get('max_charge_kw'), p_dis_max_kw=get('max_discharge_kw'), eta_ch=get('charge_efficiency'),
                eta_dis=get('discharge_efficiency'), soc_min=get('soc_min'), soc_max=get('soc_max'))


def day_data(env, first_step, soc0):
    """Realized profiles of the env's current episode from first_step to the end of the day (perfect foresight)."""
    start = int(env.episode_start_index)
    rows = [env._get_profiles(start + t) for t in range(int(first_step), env.episode_length)]
    batteries = battery_arrays(env)
    return DayData(load_kw=np.array([r[0] for r in rows]), load_kvar=np.array([r[1] for r in rows]),
                   pv_kw=np.array([r[2] for r in rows]), price=np.array([r[3] for r in rows], dtype=float),
                   soc0=np.asarray(soc0, dtype=float), day_mean_price=float(env.day_mean_price),
                   day_start_energy_kwh=float(np.asarray(env.start_of_day_soc) @ batteries['capacity_kwh']),
                   dt_hours=float(env.delta_t_hours), export_limit_kw=float(env.grid_export_limit_kw), **batteries)


class _Index:
    def __init__(self, n, N, binary):
        self.n, self.N = n, N
        names = [('ch', n * N), ('dis', n * N), ('soc', n * N), ('imp', n), ('exp', n), ('cur', n), ('svlo', n),
                 ('svhi', n), ('sbr', n)] + ([('z', n * N)] if binary else [])
        self.offset, k = {}, 0
        for name, size in names:
            self.offset[name], k = k, k + size
        self.size = k

    def __call__(self, name, t, i=None):
        return self.offset[name] + (t * self.N + i if i is not None else t)


def solve(net, d: DayData, settings: OracleSettings, binary=False):
    started = time.perf_counter()
    n, N, S = len(d.price), len(d.soc0), net.s_base_kva
    ix = _Index(n, N, binary)
    c = np.zeros(ix.size)
    lb, ub = np.zeros(ix.size), np.full(ix.size, np.inf)
    rows, cols, vals, rlo, rhi = [], [], [], [], []

    def row(entries, lo, hi):
        r = len(rlo)
        for col, val in entries:
            rows.append(r); cols.append(col); vals.append(val)
        rlo.append(lo); rhi.append(hi)

    pv_total = d.pv_kw.sum(1)
    for t in range(n):
        for i in range(N):
            ub[ix('ch', t, i)], ub[ix('dis', t, i)] = d.p_ch_max_kw[i], d.p_dis_max_kw[i]
            lb[ix('soc', t, i)], ub[ix('soc', t, i)] = d.soc_min[i], d.soc_max[i]
            c[ix('ch', t, i)] = c[ix('dis', t, i)] = settings.eps_throughput * d.dt_hours
            a = d.eta_ch[i] * d.dt_hours / d.capacity_kwh[i]
            b = d.dt_hours / (d.eta_dis[i] * d.capacity_kwh[i])
            entries = [(ix('soc', t, i), 1.0), (ix('ch', t, i), -a), (ix('dis', t, i), b)]
            if t > 0:
                entries.append((ix('soc', t - 1, i), -1.0))
                row(entries, 0.0, 0.0)
            else:
                row(entries, d.soc0[i], d.soc0[i])
            if binary:
                ub[ix('z', t, i)] = 1.0
                row([(ix('ch', t, i), 1.0), (ix('z', t, i), -d.p_ch_max_kw[i])], -np.inf, 0.0)
                row([(ix('dis', t, i), 1.0), (ix('z', t, i), d.p_dis_max_kw[i])], -np.inf, d.p_dis_max_kw[i])
        ub[ix('exp', t)] = d.export_limit_kw
        ub[ix('cur', t)] = 1.0 if pv_total[t] > 0 else 0.0
        c[ix('imp', t)] = d.price[t] * d.dt_hours / 1000.0
        c[ix('cur', t)] = settings.eps_curtailment * pv_total[t] * d.dt_hours
        c[ix('svlo', t)] = c[ix('svhi', t)] = c[ix('sbr', t)] = settings.slack_penalty
        balance = [(ix('imp', t), 1.0), (ix('exp', t), -1.0), (ix('cur', t), -pv_total[t])]
        balance += [(ix('ch', t, i), -1.0) for i in range(N)] + [(ix('dis', t, i), 1.0) for i in range(N)]
        rhs = float(d.load_kw[t].sum() - pv_total[t])
        row(balance, rhs, rhs)

        # Network: net demand p = load - PV + PV c_t - (dis - ch) at battery buses (pu); q = load kvar (fixed).
        base_p = (d.load_kw[t] - d.pv_kw[t]) / S
        q = d.load_kvar[t] / S
        w0 = 1.0 - 2.0 * (net.r_path @ base_p + net.x_path @ q)
        dw_cur = -2.0 * (net.r_path @ (d.pv_kw[t] / S))
        lo2 = (net.v_min + settings.voltage_margin_pu) ** 2
        hi2 = (net.v_max - settings.voltage_margin_pu) ** 2
        for bus in net.non_slack:
            entries = [(ix('cur', t), dw_cur[bus])] if dw_cur[bus] else []
            for i, bb in enumerate(d.battery_buses):
                k = 2.0 * net.r_path[bus, bb] / S          # w increases with net injection (dis - ch)
                if k:
                    entries += [(ix('dis', t, i), k), (ix('ch', t, i), -k)]
            row(entries + [(ix('svlo', t), 1.0)], lo2 - w0[bus], np.inf)
            row(entries + [(ix('svhi', t), -1.0)], -np.inf, hi2 - w0[bus])
        flow0, flow_q = net.path @ base_p, net.path @ q
        dflow_cur = net.path @ (d.pv_kw[t] / S)
        s_lim = settings.branch_margin * net.current_limit_pu * net.v_min
        p_lim = np.sqrt(np.maximum(s_lim ** 2 - flow_q ** 2, 0.0))
        for e in range(net.num_lines):
            entries = [(ix('cur', t), dflow_cur[e])] if dflow_cur[e] else []
            for i, bb in enumerate(d.battery_buses):
                if net.path[e, bb]:
                    entries += [(ix('dis', t, i), -1.0 / S), (ix('ch', t, i), 1.0 / S)]
            row(entries + [(ix('sbr', t), -1.0)], -np.inf, p_lim[e] - flow0[e])
            row(entries + [(ix('sbr', t), 1.0)], -p_lim[e] - flow0[e], np.inf)
    for i in range(N):
        c[ix('soc', n - 1, i)] -= d.day_mean_price * d.capacity_kwh[i] / 1000.0

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(rlo), ix.size))
    integrality = np.zeros(ix.size)
    if binary:
        integrality[ix.offset['z']:ix.offset['z'] + n * N] = 1
    res = milp(c, integrality=integrality, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, rlo, rhi),
               options=dict(time_limit=settings.time_limit_s, mip_rel_gap=settings.mip_rel_gap, disp=False))
    if res.x is None:
        raise PlanError(f'oracle {"MILP" if binary else "LP"} failed: status {res.status} '
                        f'({HIGHS_OUTCOMES.get(res.status, "failed")}) {res.message}')
    x = res.x
    take = lambda name, width: x[ix.offset[name]:ix.offset[name] + n * width].reshape(n, width) if width > 1 \
        else x[ix.offset[name]:ix.offset[name] + n]
    ch, dis = np.maximum(take('ch', N), 0.0), np.maximum(take('dis', N), 0.0)
    soc = np.vstack([d.soc0, take('soc', N)])
    imp, cur = take('imp', 1), np.clip(take('cur', 1), 0.0, 1.0)
    raw = float(np.sum(d.price * imp * d.dt_hours / 1000.0))
    end_energy = float(soc[-1] @ d.capacity_kwh)
    v_slack = np.maximum(take('svlo', 1), 0.0) + np.maximum(take('svhi', 1), 0.0)
    b_slack = np.maximum(take('sbr', 1), 0.0)
    plan = Plan(status='optimal' if res.status == 0 else f'status_{res.status}', milp_used=binary,
                overlap_steps_lp=0, charge_kw=ch, discharge_kw=dis, soc=soc, import_kw=imp, export_kw=take('exp', 1),
                curtail_fraction=cur, voltage_slack=float(take('svlo', 1).sum() + take('svhi', 1).sum()),
                branch_slack=float(take('sbr', 1).sum()), objective=float(res.fun),
                predicted_raw_cost=raw,
                predicted_adjusted_cost=raw - d.day_mean_price * (end_energy - d.day_start_energy_kwh) / 1000.0,
                solve_time_s=time.perf_counter() - started, solver_status_code=int(res.status),
                solver_message=str(res.message), solver_outcome=HIGHS_OUTCOMES.get(int(res.status), 'failed'),
                proven_optimal=bool(res.status == 0), voltage_slack_max=float(v_slack.max(initial=0.0)),
                branch_slack_max=float(b_slack.max(initial=0.0)))
    _predicted_network(net, d, plan)
    return plan


def _predicted_network(net, d, plan):
    S, vmin, vmax, load = net.s_base_kva, np.inf, -np.inf, 0.0
    for t in range(len(d.price)):
        p = d.load_kw[t] - d.pv_kw[t] * (1.0 - plan.curtail_fraction[t])
        np.add.at(p, d.battery_buses, -plan.net_power_kw[t])
        v = net.squared_voltage(p / S, d.load_kvar[t] / S)[net.non_slack]
        fp, fq = net.branch_flow(p / S, d.load_kvar[t] / S)
        vmin, vmax = min(vmin, v.min()), max(vmax, v.max())
        load = max(load, float(np.max(np.hypot(fp, fq) / (net.current_limit_pu * net.v_min))))
    plan.predicted_v_min, plan.predicted_v_max, plan.predicted_branch_loading_max = vmin, vmax, load


def overlap_steps(plan):
    return int(np.sum(np.any(np.minimum(plan.charge_kw, plan.discharge_kw) > OVERLAP_TOL_KW, axis=1)))


def _check_outcome(plan, allow_non_optimal, stage):
    if plan.solver_outcome not in ('optimal', 'time_limit_feasible'):
        raise PlanError(f'{stage}: solver outcome {plan.solver_outcome} (status {plan.solver_status_code}) '
                        f'{plan.solver_message}')
    if not plan.proven_optimal and not allow_non_optimal:
        raise PlanError(f'{stage}: {plan.solver_outcome} (status {plan.solver_status_code}, {plan.solver_message}); '
                        'not proven optimal and non-optimal plans were not allowed')


def solve_day(net, d: DayData, settings: OracleSettings, allow_non_optimal=True):
    """LP first; if any step charges and discharges the same battery, re-solve as an MILP (exclusivity binaries).

    allow_non_optimal=False (maintained PF-MPC default): a solve that is not proven optimal raises PlanError.
    True (historical behaviour): a time-limited solution is used and labelled by solver_outcome/proven_optimal."""
    plan = solve(net, d, settings, binary=False)
    _check_outcome(plan, allow_non_optimal, 'LP')
    overlap = overlap_steps(plan)
    if overlap:
        lp_objective = plan.objective
        plan = solve(net, d, settings, binary=True)
        _check_outcome(plan, allow_non_optimal, 'MILP')
        plan.extra['lp_objective_with_overlap'] = lp_objective
        if overlap_steps(plan):
            raise RuntimeError('MILP solution still charges and discharges simultaneously')
    plan.overlap_steps_lp = overlap
    return plan


def normalized_action(plan, d: DayData, step):
    """Env action for the window step: net power / rating, the env's own sign convention (+ discharge)."""
    p = plan.net_power_kw[step]
    return np.where(p >= 0, p / d.p_dis_max_kw, p / d.p_ch_max_kw).clip(-1.0, 1.0).astype(np.float32)


def policy_action(plan, d: DayData, step, env):
    """Policy-side action that makes the env execute the planned power under its action mapping.

    rated: a = P / P_rated (normalized_action). soc_feasible (affine, environments.battery_feasibility): the env
    executes P = -P_ch_avail + (a + 1)/2 * (P_ch_avail + P_dis_avail) at the current (pre-action) SoC, so
    a = 2 * (P + P_ch_avail) / (P_ch_avail + P_dis_avail) - 1 (clipped to [-1, 1]; 0 when nothing is available, which then
    executes 0 kW). The optimisation problem, constraints and objective are identical for both mappings; only this
    conversion differs."""
    if getattr(env, 'action_mapping', 'rated') == 'rated':
        return normalized_action(plan, d, step)
    p = plan.net_power_kw[step]
    out = np.zeros(len(p), dtype=np.float32)
    for i, (power, battery) in enumerate(zip(p, env.batteries)):
        charge = max(0.0, float(battery.get_available_charge_kw()))
        discharge = max(0.0, float(battery.get_available_discharge_kw()))
        if charge + discharge > 0:
            out[i] = np.float32(np.clip(2.0 * (float(power) + charge) / (charge + discharge) - 1.0, -1.0, 1.0))
    return out
