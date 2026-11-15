"""Task 2: CPU-only controller computation time per simulated 24-hour day. Maintained entry point.

    python -u -m evaluation.final_analyses.timing run --feeder ieee33 --controller mlp_a0 [--seeds 0 1 2]
            [--topologies TP1] [--scales 1.0] [--max-days N] [--threads 4] [--warmup-days 1]
            [--fmpc-forecast precomputed|online]
    python -u -m evaluation.final_analyses.timing summarize --feeder ieee33 [--runs DIR ...]

Every controller is driven through the unchanged December test environment, 96 decisions per day, on CPU only (the
process refuses to run if CUDA is visible) with the same thread limits (OMP/MKL/OPENBLAS set by the launcher before
Python starts; torch.set_num_threads(--threads)). Only the controller is timed (time.perf_counter, wall clock); the AC
power flow, network backup, environment bookkeeping and table writing are outside every timer. Timer boundary per
decision: from the observation (or environment state) to the battery command, including the controller's own input
preparation, decision computation, action mapping and fallback selection:
  A0-A3     select_action(obs) on CPU (all battery actors) + soc_feasible_action (the affine_v2 SOC-feasible mapping the
            environment applies; computed here once more only to time it, the environment's own result is used)
  PF-MPC    OracleController.select_action, replan 'never', direct kW: the single daily LinDistFlow build + solve at step
            0 (and plan look-ups at steps 1-95); never solved 96 times
  F-MPC     ForecastMPCController.select_action at all 96 steps: forecast retrieval (precomputed) or ONLINE AutoGluon
            recursive inference (--fmpc-forecast online: the saved 'dec' predictors, exactly the causal procedure of
            evaluation.forecast_mpc.generate for the single issue step), DayData preparation, solve, and the 0 kW
            fallback on a failed or non-optimal solve (never excluded). Components: forecast_s, solve_s, prep_s.
            Offline predictor TRAINING is never counted; online inference is reported separately from the solve.
            In online mode every forecast is compared with the saved December forecast (max_abs_diff_vs_saved_kw).
            Solver = the production GUROBI backend exactly as evaluation.forecast_mpc.gurobi_run (GurobiFMPC: MIPGap
            1e-4, TimeLimit 60 s per LP/MILP call, feasible time-limited incumbents accepted, Seed 42, Threads =
            --threads), HiGHS tripwire active; per decision gurobi_calls, gurobi_runtime_s (Gurobi's own optimize
            time; solve_s - gurobi_runtime_s = model construction) and gurobi_status; per day fallback_steps and
            time_limit_incumbent_steps. The backend check and invocation records are excluded from the timer. No table
            is written unless gurobi_run.verify_backend_records proves every solve was Gurobi (environment.json:
            gurobi_verification, highs_calls 0); `summarize` refuses F-MPC runs without it. gurobipy environment
            start (licence session) is reported once as gurobi_env_start_s.
  PQ-RBC    PriceQuantileRBC.select_action with the frozen thresholds file (loaded once, before timing)
Initialisation (checkpoint load, network model build, predictor loading) is timed separately (init.csv). Warm-up: the
first --warmup-days days are run untimed first. Outputs (never overwritten): <out-root>/timing/<feeder>/<UTC>_<controller>/
per_decision.csv, per_day.csv, init.csv, environment.json; `summarize` writes summary.csv (per controller and feeder:
mean, median, SD, p95, max of seconds per day, n days, worst single decision and the share of decisions over 1 s and
over the 900 s control interval) and refuses to combine runs from different CPU models or thread settings.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, TOPOLOGIES, write_json
from evaluation.final_analyses.common import (OUT_ROOT, RESULTS_ROOT, RL, SEEDS, environment_info, guard_output,
                                              rl_test_env, selected_checkpoint, stamp)
from experiments.load_stress.stress_env import check_scale
from utility.evaluation_reporting import reset_for_evaluation
from utility.run_manifest import reserve_directory

TIMED = RL + ('pf_mpc', 'forecast_mpc', 'price_quantile_rbc')
INTERVAL_S = 900.0
STEPS = 96
perf = time.perf_counter


class OnlineForecastStore:
    """Forecast store that runs the saved AutoGluon predictors at each decision (causal, one issue step), timed."""

    def __init__(self, results_root, feeder, saved):
        from evaluation.forecast_mpc.common import load_series, models_dir, read_json, targets, target_kind_bus
        from evaluation.forecast_mpc.generate import load_predictor
        self.series, self.saved, self.feeder = load_series(feeder), saved, feeder
        mdir = models_dir(results_root, feeder, 'dec')
        self.models, started = [], perf()
        for t in targets(feeder):
            rec = read_json(mdir / f'{t}.json')
            if rec.get('series_sha256') != self.series.sha256:
                raise SystemExit(f'{t}: dec model trained on another series')
            kind, bus = target_kind_bus(t)
            self.models.append((t, kind, bus, load_predictor(mdir / t, rec), rec['clip_lower'], rec['clip_upper']))
        self.init_s = perf() - started
        self.n = len(self.models) // 2
        self.last_s, self.last = np.nan, None

    def has_day(self, start):
        return self.saved.has_day(start)

    def window(self, start, step):
        from evaluation.forecast_mpc.generate import recursive_forecast
        t0 = perf()
        out = dict(load=np.empty((96 - step, self.n)), pv=np.empty((96 - step, self.n)))
        s = self.series
        for t, kind, bus, predict, lo, hi in self.models:
            f, _ = recursive_forecast(s.target(t), s.price, s.timestamps, np.array([int(start)]), predict, lo, hi,
                                      issue_steps=[int(step)])
            out[kind][:, bus - 1] = f[0, step, step:]
        self.last_s = perf() - t0
        self.last = (out['load'], out['pv'])
        return out['load'], out['pv']


class GurobiFMPC:
    """The production Gurobi F-MPC backend for timing, through the SAME mechanism as evaluation.forecast_mpc.gurobi_run:
    GurobiBackend (gurobipy env: Threads = --threads, Seed 42), OracleSettings(time_limit_s=60, mip_rel_gap=1e-4),
    feasible time-limited incumbents accepted (GUROBI_FMPC_CONFIG), solver.milp rebound by GurobiBackend.active() and
    the HiGHS tripwire, both only around the F-MPC timing loop of this process (PF-MPC is timed in its own process and
    keeps HiGHS). Each decision is checked (solver.milp is the backend; >= 1 Gurobi call executed) and annotated like
    the production fmpc_solves.csv; at the end verify() reconciles every solve with the backend's own invocation
    records (gurobi_run.verify_backend_records): no timing is written unless every optimisation was Gurobi. One gurobipy
    environment (= one licence session) for the whole process; it ends with the process."""

    def __init__(self, threads):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend
        from evaluation.forecast_mpc.gurobi_run import GUROBI_FMPC_CONFIG, gurobi_identity
        from evaluation.oracle.solver import OracleSettings
        self.config = GUROBI_FMPC_CONFIG
        t0 = perf()
        self.backend = GurobiBackend(int(threads), GUROBI_FMPC_CONFIG['seed'])
        self.env_start_s = perf() - t0                        # gurobipy import + licensed environment start
        self.identity = gurobi_identity(self.backend)
        self.settings = OracleSettings(time_limit_s=GUROBI_FMPC_CONFIG['time_limit_s'],
                                       mip_rel_gap=GUROBI_FMPC_CONFIG['mip_rel_gap'])
        self.allow_non_optimal = bool(GUROBI_FMPC_CONFIG['accept_feasible_incumbent'])
        self.highs_calls, self.solves, self.rows, self._reported = 0, 0, [], False

    def take_env_start(self):
        """The one-off environment start time for the first init row, NaN afterwards (never double counted)."""
        first, self._reported = not self._reported, True
        return self.env_start_s if first else np.nan

    @contextmanager
    def active(self):
        from evaluation.forecast_mpc.gurobi_run import highs_tripwire
        with highs_tripwire(self), self.backend.active():
            yield

    def before(self, ctl):
        from evaluation.forecast_mpc.gurobi_run import BackendVerificationError
        from evaluation.oracle import solver
        if solver.milp != self.backend.milp:
            raise BackendVerificationError('evaluation.oracle.solver.milp is not the Gurobi backend')
        self.backend.last_calls = []
        return self.backend.calls_completed, len(ctl.solves)

    def after(self, ctl, state):
        from evaluation.forecast_mpc.gurobi_run import BackendVerificationError
        before, rows_before = state
        calls = list(self.backend.last_calls)
        if len(ctl.solves) != rows_before + 1:
            raise BackendVerificationError('the controller did not log exactly one solve for this decision')
        if not calls or self.backend.calls_completed - before != len(calls):
            raise BackendVerificationError(f'{len(calls)} Gurobi call(s) recorded, '
                                           f'{self.backend.calls_completed - before} executed')
        row = ctl.solves[-1]
        row.update(solver_backend='gurobi', gurobi_calls=len(calls), solver_native=json.dumps(calls, default=str))
        self.rows.append(row)
        self.solves += 1
        return row, calls

    def verify(self):
        from evaluation.forecast_mpc.gurobi_run import verify_backend_records
        return verify_backend_records(self.rows, self, self.identity)


class TimedStore:
    """Precomputed forecasts with the retrieval timed."""

    def __init__(self, store):
        self.store, self.last_s, self.init_s = store, np.nan, 0.0

    def has_day(self, start):
        return self.store.has_day(start)

    def window(self, start, step):
        t0 = perf()
        w = self.store.window(start, step)
        self.last_s = perf() - t0
        return w


def build(controller, feeder, results_root, out, topology, scale, seed, fmpc_forecast, gurobi=None):
    """(env, decide(obs) -> (command, components), init seconds dict, plan). F-MPC needs `gurobi` (GurobiFMPC)."""
    import torch
    from evaluation.reference_runs import reference_env, split_protocol
    from utility.evaluation_protocol import episode_plan
    init = {}
    if controller in RL:
        from environments.battery_feasibility import soc_feasible_action
        from evaluation.evaluate import load_model
        checkpoint, protocol, record = selected_checkpoint(results_root, feeder, controller, seed)
        env = rl_test_env(feeder, topology, out, protocol, record, seed, scale)
        t0 = perf()
        agent = load_model(controller, checkpoint, env, 'cpu')
        init['checkpoint_load_s'] = perf() - t0
        if any(p.device.type != 'cpu' for p in agent.actor.parameters()):
            raise SystemExit('actor is not on CPU')
        plan_seed = 42 + seed

        def decide(obs):
            t0 = perf()
            with torch.no_grad():
                a = agent.select_action(obs, noise_std=0.0)
            t1 = perf()
            soc_feasible_action(a, env.batteries)
            t2 = perf()
            return a, dict(actor_s=t1 - t0, mapping_s=t2 - t1)
    else:
        protocol = split_protocol('test', [topology])
        env = reference_env(feeder, topology, out, 'soc_feasible', protocol, scale, 'test')
        plan_seed = 42
        if controller == 'pf_mpc':
            from evaluation.oracle.network import LinDistFlowNetwork
            from evaluation.oracle.run import OracleController
            from evaluation.oracle.solver import OracleSettings
            t0 = perf()
            net = LinDistFlowNetwork(env)
            init['network_model_build_s'] = perf() - t0
            ctl = OracleController(env, net, OracleSettings(), topology, scale, dispatch='direct_kw', replan='never',
                                   allow_non_optimal=False, detailed=True)

            def decide(obs):
                before = ctl.day['solve_time_s'] if ctl.day and env.current_step else 0.0
                a = ctl.select_action(obs)
                return a, dict(solve_s=ctl.day['solve_time_s'] - before, solves=ctl.day['solves'])
        elif controller == 'forecast_mpc':
            from evaluation.forecast_mpc.common import ForecastStore, forecasts_dir
            from evaluation.forecast_mpc.controller import ForecastMPCController
            from evaluation.oracle.network import LinDistFlowNetwork
            if gurobi is None:
                raise SystemExit('F-MPC timing runs only with the production Gurobi backend (GurobiFMPC)')
            saved = ForecastStore(forecasts_dir(results_root, feeder, 'dec'), feeder, stage='dec')
            store = OnlineForecastStore(results_root, feeder, saved) if fmpc_forecast == 'online' else TimedStore(saved)
            init['forecast_init_s'] = store.init_s
            init['gurobi_env_start_s'] = gurobi.take_env_start()
            t0 = perf()
            net = LinDistFlowNetwork(env)
            init['network_model_build_s'] = perf() - t0
            ctl = ForecastMPCController(env, net, gurobi.settings, store, feeder, topology, scale,
                                        allow_non_optimal=gurobi.allow_non_optimal)

            def decide(obs):
                t0 = perf()
                state = gurobi.before(ctl)                    # backend check: instrumentation, excluded
                t1 = perf()
                a = ctl.select_action(obs)
                t2 = perf()
                row, calls = gurobi.after(ctl, state)         # invocation records: instrumentation, excluded
                excluded = (t1 - t0) + (perf() - t2)
                comp = dict(forecast_s=store.last_s, solve_s=row['solve_time_s'], fallback=row['fallback'],
                            solver_outcome=row['solver_outcome'], proven_optimal=bool(row['proven_optimal']),
                            solver_backend='gurobi', gurobi_calls=len(calls),
                            gurobi_runtime_s=float(sum(c['runtime_s'] for c in calls)),
                            gurobi_status=calls[-1]['status_name'])
                if fmpc_forecast == 'online':                 # reproduction check: excluded from the timing
                    t0 = perf()
                    ref = saved.window(int(env.episode_start_index), int(env.current_step))
                    comp['max_abs_diff_vs_saved_kw'] = float(max(np.abs(store.last[0] - ref[0]).max(),
                                                                 np.abs(store.last[1] - ref[1]).max()))
                    excluded += perf() - t0
                comp['_excluded_s'] = excluded
                return a, comp
        elif controller == 'price_quantile_rbc':
            from evaluation.rule_based.pq_rbc import PriceQuantileRBC, load_thresholds
            _, th = load_thresholds(results_root, feeder)
            rbc = PriceQuantileRBC(env, th['q30'], th['q70'], topology, scale)

            def decide(obs):
                return rbc.select_action(obs), {}
        else:
            raise SystemExit(f'unknown controller {controller}')
    starts, seeds, _ = episode_plan(env, 'all', plan_seed, protocol)
    return env, decide, init, list(zip(starts, seeds))


def time_day(env, decide, start, seed):
    obs, info = reset_for_evaluation(env, start, seed)
    rows, done = [], False
    while not done:
        step = int(env.current_step)
        t0 = perf()
        command, comp = decide(obs)
        elapsed = perf() - t0                      # comp extraction above is inside; reference checks are not
        rows.append(dict(step=step, controller_s=elapsed - comp.pop('_excluded_s', 0.0), **comp))
        obs, _, terminated, truncated, info = env.step(command)
        done = bool(terminated or truncated or info.get('episode_failed', False))
    return rows


def run(feeder, controller, results_root, out_root, seeds, topologies, scales, max_days, threads, warmup, fmpc_forecast,
        group='adhoc'):
    import torch
    base = guard_output(out_root, results_root)
    if torch.cuda.is_available():
        raise SystemExit('CUDA is visible; run CPU-only (CUDA_VISIBLE_DEVICES="")')
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    label = controller + ('_online' if controller == 'forecast_mpc' and fmpc_forecast == 'online' else '')
    gurobi = GurobiFMPC(threads) if controller == 'forecast_mpc' else None    # licence/API errors stop here
    out = reserve_directory(base / 'timing' / feeder / group / f'{stamp()}_{label}')
    decisions, inits = [], []
    with gurobi.active() if gurobi else nullcontext():
        for seed in (seeds if controller in RL else [None]):
            for tp in topologies:
                for scale in scales:
                    env, decide, init, plan = build(controller, feeder, results_root, out / f'env_{tp}_{seed}', tp,
                                                    scale, seed or 0, fmpc_forecast, gurobi)
                    inits.append(dict(controller=controller, seed=seed, topology=tp, scale=scale, **init))
                    for start, s in plan[:warmup]:
                        time_day(env, decide, start, s)                          # warm-up, untimed
                    for start, s in plan[:max_days or None]:
                        for r in time_day(env, decide, start, s):
                            decisions.append(dict(feeder=feeder, controller=controller, label=label, seed=seed,
                                                  topology=tp, scale=scale, start_index=int(start), **r))
                    print('FINAL_TIMING_PROGRESS', controller, seed, tp, scale, len(decisions), flush=True)
    verification = gurobi.verify() if gurobi else None         # before any table: unverified timings are never written
    if verification:
        print('FINAL_TIMING_GUROBI_VERIFIED', json.dumps(verification), flush=True)
    per = pd.DataFrame(decisions)
    per.to_csv(out / 'per_decision.csv', index=False)
    agg = dict(decisions=('step', 'size'), controller_s=('controller_s', 'sum'),
               max_decision_s=('controller_s', 'max'))
    for col in ('actor_s', 'mapping_s', 'forecast_s', 'solve_s', 'gurobi_runtime_s', 'gurobi_calls'):
        if col in per:
            agg[col] = (col, 'sum')
    keys = ['feeder', 'controller', 'label', 'seed', 'topology', 'scale', 'start_index']
    if 'fallback' in per:
        per['fallback_step'] = (per.fallback != 'none').astype(int)
        agg['fallback_steps'] = ('fallback_step', 'sum')
    if 'proven_optimal' in per:                    # accepted time-limited incumbents (solved, not proven optimal)
        per['incumbent_step'] = ((per.fallback == 'none') & ~per.proven_optimal.astype(bool)).astype(int)
        agg['time_limit_incumbent_steps'] = ('incumbent_step', 'sum')
    day = per.groupby(keys, dropna=False).agg(**agg).reset_index()
    if 'forecast_s' in day and 'solve_s' in day:
        day['prep_s'] = day.controller_s - day.forecast_s - day.solve_s
    day.to_csv(out / 'per_day.csv', index=False)
    pd.DataFrame(inits).to_csv(out / 'init.csv', index=False)
    solver = dict(forecast_mpc='gurobi', pf_mpc='highs').get(controller)
    write_json(out / 'environment.json', dict(
        environment_info(threads), controller=controller, label=label,
        fmpc_forecast=fmpc_forecast if controller == 'forecast_mpc' else None, warmup_days=warmup, max_days=max_days,
        seeds=list(seeds), topologies=list(topologies), scales=list(scales), solver_backend=solver,
        gurobi=gurobi.identity if gurobi else None, gurobi_verification=verification,
        gurobi_env_start_s=gurobi.env_start_s if gurobi else None,
        solver_settings=gurobi.settings.as_dict() if gurobi else None,
        allow_non_optimal=gurobi.allow_non_optimal if gurobi else None))
    print('FINAL_TIMING_DONE', feeder, label, out, f'days={len(day)}', flush=True)
    return out


def summarize(feeder, out_root, runs, group='adhoc'):
    base = guard_output(out_root)
    runs = [Path(r) for r in runs] if runs else sorted((base / 'timing' / feeder / group).glob('*_*'))
    runs = [r for r in runs if (r / 'per_day.csv').is_file()]
    labels = [json.loads((r / 'environment.json').read_text()).get('label') for r in runs]
    if len(labels) != len(set(labels)):
        raise SystemExit(f'more than one run of the same controller in group {group}: {labels}; pass --runs explicitly')
    if not runs:
        raise SystemExit('no finished timing runs')
    envs = [json.loads((r / 'environment.json').read_text()) for r in runs]
    keys = {(e.get('cpu_model'), e.get('requested_threads'), e.get('omp_threads')) for e in envs}
    if len(keys) != 1:
        raise SystemExit(f'timing runs come from different CPU models or thread settings: {sorted(map(str, keys))}')
    for r, e in zip(runs, envs):                           # F-MPC = verified Gurobi only; never a HiGHS-era run
        v = e.get('gurobi_verification') or {}
        if e.get('controller') == 'forecast_mpc' and not (
                e.get('solver_backend') == 'gurobi' and v.get('verified') == 'gurobi_invocation_records'
                and v.get('highs_calls') == 0 and v.get('gurobi_calls', 0) > 0):
            raise SystemExit(f'{r}: F-MPC timing run without verified Gurobi invocation records (HiGHS-era or '
                             'unverified); rerun it or leave it out of --runs')
    backend_of = {e.get('label'): e.get('solver_backend') for e in envs}
    day = pd.concat([pd.read_csv(r / 'per_day.csv') for r in runs], ignore_index=True)
    dec = pd.concat([pd.read_csv(r / 'per_decision.csv') for r in runs], ignore_index=True)
    rows = []
    for (label, scale), g in day.groupby(['label', 'scale']):
        x = g.controller_s.to_numpy()
        d = dec[(dec.label == label) & (dec.scale == scale)].controller_s.to_numpy()
        r = dict(feeder=feeder, controller=label, solver_backend=backend_of.get(label), scale=scale, n_days=len(x),
                 mean_s=x.mean(), median_s=np.median(x),
                 sd_s=x.std(ddof=1) if len(x) > 1 else np.nan, p95_s=np.percentile(x, 95), max_s=x.max(),
                 decisions=len(d), worst_decision_s=d.max(), share_decisions_over_1s=float((d > 1).mean()),
                 share_decisions_over_interval=float((d > INTERVAL_S).mean()),
                 incomplete_days=int((g.decisions != STEPS).sum()) if 'decisions' in g else np.nan)
        for col in ('actor_s', 'mapping_s', 'forecast_s', 'solve_s', 'prep_s', 'gurobi_runtime_s'):
            if col in g and g[col].notna().any():
                r[f'mean_{col}_per_day'] = g[col].mean()
        for col in ('fallback_steps', 'time_limit_incumbent_steps', 'gurobi_calls'):
            if col in g and g[col].notna().any():
                r[f'total_{col}'] = int(g[col].sum())
        rows.append(r)
    out = reserve_directory(base / 'timing' / feeder / group / f'summary_{stamp()}')
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    write_json(out / 'environment.json', dict(runs=[str(r) for r in runs], hardware=envs[0]))
    print('FINAL_TIMING_SUMMARY', out, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--feeder', required=True, choices=FEEDERS)
    r.add_argument('--controller', required=True, choices=TIMED)
    r.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    r.add_argument('--topologies', nargs='+', default=['TP1'])
    r.add_argument('--scales', type=float, nargs='+', default=[1.0])
    r.add_argument('--max-days', type=int)
    r.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    r.add_argument('--warmup-days', type=int, default=1)
    r.add_argument('--fmpc-forecast', choices=('precomputed', 'online'), default='precomputed')
    s = sub.add_parser('summarize')
    s.add_argument('--feeder', required=True, choices=FEEDERS)
    s.add_argument('--runs', nargs='+')
    for x in (r, s):
        x.add_argument('--results-root', default=RESULTS_ROOT)
        x.add_argument('--out-root', default=OUT_ROOT)
        x.add_argument('--group', default='adhoc', help='timing campaign folder (timing_all uses job_<SLURM_JOB_ID>)')
    a = p.parse_args(argv)
    if a.command == 'summarize':
        return summarize(a.feeder, a.out_root, a.runs, a.group)
    unknown = sorted(set(t.upper() for t in a.topologies) - set(TOPOLOGIES[a.feeder]))
    if unknown:
        p.error(f'unknown topologies {unknown}')
    return run(a.feeder, a.controller, a.results_root, a.out_root, a.seeds, [t.upper() for t in a.topologies],
               [check_scale(x) for x in a.scales], a.max_days, a.threads, a.warmup_days, a.fmpc_forecast, a.group)


if __name__ == '__main__':
    main()
