"""Forecast-Based MPC (F-MPC) evaluation in the unchanged AC environment. Maintained entry point.

    python -u -m evaluation.forecast_mpc.run --feeder ieee33 --split val  [--max-days N | --days i j k] [--scales ...]
    python -u -m evaluation.forecast_mpc.run --feeder ieee33 --split test [--scales 1.0] [--topologies ...]

Controller: evaluation.forecast_mpc.controller (rest-of-day receding horizon, re-optimised every 15 minutes with PF-MPC's
unchanged LinDistFlow model, causal AutoGluon forecasts, measured SoC, known day-ahead prices; first planned kW
dispatched through direct_kw; physical limiter and AC network backup active; solver failure -> 0 kW request, logged).
Forecasts (monthly expanding-window walk-forward; evaluation.forecast_mpc.common): a validation day in October uses the
'oct' stage forecasts (models fitted on Jan-Sep), a November day the 'nov' forecasts (Jan-Oct), a December day the 'dec'
forecasts (Jan-Nov). Only the stages of the planned days are loaded. The same nominal forecasts serve every load factor:
load x lambda, PV unscaled. The environment, days, reset seeds,
terminal valuation and accounting are those of PF-MPC and the learned controllers (evaluation.reference_runs).

Outputs (never overwritten): test  <root>/<feeder>/forecast_mpc/load_x<pct>[_smoke<N>]/
                             val   <root>/<feeder>/forecast_mpc_validation/load_x<pct>_<days>/
with run_manifest.json ('forecast_mpc_record' block), step_metrics.csv, episode_summary.csv, aggregate_summary.csv,
terminal_soc_summary.csv, fmpc_days.csv (per day: solves, failures, fallback steps, solve times, dispatch statistics)
and fmpc_solves.csv (one row per solve, joined with the executed step's backup activation and AC feasibility).
A single deterministic run per (feeder, load factor): it is never averaged like training seeds.
"""
import argparse
import json
import platform

import pandas as pd
import scipy

from configs.experiments import (B4_FAMILY, MAINTAINED_PROTOCOL, PROTOCOLS, maintained_output_root,
                                 protocol_id, protocol_name)
from evaluation.evaluate import POLICY, feeder_tools, run_episodes, write_tables
from evaluation.forecast_mpc.common import (LABEL, SPLIT_STAGES, STAGE_MONTH, STAGES, ForecastStore, code_inventory,
                                           forecasts_dir)
from evaluation.forecast_mpc.controller import FALLBACK, PRICE_AVAILABILITY, ForecastMPCController
from evaluation.oracle.network import LinDistFlowNetwork
from evaluation.oracle.run import add_oracle_arguments, dispatch_statistics, settings_from
from evaluation.reference_runs import (SEED, SPLITS, check_reference_env, day_plan, env_series_sha256, output_dir,
                                       reference_env, resolve_topologies, split_protocol)
from experiments.load_stress.stress_env import check_scale, scenario

BENCHMARK_NAME = 'Forecast-Based MPC (F-MPC): Rest-of-Day Receding-Horizon LinDistFlow Optimization on Causal Forecasts'
DISCLAIMER = (f'{BENCHMARK_NAME}. Online controller with causal per-bus AutoGluon load/PV forecasts (issued from data '
              'before each decision interval), known day-ahead prices and measured SoC; PF-MPC\'s unchanged lossless '
              'LinDistFlow model re-solved every 15 minutes to midnight; first planned kW dispatched directly in the '
              'nonlinear AC environment with the physical limiter and network backup active. Not a certified AC '
              'optimum; a single deterministic run (not seeds).')
STEP_JOIN = ('backup_activated', 'final_network_feasible', 'episode_failed', 'physical_limited_power_kw',
             'final_battery_power_kw')


def solve_table(solves, frame):
    """One row per solve, joined with the executed step (backup activation, AC feasibility, limiter, executed kW)."""
    table = pd.DataFrame(solves)
    for col in ('soc0', 'requested_kw'):
        table[col] = table[col].map(json.dumps)
    steps = frame[frame.record_type == 'dispatch_attempt'] if 'record_type' in frame else frame
    keys = ['topology_case', 'start_index', 'step']
    keep = keys + [c for c in STEP_JOIN if c in steps]
    return table.merge(steps[keep].drop_duplicates(keys, keep='last'), on=keys, how='left', validate='one_to_one')


def stage_of_day(env, start, split):
    """Monthly walk-forward stage whose models forecast the day starting at row `start`."""
    ts = env.time_series
    month = str(pd.Timestamp(ts['df'][ts['datetime_col']].iloc[int(start)]).strftime('%Y-%m'))
    stage = next((s for s in STAGES if STAGE_MONTH[s] == month), None)
    if stage not in SPLIT_STAGES[split]:
        raise SystemExit(f'day at row {start} ({month}) has no {split} forecast stage')
    return stage


def plan_store(feeder, env, plan, split, stage_dirs, cache):
    """Merged store of exactly the monthly stages the planned days need (each loaded once and cached)."""
    needed = [s for s in SPLIT_STAGES[split] if s in {stage_of_day(env, start, split) for start, _ in plan}]
    for s in needed:
        if s not in cache:
            cache[s] = ForecastStore(stage_dirs[s], feeder, stage=s)
    return ForecastStore.merge([cache[s] for s in needed])


def run_scale(args, scale, settings, stage_dirs, cache):
    feeder, split = args.feeder, args.split
    out = output_dir(args.output_root, feeder, LABEL, split, scale, args.max_days, args.days)
    run = feeder_tools(feeder)[3](out, LABEL, SEED, split, None, vars(args))
    protocol = split_protocol(split, args.topologies)
    run.record_protocol(protocol, [LABEL])
    run.data['forecast_mpc_record'] = dict(
        controller=LABEL, benchmark=BENCHMARK_NAME, disclaimer=DISCLAIMER, experiment=args.experiment,
        protocol=protocol_name(args.experiment), split=split,
        forecast_stages={s: dict(month=STAGE_MONTH[s], directory=str(stage_dirs[s])) for s in SPLIT_STAGES[split]},
        forecast_method='monthly expanding-window walk-forward: one AutoGluon model per bus and quantity per monthly '
                        'cutoff, fitted on every row before the month; causal recursive rest-of-day forecasts at every '
                        '15-minute decision step; never refitted within the month',
        forecast_stores={}, load_scaling_factor=scale, pv_scaling_factor=1.0,
        load_transformation='nominal per-bus load forecast x load_scaling_factor; reactive load = base kvar x total '
                            'load / total base load (DESSEnv._get_profiles); PV forecast unscaled',
        dispatch='direct_kw (first planned kW; physical limiter, network backup and AC power flow active)',
        horizon='rest of the day: decision step k plans steps k..95; re-solved every step (96 solves per day)',
        price_availability=args.price_availability, fallback=f'{FALLBACK}: 0 kW requested for every battery '
        'through direct_kw when a solve fails (or is not proven optimal unless --allow-non-optimal); never an old plan',
        allow_non_optimal=bool(args.allow_non_optimal), settings=settings.as_dict(),
        solver=f'scipy.optimize.milp (HiGHS), scipy {scipy.__version__}', optimisation_model='evaluation.oracle.solver.'
        'solve_day (PF-MPC, unchanged)', python_version=platform.python_version(), source_inventory=code_inventory(),
        evaluation_policy_tag=f'{POLICY} (evaluator dispatch name; controller={LABEL})',
        days=(f'positions {args.days}' if args.days else f'first {args.max_days}' if args.max_days else 'all'),
        deterministic_single_run=True)
    frames, envs, days, solves, stats = [], {}, [], [], []
    for tp in args.topologies:
        env = reference_env(feeder, tp, out, args.experiment, protocol, scale, split)
        check_reference_env(env, scale, feeder)
        plan = day_plan(env, protocol, split, args.max_days, args.days)
        store = plan_store(feeder, env, plan, split, stage_dirs, cache)
        if env_series_sha256(env) != store.series_sha256:
            raise SystemExit(f'forecasts {store.directory} were made from another time series than the environment input')
        missing = [int(s) for s, _ in plan if not store.has_day(s)]
        if missing:
            raise SystemExit(f'no forecasts for {len(missing)} planned days (start rows {missing[:5]}) in {store.directory}')
        run.data['forecast_mpc_record']['forecast_stores'][tp] = store.summary()
        run.add_environment(env, topology=tp, scenario=scenario(scale), intended_episodes=len(plan))
        controller = ForecastMPCController(env, LinDistFlowNetwork(env), settings, store, feeder, tp, scale,
                                           allow_non_optimal=args.allow_non_optimal,
                                           price_mode=args.price_availability)
        frame = run_episodes(feeder, env, controller, POLICY, tp, plan)
        controller.finish()
        frames.append(frame)
        days.extend(controller.records)
        solves.append(solve_table(controller.solves, frame))
        stats.append(dispatch_statistics(frame, env.delta_t_hours))
        envs[tp] = env
        print('FMPC_TOPOLOGY_DONE', feeder, split, tp, scale, len(plan), 'days', flush=True)
    write_tables(feeder, out, frames, envs, LABEL, dict(run_id=None, load_scaling_factor=scale))
    episodes = pd.read_csv(out / 'episode_summary.csv')
    table = pd.DataFrame(days).merge(episodes[['topology_case', 'start_index', 'raw_energy_cost', 'adjusted_energy_cost']],
                                     on=['topology_case', 'start_index'], how='left', validate='one_to_one')
    table = table.merge(pd.concat(stats, ignore_index=True), on=['topology_case', 'start_index'], how='left',
                        validate='one_to_one')
    table.insert(0, 'controller', LABEL)
    table.to_csv(out / 'fmpc_days.csv', index=False)
    solve_log = pd.concat(solves, ignore_index=True)
    solve_log.insert(0, 'controller', LABEL)
    solve_log.to_csv(out / 'fmpc_solves.csv', index=False)
    failed = ~solve_log.solve_ok.astype(bool)
    backup = solve_log.get('backup_activated', pd.Series(False, index=solve_log.index)).fillna(False).astype(bool)
    run.data['forecast_mpc_record']['summary'] = dict(
        days=int(len(table)), solves=int(len(solve_log)), max_solves_per_day=int(table.solves.max()),
        failed_solves=int(failed.sum()), failed_solve_fraction=float(failed.mean()),
        fallback_steps=int((solve_log.fallback != 'none').sum()),
        fallback_steps_with_backup_activation=int((failed & backup).sum()),
        solves_not_proven_optimal=int(table.solves_not_proven_optimal.sum()), milp_solves=int(table.milp_solves.sum()),
        solves_with_network_slack=int(table.solves_with_network_slack.sum()),
        mean_solve_time_s=float(solve_log.solve_time_s.mean()), max_solve_time_s=float(solve_log.solve_time_s.max()),
        total_solve_time_s=float(solve_log.solve_time_s.sum()), days_with_price_substitution=int(table.price_substituted.sum()),
        steps_backup_activated=int(backup.sum()),
        mean_ac_network_feasible_step_rate=float(table.get('ac_network_feasible_step_rate', pd.Series(dtype=float)).mean()),
        max_abs_planned_minus_executed_kw=float(table.get('max_abs_planned_minus_executed_kw',
                                                          pd.Series(dtype=float)).max()))
    run.finish()
    print('FMPC_DONE', out, json.dumps(run.data['forecast_mpc_record']['summary']), flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_oracle_arguments(p)             # --feeder --topologies --scales --max-days and PF-MPC's solver settings
    p.add_argument('--split', required=True, choices=SPLITS,
                   help='val: validation days, TP1, oct/nov forecasts; test: December, all topologies, dec forecasts')
    p.add_argument('--days', type=int, nargs='+', help='val only: positions of validation days (0-based)')
    p.add_argument('--protocol', choices=sorted(k for k, v in PROTOCOLS.items() if v in B4_FAMILY),
                   default=MAINTAINED_PROTOCOL)
    p.add_argument('--output-root', help='default results/<protocol>')
    p.add_argument('--forecast-dirs', nargs='+', metavar='STAGE=DIR',
                   help='override stage folders, e.g. oct=DIR nov=DIR (default <root>/<feeder>/forecast_mpc/forecasts/<stage>)')
    p.add_argument('--allow-non-optimal', action='store_true',
                   help='use time-limited solutions (labelled, not proven optimal) instead of the 0 kW fallback')
    p.add_argument('--price-availability', choices=PRICE_AVAILABILITY, default='dst_guard',
                   help='dst_guard (default): on 2018-03-11..11-03 days the last hour price (next market day) is '
                        'replaced by the known same-hour price of the previous day; identical to full_day in December')
    args = p.parse_args(argv)
    args.experiment = protocol_id(args.protocol)
    try:
        args.output_root = maintained_output_root(args.protocol, args.output_root)
        args.topologies = resolve_topologies(args.feeder, args.split, args.topologies)
    except ValueError as exc:
        p.error(str(exc))
    if args.max_days is not None and not 1 <= args.max_days:
        p.error('--max-days must be positive')
    if args.days and (args.split != 'val' or args.max_days):
        p.error('--days is for --split val and excludes --max-days')
    if args.split == 'test' and args.max_days is not None and args.max_days >= 31:
        p.error('--max-days must be below 31 (omit it for all 31 December days)')
    settings = settings_from(args)
    stage_dirs = {s: forecasts_dir(args.output_root, args.feeder, s) for s in SPLIT_STAGES[args.split]}
    for item in args.forecast_dirs or []:
        stage, sep, directory = item.partition('=')
        if not sep or stage not in stage_dirs:
            p.error(f'--forecast-dirs entries are STAGE=DIR with STAGE in {list(stage_dirs)}, got {item!r}')
        stage_dirs[stage] = directory
    cache = {}                                          # nominal forecasts: loaded once, shared by every load factor
    return [run_scale(args, check_scale(s), settings, stage_dirs, cache) for s in args.scales]


if __name__ == '__main__':
    main()
