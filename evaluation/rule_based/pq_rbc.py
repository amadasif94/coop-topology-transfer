"""Price-Quantile Rule-Based Control (PQ-RBC). Maintained entry point.

    python -u -m evaluation.rule_based.pq_rbc freeze --feeder ieee33
    python -u -m evaluation.rule_based.pq_rbc run    --feeder ieee33 --split val  [--max-days N | --days i j k]
    python -u -m evaluation.rule_based.pq_rbc run    --feeder ieee33 --split test [--scales ...] [--topologies ...]

freeze  q30 and q70 of the day-ahead prices of the environment's TRAINING split (2018-01-01 .. 09-30; the
        environment's own loader and split rule; numpy.quantile, method 'linear'), written once to
        <root>/<feeder>/price_quantile_rbc/thresholds.json with the series sha256. Never recomputed per topology,
        load factor or split; an existing file is never overwritten.
run     at every step, with p = the current interval's day-ahead price (published before the day):
            p <= q30  ->  request -P_ch_avail   (charge at the maximum available power)
            p >= q70  ->  request +P_dis_avail  (discharge at the maximum available power)
            otherwise ->  0 kW
        per battery, with P_ch_avail / P_dis_avail the SoC-dependent available powers of the authoritative battery
        model (environments.battery_feasibility.feasible_powers_kw, the same limits the RL action mapping uses);
        + = discharge, the environment's convention. Requests go through the environment's direct_kw path (no RL action
        mapping), so the physical limiter and the AC network backup act as for every other controller. No optimiser,
        no forecasts, no load/PV data.

Outputs (never overwritten): test <root>/<feeder>/price_quantile_rbc/load_x<pct>[_smoke<N>]/, val
<root>/<feeder>/price_quantile_rbc_validation/load_x<pct>_<days>/: run_manifest.json ('pq_rbc_record'), the standard
tables and pqrbc_days.csv (charge/idle/discharge steps and dispatch statistics per day). A single deterministic run per
(feeder, load factor); never averaged like training seeds.
"""
import argparse
import json
import platform

import numpy as np
import pandas as pd

from configs.experiments import (B4_FAMILY, FEEDERS, MAINTAINED_PROTOCOL, PROTOCOLS, ROOT, feeder_config,
                                 maintained_output_root, protocol_id, protocol_name, sha, source_inventory, write_json)
from environments.battery_feasibility import feasible_powers_kw
from evaluation.evaluate import POLICY, feeder_tools, run_episodes, write_tables
from evaluation.oracle.run import dispatch_statistics
from evaluation.reference_runs import (SEED, SPLITS, check_reference_env, day_plan, env_series_sha256, output_dir,
                                       reference_env, resolve_topologies, split_protocol)
from experiments.load_stress.stress_env import ALLOWED_SCALES, check_scale, scenario
from utility.data_loader import load_processed_node_time_series, split_time_series_by_date

LABEL = 'price_quantile_rbc'
THRESHOLD_SCHEMA = 'graph-marl-dess-pq-rbc-thresholds/1'
QUANTILES = (0.30, 0.70)
NUM_BUSES = dict(ieee33=33, ieee69=69)
PQ_SOURCES = ('evaluation/rule_based/__init__.py', 'evaluation/rule_based/pq_rbc.py', 'evaluation/reference_runs.py',
              'evaluation/oracle/run.py')
BENCHMARK_NAME = 'Price-Quantile Rule-Based Control (PQ-RBC)'


def thresholds_path(root, feeder):
    return maintained_output_root(MAINTAINED_PROTOCOL, root) / feeder / LABEL / 'thresholds.json'


def compute_thresholds(feeder, time_series=None):
    config = feeder_config(feeder, time_series, experiment='soc_feasible')
    path = config['paths']['processed_time_series']
    ts = load_processed_node_time_series(path, NUM_BUSES[feeder], allow_empty=False)
    rows = np.asarray(split_time_series_by_date(ts, config)['train_indices'], dtype=int)
    factor = float(config.get('data_scaling', {}).get('price_scaling_factor', 1.0))
    prices = ts['df'][ts['price_col']].to_numpy(dtype=float)[rows] * factor
    if not np.isfinite(prices).all() or prices.size == 0:
        raise ValueError('training prices missing or nonfinite')
    low, high = (float(np.quantile(prices, q, method='linear')) for q in QUANTILES)
    if not low < high:
        raise ValueError(f'q30 {low} is not below q70 {high}')
    stamps = ts['df'][ts['datetime_col']]
    return dict(schema=THRESHOLD_SCHEMA, feeder=feeder, controller=LABEL, q30=low, q70=high,
                quantiles=list(QUANTILES), method='numpy.quantile(method="linear") over 15-minute training rows',
                split='train', split_first_time=str(stamps.iloc[rows.min()]), split_last_time=str(stamps.iloc[rows.max()]),
                rows=int(rows.size), price_scaling_factor=factor, series_path=str(path), series_sha256=sha(path),
                train_price_min=float(prices.min()), train_price_max=float(prices.max()),
                train_price_mean=float(prices.mean()), fraction_at_or_below_q30=float((prices <= low).mean()),
                fraction_at_or_above_q70=float((prices >= high).mean()))


class PriceQuantileRBC:
    """Duck-typed agent for the existing evaluators: select_action(obs, noise_std) returns the battery kW request."""

    def __init__(self, env, q30, q70, topology, scale):
        if not q30 < q70:
            raise ValueError('q30 must be below q70')
        self.env, self.q30, self.q70, self.topology, self.scale = env, float(q30), float(q70), topology, float(scale)
        env.set_dispatch_mode('direct_kw')
        self.records, self.day = [], None

    def current_price(self):
        """Day-ahead price of the interval being dispatched (the price the environment charges for it)."""
        ts = self.env.time_series
        return float(ts['df'][ts['price_col']].iloc[int(self.env.current_index)]) * float(self.env.price_scaling_factor)

    def decide(self, price, charge_available_kw, discharge_available_kw):
        """The rule alone: (mode, kW request per battery), + = discharge."""
        if price <= self.q30:
            return 'charge', -np.asarray(charge_available_kw, dtype=float)
        if price >= self.q70:
            return 'discharge', np.asarray(discharge_available_kw, dtype=float).copy()
        return 'idle', np.zeros(len(charge_available_kw), dtype=float)

    def select_action(self, obs, noise_std=0.0):
        env = self.env
        if getattr(env, 'dispatch_mode', 'policy') != 'direct_kw':
            raise RuntimeError('PQ-RBC needs the environment in direct_kw dispatch mode')
        if int(env.current_step) == 0:
            self.finish()
            self.day = dict(topology_case=self.topology, load_scaling_factor=self.scale,
                            start_index=int(env.episode_start_index), charge_steps=0, idle_steps=0, discharge_steps=0,
                            requested_charge_kwh=0.0, requested_discharge_kwh=0.0)
        charge, discharge = feasible_powers_kw(env)              # pre-action, SoC-dependent, authoritative model
        mode, request = self.decide(self.current_price(), charge, discharge)
        self.day[f'{mode}_steps'] += 1
        self.day['requested_charge_kwh'] += float(np.clip(-request, 0, None).sum() * env.delta_t_hours)
        self.day['requested_discharge_kwh'] += float(np.clip(request, 0, None).sum() * env.delta_t_hours)
        return request

    def finish(self):
        if self.day is not None:
            self.records.append(self.day)
            self.day = None


def load_thresholds(root, feeder):
    path = thresholds_path(root, feeder)
    if not path.is_file():
        raise SystemExit(f'{path} missing: run "pq_rbc freeze --feeder {feeder}" first')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != THRESHOLD_SCHEMA or record.get('feeder') != feeder:
        raise SystemExit(f'{path} is not a PQ-RBC threshold record of {feeder}')
    return path, record


def run_scale(args, scale, thresholds_file, thresholds):
    feeder, split = args.feeder, args.split
    out = output_dir(args.output_root, feeder, LABEL, split, scale, args.max_days, args.days)
    run = feeder_tools(feeder)[3](out, LABEL, SEED, split, None, vars(args))
    protocol = split_protocol(split, args.topologies)
    run.record_protocol(protocol, [LABEL])
    run.data['pq_rbc_record'] = dict(
        controller=LABEL, benchmark=BENCHMARK_NAME, experiment=args.experiment, protocol=protocol_name(args.experiment),
        split=split, load_scaling_factor=scale, pv_scaling_factor=1.0, q30=thresholds['q30'], q70=thresholds['q70'],
        thresholds_file=str(thresholds_file), thresholds_sha256=sha(thresholds_file),
        rule='price <= q30: -P_ch_avail; price >= q70: +P_dis_avail; else 0 kW (per battery; + = discharge)',
        available_power='environments.battery_feasibility.feasible_powers_kw (pre-action SoC, rating, efficiency)',
        dispatch='direct_kw (no RL action mapping; physical limiter, network backup and AC power flow active)',
        inputs='current day-ahead price, battery state, frozen thresholds only (no optimiser, no forecasts)',
        python_version=platform.python_version(),
        source_inventory=dict(source_inventory(), **{p: sha(ROOT / p) for p in PQ_SOURCES}),
        days=(f'positions {args.days}' if args.days else f'first {args.max_days}' if args.max_days else 'all'),
        deterministic_single_run=True)
    frames, envs, days, stats = [], {}, [], []
    for tp in args.topologies:
        env = reference_env(feeder, tp, out, args.experiment, protocol, scale, split)
        check_reference_env(env, scale, feeder)
        if env_series_sha256(env) != thresholds['series_sha256']:
            raise SystemExit('thresholds were computed from another time series than the environment input')
        plan = day_plan(env, protocol, split, args.max_days, args.days)
        run.add_environment(env, topology=tp, scenario=scenario(scale), intended_episodes=len(plan))
        controller = PriceQuantileRBC(env, thresholds['q30'], thresholds['q70'], tp, scale)
        frame = run_episodes(feeder, env, controller, POLICY, tp, plan)
        controller.finish()
        frames.append(frame)
        days.extend(controller.records)
        stats.append(dispatch_statistics(frame, env.delta_t_hours))
        envs[tp] = env
        print('PQ_RBC_TOPOLOGY_DONE', feeder, split, tp, scale, len(plan), 'days', flush=True)
    write_tables(feeder, out, frames, envs, LABEL, dict(run_id=None, load_scaling_factor=scale))
    episodes = pd.read_csv(out / 'episode_summary.csv')
    table = pd.DataFrame(days).merge(episodes[['topology_case', 'start_index', 'raw_energy_cost', 'adjusted_energy_cost']],
                                     on=['topology_case', 'start_index'], how='left', validate='one_to_one')
    table = table.merge(pd.concat(stats, ignore_index=True), on=['topology_case', 'start_index'], how='left',
                        validate='one_to_one')
    table.insert(0, 'controller', LABEL)
    table.to_csv(out / 'pqrbc_days.csv', index=False)
    run.data['pq_rbc_record']['summary'] = dict(
        days=int(len(table)), charge_steps=int(table.charge_steps.sum()), idle_steps=int(table.idle_steps.sum()),
        discharge_steps=int(table.discharge_steps.sum()),
        steps_backup_activated=int(table.get('ac_backup_activation_steps', pd.Series(0)).sum()),
        mean_ac_network_feasible_step_rate=float(table.get('ac_network_feasible_step_rate', pd.Series(dtype=float)).mean()),
        max_abs_planned_minus_executed_kw=float(table.get('max_abs_planned_minus_executed_kw',
                                                          pd.Series(dtype=float)).max()))
    run.finish()
    print('PQ_RBC_DONE', out, run.data['pq_rbc_record']['summary'], flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'run'):
        s = sub.add_parser(name)
        s.add_argument('--feeder', required=True, choices=FEEDERS)
        s.add_argument('--output-root', help='default results/<protocol>')
    sub.choices['freeze'].add_argument('--time-series', help='processed series (default: the environment configuration)')
    r = sub.choices['run']
    r.add_argument('--split', required=True, choices=SPLITS)
    r.add_argument('--topologies', nargs='+', help='test default: every topology; val: TP1 only')
    r.add_argument('--scales', type=float, nargs='+', default=list(ALLOWED_SCALES))
    r.add_argument('--max-days', type=int)
    r.add_argument('--days', type=int, nargs='+', help='val only: positions of validation days (0-based)')
    r.add_argument('--protocol', choices=sorted(k for k, v in PROTOCOLS.items() if v in B4_FAMILY),
                   default=MAINTAINED_PROTOCOL)
    a = p.parse_args(argv)
    if a.command == 'freeze':
        path = thresholds_path(a.output_root, a.feeder)
        if path.exists():
            raise SystemExit(f'{path} exists; frozen thresholds are never recomputed or overwritten')
        record = compute_thresholds(a.feeder, a.time_series)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, record)
        print('PQ_RBC_THRESHOLDS', a.feeder, f'q30={record["q30"]:.4f}', f'q70={record["q70"]:.4f}', path, flush=True)
        return path
    a.experiment = protocol_id(a.protocol)
    try:
        a.output_root = maintained_output_root(a.protocol, a.output_root)
        a.topologies = resolve_topologies(a.feeder, a.split, a.topologies)
    except ValueError as exc:
        p.error(str(exc))
    if a.max_days is not None and a.max_days < 1:
        p.error('--max-days must be positive')
    if a.days and (a.split != 'val' or a.max_days):
        p.error('--days is for --split val and excludes --max-days')
    if a.split == 'test' and a.max_days is not None and a.max_days >= 31:
        p.error('--max-days must be below 31 (omit it for all 31 December days)')
    path, thresholds = load_thresholds(a.output_root, a.feeder)
    return [run_scale(a, check_scale(s), path, thresholds) for s in a.scales]


if __name__ == '__main__':
    main()
