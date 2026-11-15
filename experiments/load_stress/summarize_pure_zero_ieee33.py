"""Summarize the IEEE33 pure zero-action reference (0 kW, no backup) per load scale and topology.

    python -u -m experiments.load_stress.summarize_pure_zero_ieee33 \
        --root results/load_stress/ieee33/pure_zero_no_backup --output <new dir> \
        [--zero-backup-root results/load_stress/ieee33]

Read only except for the new --output directory. Pilot runs are never used. Day-level values; 95% CIs by
day bootstrap (one run: the reference does not depend on training). Failed days are counted, never dropped.
Outputs:
  episode_metrics_long.csv         one row per scale x topology x day
  absolute_by_scale.csv            mean and day-bootstrap CI per scale x topology x metric
  zero_backup_minus_pure_by_day.csv / _summary.csv
                                   existing zero-action + backup run minus pure zero, same day (only where both exist)
  inclusion.json                   included / missing scales, failed days, cross-check status per scale
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.load_stress import summarize_load_stress as S33
from experiments.load_stress.evaluate_load_stress_ieee33 import run_dir as stress_run_dir
from experiments.load_stress.evaluate_pure_zero_ieee33 import DAYS, TOPOLOGIES, run_dir
from experiments.load_stress.pure_zero_env import LABEL
from experiments.load_stress.stress_env import ALLOWED_SCALES
from experiments.no_shared_kpi.protocol import read_json, sha, write_json
from utility.run_manifest import reserve_directory

KEYS = S33.KEYS
EPISODE_METRICS = ['total_energy_cost', 'adjusted_energy_cost', 'raw_energy_cost', 'total_grid_import_mwh',
                   'mean_voltage_deviation', 'max_voltage_deviation', 'min_voltage_pu', 'max_voltage_pu',
                   'max_line_current_pu', 'max_voltage_violation', 'max_line_current_violation', 'feasible_rate',
                   'converged_rate', 'total_curtailment_mwh']
STEP_RATES = dict(violating_step_rate='executed_network_violation', voltage_violation_step_rate='executed_voltage_violation',
                  current_violation_step_rate='executed_current_violation', episode_failure_rate='episode_failed')
STEP_SUMS = dict(grid_import_mwh='grid_import_mwh', energy_cost='energy_cost', curtailment_mwh='curtailment_mwh')
STEP_MEANS = dict(mean_voltage_deviation_step='voltage_deviation', mean_min_voltage_pu='min_voltage_pu',
                  mean_max_line_current_pu='max_line_current_pu')


def episode_table(d, scale, label):
    ep = pd.read_csv(Path(d) / 'episode_summary.csv')
    wanted = {*KEYS, 'converged', 'record_type', *STEP_RATES.values(), *STEP_SUMS.values(), *STEP_MEANS.values()}
    steps = pd.read_csv(Path(d) / 'step_metrics.csv', usecols=lambda c: c in wanted, low_memory=False)
    if 'record_type' in steps:
        steps = steps[steps['record_type'] == 'dispatch_attempt']
    rates = steps[KEYS].copy()
    rates['pf_failure_rate'] = ~S33.as_bool(steps['converged'])
    for name, col in STEP_RATES.items():
        rates[name] = S33.as_bool(steps[col]) if col in steps else np.nan
    for name, col in {**STEP_SUMS, **STEP_MEANS}.items():
        rates[name] = pd.to_numeric(steps[col], errors='coerce') if col in steps else np.nan
    g = rates.groupby(KEYS)
    rates = pd.concat([g[[*STEP_RATES, 'pf_failure_rate', *STEP_MEANS]].mean().astype(float),  # unknown flags excluded
                       g[list(STEP_SUMS)].sum(min_count=1).add_prefix('step_sum_'),
                       g.size().rename('dispatch_steps')], axis=1).reset_index()
    out = ep[KEYS + [m for m in EPISODE_METRICS + ['complete'] if m in ep]].merge(rates, on=KEYS, validate='one_to_one')
    out['failed_day'] = ~out.pop('complete').astype(bool) if 'complete' in out else np.nan
    out['scale'], out['source'] = scale, label
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--zero-backup-root', type=Path, default=Path('results/load_stress/ieee33'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261027)
    a = p.parse_args(argv)
    included, missing, tables, backup_tables = [], [], [], []
    for s in ALLOWED_SCALES:
        d = run_dir(a.root, s)
        m = d / 'run_manifest.json'
        if not m.is_file() or read_json(m).get('status') != 'finished':
            missing.append(s)
            continue
        cross = d / 'cross_check_report.json'
        info = read_json(m)['pure_zero']
        included.append(dict(scale=s, dir=str(d), manifest_sha256=sha(m), episodes=info.get('episodes'),
                             cross_check=read_json(cross).get('status') if cross.is_file() else 'NOT_PERFORMED'))
        tables.append(episode_table(d, s, LABEL))
        zb = stress_run_dir(a.zero_backup_root, 'zero', 0, s)
        if (zb / 'run_manifest.json').is_file() and read_json(zb / 'run_manifest.json').get('status') == 'finished':
            backup_tables.append(episode_table(zb, s, 'zero_action_backup'))
    if not tables:
        raise SystemExit('No finished pure zero runs under ' + str(a.root))
    out = reserve_directory(a.output)
    rng = np.random.default_rng(a.seed)
    table = pd.concat(tables, ignore_index=True)
    table = table[table.topology_case.isin(TOPOLOGIES)]
    table['failed_day'] = table['failed_day'].astype(float)
    metrics = [c for c in table.columns if c not in ('scale', 'source', *KEYS)]
    table.to_csv(out / 'episode_metrics_long.csv', index=False)
    S33.summarize_deltas(table, metrics, ['scale', 'topology_case'], rng, a.bootstrap) \
        .rename(columns={'mean_delta': 'mean'}).to_csv(out / 'absolute_by_scale.csv', index=False)
    if backup_tables:
        both = pd.concat([table, *backup_tables], ignore_index=True)
        both['failed_day'] = both['failed_day'].astype(float)
        both['model'], both['run'] = both['source'], 0
        delta = S33.paired(both, metrics, "source == 'zero_action_backup'", f"source == '{LABEL}'", ['scale'])
        delta = delta.dropna(subset=metrics, how='all')
        delta.to_csv(out / 'zero_backup_minus_pure_by_day.csv', index=False)
        S33.summarize_deltas(delta, metrics, ['scale', 'topology_case'], rng, a.bootstrap) \
            .to_csv(out / 'zero_backup_minus_pure_summary.csv', index=False)
    expected = len(ALLOWED_SCALES) * len(TOPOLOGIES) * DAYS
    write_json(out / 'inclusion.json', dict(root=str(Path(a.root).resolve()), bootstrap=a.bootstrap, seed=a.seed,
        included=included, missing_scales=missing, episodes=int(len(table)), expected_episodes=expected,
        failed_days=int(table['failed_day'].sum()),
        zero_backup_scales=sorted({float(t['scale'].iloc[0]) for t in backup_tables}),
        note='Pure zero-action reference (0 kW, no backup): not a controller; executed dispatch may violate limits.'))
    print('SUMMARY', out, json.dumps(dict(scales=len(included), episodes=int(len(table)), expected=expected,
          failed_days=int(table['failed_day'].sum()))), flush=True)


if __name__ == '__main__':
    main()
