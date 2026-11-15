"""Summarize IEEE69 load-stress runs (A0/A1/A2, TP1-TP10): change versus 1.0x and observation range.

    python -u -m experiments.load_stress.summarize_load_stress_ieee69 --root results/load_stress/ieee69 \
        --output results/load_stress/ieee69/summary_<UTC>

Read only except for the new --output directory. Pilot runs are never used. The 1.0x identity check is
optional: each included run records identity_check PASS / PASS_PARTIAL / NOT_PERFORMED; a model/run whose
identity report exists but is FAIL (or stale) is excluded. Without load_x100 runs, the change-vs-1.0x and
out-of-distribution tables are empty and only absolute per-scale tables are produced. Statistics reuse the
IEEE33 summarizer (day-paired deltas; per-run day bootstrap; two-level run/day bootstrap per controller).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.load_stress import summarize_load_stress as S33
from experiments.load_stress.evaluate_load_stress_ieee69 import IDENTITY_ACCEPTED, MODELS, TOPOLOGIES, run_dir
from experiments.load_stress.stress_env import ALLOWED_SCALES
from experiments.no_shared_kpi.protocol import read_json, sha, write_json
from utility.run_manifest import reserve_directory

KEYS = S33.KEYS
EPISODE_METRICS = ['total_energy_cost', 'adjusted_energy_cost', 'total_grid_import_mwh', 'mean_voltage_deviation',
                   'max_voltage_deviation', 'min_voltage_pu', 'max_voltage_pu', 'max_line_current_pu',
                   'max_voltage_violation', 'max_line_current_violation', 'feasible_rate', 'converged_rate',
                   'infeasible_requested_count', 'mean_backup_adjustment_kw', 'mean_physical_clipping_kw']
STEP_RATES = dict(S33.STEP_RATES, major_infeasible_rate='major_infeasible_action',
                  minor_correction_rate='minor_corrected_action', correction_fallback_rate='correction_fallback_to_zero',
                  episode_failure_rate='episode_failed')
STEP_MEANS = dict(mean_correction_iterations='correction_iterations', mean_backup_pf_evaluations='backup_pf_evaluations')


def discover(root):
    runs, excluded = [], []
    for model in sorted(MODELS):
        for r in (0, 1, 2):
            base = run_dir(root, model, r, 1.0)
            identity = 'NOT_PERFORMED'
            rep = base / 'identity_report.json'
            if rep.is_file():
                report = read_json(rep)
                if report.get('status') not in IDENTITY_ACCEPTED or \
                        report.get('evaluation_manifest_sha256') != sha(base / 'run_manifest.json'):
                    excluded.append(dict(model=model, run=r, reason='identity report present but not a valid PASS'))
                    continue
                identity = report['status']
            for s in ALLOWED_SCALES:
                d = run_dir(root, model, r, s)
                m = d / 'run_manifest.json'
                if not m.is_file() or read_json(m).get('status') != 'finished':
                    excluded.append(dict(model=model, run=r, scale=s, reason='missing or unfinished'))
                    continue
                runs.append(dict(model=model, run=r, scale=s, dir=d, manifest_sha256=sha(m), identity_check=identity))
    return runs, excluded


def episode_table(item):
    ep = pd.read_csv(item['dir'] / 'episode_summary.csv')
    wanted = set(KEYS) | {'converged'} | set(STEP_RATES.values()) | set(STEP_MEANS.values())
    steps = pd.read_csv(item['dir'] / 'step_metrics.csv', usecols=lambda c: c in wanted, low_memory=False)
    rates = steps[KEYS].copy()
    rates['pf_failure_rate'] = ~S33.as_bool(steps['converged'])
    for name, col in STEP_RATES.items():
        rates[name] = S33.as_bool(steps[col]) if col in steps else np.nan
    for name, col in STEP_MEANS.items():
        rates[name] = pd.to_numeric(steps[col], errors='coerce') if col in steps else np.nan
    # Nullable flags: unknown steps are excluded from each rate (not counted as False);
    # plain float keeps all-unknown days as NaN for the bootstrap and run statistics.
    rates = rates.groupby(KEYS).mean().astype(float).reset_index()
    out = ep[KEYS + [m for m in EPISODE_METRICS if m in ep]].merge(rates, on=KEYS, validate='one_to_one')
    out['model'], out['run'], out['scale'], out['identity_check'] = item['model'], item['run'], item['scale'], item['identity_check']
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261027)
    a = p.parse_args(argv)
    runs, excluded = discover(a.root)
    if not runs:
        raise SystemExit('No eligible IEEE69 load-stress runs found under ' + str(a.root))
    out = reserve_directory(a.output)
    rng = np.random.default_rng(a.seed)
    table = pd.concat([episode_table(i) for i in runs], ignore_index=True)
    table = table[table.topology_case.isin(TOPOLOGIES)]
    metrics = [c for c in table.columns if c not in ('model', 'run', 'scale', 'identity_check', *KEYS)]
    table.to_csv(out / 'episode_metrics_long.csv', index=False)

    # Absolute values per scale (available without the optional 1.0x runs).
    S33.summarize_deltas(table, metrics, ['model', 'run', 'scale', 'topology_case'], rng, a.bootstrap) \
        .rename(columns={'mean_delta': 'mean'}).to_csv(out / 'absolute_by_run.csv', index=False)
    absolute_by_model = []
    for (model, scale, tp), g in table.groupby(['model', 'scale', 'topology_case']):
        for m in metrics:
            run_means = g.groupby('run')[m].mean()
            lo, hi = S33.boot_two_level([x[m].to_numpy() for _, x in g.groupby('run')], rng, a.bootstrap)
            absolute_by_model.append(dict(model=model, scale=scale, topology_case=tp, metric=m, runs=int(run_means.notna().sum()),
                                          mean=float(run_means.mean()), between_run_sd=float(run_means.std(ddof=1)),
                                          ci95_low=lo, ci95_high=hi))
    pd.DataFrame(absolute_by_model).to_csv(out / 'absolute_by_model.csv', index=False)

    deltas = []
    if (table.scale == 1.0).any():
        for s in ALLOWED_SCALES[1:]:
            d = S33.paired(table, metrics, f'scale == {s}', 'scale == 1.0', ['model', 'run'])
            d['scale'] = s
            deltas.append(d)
    delta = pd.concat(deltas, ignore_index=True).dropna(subset=metrics, how='all') if deltas else pd.DataFrame()
    delta.to_csv(out / 'delta_vs_1x_by_day.csv', index=False)
    if len(delta):
        S33.summarize_deltas(delta, metrics, ['model', 'run', 'scale', 'topology_case'], rng, a.bootstrap) \
            .to_csv(out / 'delta_vs_1x_by_run.csv', index=False)
        S33.model_level(delta, metrics, rng, a.bootstrap, extra=('scale',)).to_csv(out / 'delta_vs_1x_by_model.csv', index=False)
    S33.ood_table(runs).to_csv(out / 'obs_out_of_distribution.csv', index=False)
    expected = {(m, r, s) for m in MODELS for r in (0, 1, 2) for s in ALLOWED_SCALES[1:]}
    present = {(i['model'], i['run'], i['scale']) for i in runs}
    write_json(out / 'inclusion.json', dict(root=str(Path(a.root).resolve()), bootstrap=a.bootstrap, seed=a.seed,
        included=[dict(i, dir=str(i['dir'])) for i in runs], excluded=excluded,
        missing_stress_runs=sorted([list(x) for x in expected - present]),
        stress_episodes=int(len(table[table.scale != 1.0])),
        note='Post-hoc sensitivity analysis on the December test split; never used for model choice.'))
    print('SUMMARY', out, json.dumps(dict(included=len(runs), excluded=len(excluded),
          stress_episodes=int(len(table[table.scale != 1.0])), expected_stress_episodes=8370)), flush=True)


if __name__ == '__main__':
    main()
