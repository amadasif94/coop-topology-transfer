"""Summarize IEEE33 load-stress runs: change versus 1.0x, policy versus zero action, observation range.

    python -u -m experiments.load_stress.summarize_load_stress --root results/load_stress/ieee33 \
        --output results/load_stress/ieee33/summary_<UTC>

Read only except for the new --output directory. The 1.0x identity check is optional: each included run
records identity_check PASS / NOT_PERFORMED (inclusion.json, episode_metrics_long.csv); a model/run whose
identity report exists but is not a valid PASS is excluded. Smoke runs are never used.
Unknown flags are nullable: rates exclude unknowns and export valid/missing/invalid counts.
Power percentages pool paired finite battery decisions (five per IEEE33 row), before DOE/backup;
no raw-request reconstruction is attempted from accepted or already-limited action fields.
Coverage includes manifest-planned days with absent CSV records. Failed and incomplete counts
can overlap; absent records have unknown failure status. Missing coverage is not assumed complete:
economic totals are masked only for days known to be partial/failed; legacy records without
coverage keep their recorded totals and are labelled economic_totals_status=unverified_legacy.
Existing step flags keep their meanings. New >1 kW percentages count individual batteries.
Deltas are day-paired (same topology and start_index). Bootstrap: per run over days; per model
two-level (runs with replacement, then days within each run). Seeded, B resamples.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.no_shared_kpi.protocol import read_json, sha, write_json

KEYS = ['topology_case', 'start_index']
EPISODE_METRICS = ['total_energy_cost', 'adjusted_energy_cost', 'total_grid_import_mwh', 'feasible_rate', 'converged_rate',
                   'max_voltage_violation', 'max_line_current_violation', 'infeasible_requested_count',
                   'mean_backup_adjustment_kw', 'mean_physical_clipping_kw']
STEP_RATES = dict(backup_activation_rate='backup_activated', physical_clipping_rate='physical_clipping_any',
                  executed_voltage_violation_rate='executed_voltage_violation',
                  executed_current_violation_rate='executed_current_violation')


def as_bool(s):
    """Nullable flags: unknown is neither a success nor a failure."""
    values = {'true': True, '1': True, '1.0': True,
              'false': False, '0': False, '0.0': False}
    return s.map(lambda v: values.get(str(v).strip().lower(), pd.NA)).astype('boolean')


def discover(root, models):
    from experiments.load_stress.evaluate_load_stress_ieee33 import MODELS
    from experiments.load_stress.stress_env import ALLOWED_SCALES, scale_label
    runs, excluded = [], []
    for model in models:
        label = MODELS[model][0]
        for r in ([0] if model == 'zero' else [0, 1, 2]):
            base = Path(root) / label / f'run_{r}'
            identity = 'NOT_APPLICABLE'
            if model != 'zero':
                # Optional check: absent -> included as NOT_PERFORMED; present but not a valid PASS -> excluded.
                rep = base / scale_label(1.0) / 'identity_report.json'
                identity = 'NOT_PERFORMED'
                if rep.is_file():
                    ok = read_json(rep).get('status') == 'PASS' and \
                        read_json(rep).get('evaluation_manifest_sha256') == sha(base / scale_label(1.0) / 'run_manifest.json')
                    if not ok:
                        excluded.append(dict(model=model, run=r, reason='identity report present but not a valid PASS'))
                        continue
                    identity = 'PASS'
            for s in ALLOWED_SCALES:
                d = base / scale_label(s)
                m = d / 'run_manifest.json'
                if not m.is_file() or read_json(m).get('status') != 'finished':
                    excluded.append(dict(model=model, run=r, scale=s, reason='missing or unfinished'))
                    continue
                runs.append(dict(model=model, run=r, scale=s, dir=d, manifest_sha256=sha(m), identity_check=identity))
    return runs, excluded


COVERAGE = ['intended_episodes', 'completed_episodes', 'failed_episodes', 'incomplete_episodes']
POWER_METRICS = ['requested_power_removed_pct', 'battery_correction_gt1kw_pct']


def power_observations(requested, physical, batteries=5):
    """IEEE33: five battery decisions per row, paired before DOE/backup.

    Missing/null entries and invalid (malformed/nonfinite) entries are disjoint.
    The two sums use precisely the same valid paired observations. Zero requested
    power denominator is undefined, even if the numerator is also zero.
    """
    def parse(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return [None] * batteries
        try:
            value = json.loads(value) if isinstance(value, str) else value
            if not isinstance(value, (list, tuple)) or len(value) != batteries:
                raise ValueError('wrong battery count')
            return value
        except (ValueError, TypeError):
            return [float('inf')] * batteries
    r, p = parse(requested), parse(physical)
    out = dict(power_valid_decisions=0, power_missing_decisions=0, power_invalid_decisions=0,
               requested_abs_kw_sum=0., correction_abs_kw_sum=0., correction_gt1kw_count=0)
    for x, y in zip(r, p):
        if x is None or y is None:
            out['power_missing_decisions'] += 1
            continue
        try:
            x, y = float(x), float(y)
            if not np.isfinite([x, y]).all():
                raise ValueError('nonfinite')
        except (ValueError, TypeError):
            out['power_invalid_decisions'] += 1
            continue
        out['power_valid_decisions'] += 1
        out['requested_abs_kw_sum'] += abs(x)
        out['correction_abs_kw_sum'] += abs(x-y)
        out['correction_gt1kw_count'] += int(abs(x-y) > 1.)
    return out


def power_percentages(row):
    n, requested = row['power_valid_decisions'], row['requested_abs_kw_sum']
    return dict(requested_power_removed_pct=100*row['correction_abs_kw_sum']/requested if requested > 0 else np.nan,
                battery_correction_gt1kw_pct=100*row['correction_gt1kw_count']/n if n else np.nan)


def episode_table(item, chunksize=10000):
    ep = pd.read_csv(item['dir'] / 'episode_summary.csv')
    if ep.duplicated(KEYS).any():
        raise ValueError('Duplicate episode keys: expected one policy per load-stress run')
    flags = dict(STEP_RATES, pf_failure_rate='converged', feasible_rate='feasible', converged_rate='converged')
    wanted = KEYS + list(flags.values()) + ['marl_requested_power_kw', 'policy_requested_power_kw',
                                          'physical_limited_power_kw']
    totals = {}
    for chunk in pd.read_csv(item['dir'] / 'step_metrics.csv', usecols=lambda c: c in wanted,
                             chunksize=chunksize, low_memory=False):
        for key, group in chunk.groupby(KEYS, dropna=False):
            acc = totals.setdefault(key, {})
            for name, col in flags.items():
                raw = group[col] if col in group else pd.Series(pd.NA, index=group.index)
                known = as_bool(raw)
                if name == 'pf_failure_rate':
                    known = ~known
                stats = {name+'_true_steps': int(known.sum()), name+'_valid_steps': int(known.notna().sum()),
                         name+'_unknown_steps': int(known.isna().sum()),
                         name+'_missing_steps': int(raw.isna().sum()),
                         name+'_invalid_steps': int((known.isna() & raw.notna()).sum())}
                for k, v in stats.items():
                    acc[k] = acc.get(k, 0) + v
            for row in group.to_dict('records'):
                # Never substitute requested_action or backup-selected powers.
                requested = row.get('marl_requested_power_kw', row.get('policy_requested_power_kw'))
                for k, v in power_observations(requested, row.get('physical_limited_power_kw')).items():
                    acc[k] = acc.get(k, 0) + v
    rows = []
    for key, acc in totals.items():
        for name in flags:
            n = acc[name+'_valid_steps']
            acc[name] = acc[name+'_true_steps']/n if n else np.nan
        rows.append(dict(zip(KEYS, key), **acc, **power_percentages(acc)))
    rates = pd.DataFrame(rows)
    # Manifest plans retain intended days even if no CSV episode was emitted.
    manifest = read_json(item['dir'] / 'run_manifest.json')
    plan = [dict(topology_case=tp, start_index=e['start_index'])
            for tp, entries in manifest.get('episode_plans', {}).items() for e in entries]
    if plan:
        planned = pd.DataFrame(plan)
        if planned.duplicated(KEYS).any():
            raise ValueError('Duplicate manifest episode plan')
        if len(ep.merge(planned, on=KEYS, how='inner')) != len(ep):
            raise ValueError('Episode CSV contains unplanned days')
        ep = planned.merge(ep, on=KEYS, how='left', validate='one_to_one')
    keep = KEYS + [m for m in EPISODE_METRICS + COVERAGE if m in ep]
    out = ep[keep].drop(columns=['feasible_rate', 'converged_rate'], errors='ignore')
    out['intended_episodes'] = 1
    for c in COVERAGE[1:]:
        if c not in out:
            out[c] = np.nan  # Legacy records without coverage are not assumed complete.
    if plan:
        absent = ~pd.MultiIndex.from_frame(out[KEYS]).isin(pd.MultiIndex.from_frame(pd.read_csv(item['dir'] / 'episode_summary.csv')[KEYS]))
        out.loc[absent, ['completed_episodes', 'incomplete_episodes']] = [0, 1]
    out = out.merge(rates if len(rates) else pd.DataFrame(columns=KEYS), on=KEYS, how='left', validate='one_to_one')
    # Never publish a day known to be partial or failed. Legacy records without
    # coverage keep their recorded totals (as before) and are flagged unverified.
    valid = out.completed_episodes.eq(1) & out.failed_episodes.eq(0)
    invalid = out.completed_episodes.eq(0) | out.failed_episodes.gt(0)
    out.loc[invalid, [c for c in ('total_energy_cost', 'adjusted_energy_cost', 'total_grid_import_mwh') if c in out]] = np.nan
    out['economic_totals_status'] = np.select([valid, invalid], ['verified', 'masked_incomplete_or_failed'],
                                              'unverified_legacy')
    out['model'], out['run'], out['scale'] = item['model'], item['run'], item['scale']
    out['identity_check'] = item['identity_check']
    out['episode_plan_verified'] = bool(plan)
    return out


def coverage_table(table, by):
    rows = []
    for key, group in table.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for c in COVERAGE:
            row[c] = group[c].sum(min_count=1)
            row[c+'_unknown'] = int(group[c].isna().sum())
        for c in group:
            if c.endswith(('_steps', '_decisions')) or c in ('requested_abs_kw_sum', 'correction_abs_kw_sum', 'correction_gt1kw_count'):
                row[c] = group[c].sum(min_count=1)
        if 'power_valid_decisions' in row:
            row.update(power_percentages(row))
        for c in EPISODE_METRICS + POWER_METRICS:
            if c in group:
                row[c+'_valid_days'] = int(group[c].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=by + COVERAGE)


def with_coverage(summary, table, by, reference):
    """Counts are coverage, never deltas. Reference zero days count once, not per seed."""
    left = coverage_table(table, by)
    ref_by = [c for c in by if c not in ('model', 'run', 'scale')]
    if reference == 'nominal':
        ref_by = [c for c in by if c != 'scale']
        ref = coverage_table(table[table.scale == 1.0], ref_by)
    else:
        ref_by = [c for c in by if c not in ('model', 'run')]
        ref = coverage_table(table[table.model == 'zero'], ref_by)
    ref = ref.rename(columns={c: 'reference_'+c for c in ref if c not in ref_by})
    return summary.merge(left, on=by, how='left', validate='many_to_one').merge(ref, on=ref_by, how='left', validate='many_to_one')


def boot_run(x, rng, b):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_two_level(groups, rng, b):
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.nan, np.nan
    stats = []
    for _ in range(b):
        picks = rng.integers(0, len(groups), len(groups))
        stats.append(np.mean([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))].mean() for i in picks]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired(table, metrics, left, right, by):
    """left - right on (topology, start_index) within `by` groups."""
    l = table.query(left).set_index(by + KEYS)[metrics]
    r = table.query(right).reset_index(drop=True)
    r = r.drop(columns=[c for c in ('model', 'run', 'scale') if c not in by]).set_index(by + KEYS)[metrics]
    return (l - r.reindex(l.index)).reset_index()


def summarize_deltas(delta, metrics, by, rng, b):
    rows = []
    for key, g in delta.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            lo, hi = boot_run(g[m], deepcopy(rng) if m in POWER_METRICS else rng, b)
            rows.append(dict(zip(by, key), metric=m, n_days=int(g[m].notna().sum()), mean_delta=float(g[m].mean()),
                             ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def model_level(delta, metrics, rng, b, extra=()):
    rows = []
    for key, g in delta.groupby(['model', *extra, 'topology_case']):
        for m in metrics:
            run_means = g.groupby('run')[m].mean()
            lo, hi = boot_two_level([x[m].to_numpy() for _, x in g.groupby('run')],
                                    deepcopy(rng) if m in POWER_METRICS else rng, b)
            rows.append(dict(zip(['model', *extra, 'topology_case'], key), metric=m, runs=int(run_means.notna().sum()),
                             n_days=int(g[m].notna().sum()), mean_delta=float(run_means.mean()), between_run_sd=float(run_means.std(ddof=1)),
                             ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def ood_table(runs):
    from experiments.load_stress.evaluate_load_stress_ieee33 import OBS_FEATURES
    rows = []
    obs = {}
    for item in runs:
        p = item['dir'] / 'obs_diagnostics.csv'
        if p.is_file():
            obs[item['model'], item['run'], item['scale']] = pd.read_csv(p)
    for (model, r, s), df in obs.items():
        ref = obs.get((model, r, 1.0))
        if ref is None:
            continue
        for tp, g in df.groupby('topology_case'):
            gr = ref[ref.topology_case == tp]
            row = dict(model=model, run=r, scale=s, topology_case=tp, steps=len(g))
            for f in OBS_FEATURES:
                c = f'{f}_feature_max'
                row[f'{f}_max_at_1x'] = float(gr[c].max())
                row[f'{f}_max'] = float(g[c].max())
                row[f'{f}_frac_steps_above_1x_max'] = float((g[c] > gr[c].max()).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    from experiments.load_stress.evaluate_load_stress_ieee33 import MODELS, TOPOLOGIES
    from experiments.load_stress.stress_env import ALLOWED_SCALES
    from utility.run_manifest import reserve_directory
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=sorted(MODELS), choices=sorted(MODELS))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261027)
    a = p.parse_args(argv)
    runs, excluded = discover(a.root, a.models)
    if not runs:
        raise SystemExit('No eligible load-stress runs found under ' + str(a.root))
    out = reserve_directory(a.output)
    rng = np.random.default_rng(a.seed)
    table = pd.concat([episode_table(i) for i in runs], ignore_index=True)
    table = table[table.topology_case.isin(TOPOLOGIES)]
    metrics = [c for c in EPISODE_METRICS + ['pf_failure_rate'] + list(STEP_RATES) + POWER_METRICS if c in table]
    table.to_csv(out / 'episode_metrics_long.csv', index=False)

    coverage_table(table, ['model', 'run', 'scale', 'topology_case']).to_csv(out / 'coverage_and_power_metrics_by_run.csv', index=False)
    coverage_table(table, ['model', 'scale', 'topology_case']).to_csv(out / 'coverage_and_power_metrics_by_model.csv', index=False)
    learned = table[table.model != 'zero']
    deltas = []
    for s in ALLOWED_SCALES[1:]:
        d = paired(table, metrics, f'scale == {s}', 'scale == 1.0', ['model', 'run'])
        d['scale'] = s
        deltas.append(d)
    delta = pd.concat(deltas, ignore_index=True)  # Retain intended days even when every paired metric is missing.
    delta.to_csv(out / 'delta_vs_1x_by_day.csv', index=False)
    with_coverage(summarize_deltas(delta, metrics, ['model', 'run', 'scale', 'topology_case'], rng, a.bootstrap),
        table, ['model', 'run', 'scale', 'topology_case'], 'nominal').to_csv(out / 'delta_vs_1x_by_run.csv', index=False)
    with_coverage(model_level(delta[delta.model != 'zero'], metrics, rng, a.bootstrap, extra=('scale',)),
        table, ['model', 'scale', 'topology_case'], 'nominal').to_csv(out / 'delta_vs_1x_by_model.csv', index=False)

    zero = table[table.model == 'zero']
    if len(zero) and len(learned):
        vs = learned.merge(zero.drop(columns=['model', 'run', 'identity_check']), on=['scale', *KEYS], how='left',
                           suffixes=('', '__zero'), validate='many_to_one')
        for m in metrics:
            vs[m] = vs[m] - vs[m + '__zero']
        vs = vs[['model', 'run', 'scale', *KEYS, *metrics]]
        vs.to_csv(out / 'policy_minus_zero_by_day.csv', index=False)
        with_coverage(model_level(vs, metrics, rng, a.bootstrap, extra=('scale',)),
            table, ['model', 'scale', 'topology_case'], 'zero').to_csv(out / 'policy_minus_zero_by_model.csv', index=False)
    ood_table(runs).to_csv(out / 'obs_out_of_distribution.csv', index=False)
    write_json(out / 'inclusion.json', dict(root=str(Path(a.root).resolve()), bootstrap=a.bootstrap, seed=a.seed,
        included=[dict(i, dir=str(i['dir'])) for i in runs], excluded=excluded,
        note='Post-hoc sensitivity analysis on the December test split; never used for model choice.'))
    print('SUMMARY', out, json.dumps(dict(included=len(runs), excluded=len(excluded))), flush=True)


if __name__ == '__main__':
    main()
