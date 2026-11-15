"""Task 1: paired calendar-day block bootstrap of December 2018 controller comparisons. Maintained entry point.

    python -u -m evaluation.final_analyses.bootstrap --feeder ieee33 [--results-root results/soc_feasible_affine_v2]
                                                    [--out-root results/final_paper_analyses] [--resamples 10000]

Data: episode_summary.csv of every finished authoritative run (common.load_episodes): one row per (controller, seed,
load factor, topology, December day). Rows are matched on (topology, load factor, start_index); start_index identifies
the calendar day of the environment's series (its date is recorded), never the row position. Missing or non-finite
values drop that date for that comparison (counted in n_dates_dropped); nothing is imputed.

Seed handling (mandatory procedure):
  primary    for each RL controller, the arithmetic mean over the three training seeds per (date, topology, load
             factor); a key without all three seeds is dropped and counted. PF-MPC, F-MPC, PQ-RBC and the zero-action
             reference have ONE outcome per key and are used as they are (never duplicated).
  per_seed   the same comparisons for each seed k separately: an RL controller's seed k against the other RL
             controller's seed k (seed indices are paired as labels of independent runs; they share only the RNG seed
             42 + k), or against the single deterministic outcome.
Paired difference per date: d = Y_A - Y_B (A minus B, as named). Feeder scope: d_date = mean over all evaluated
topologies of (Y_A - Y_B) at that date, so a date is resampled with all its topologies (block); a date missing in any
topology is dropped. Topology scope: one topology.
Uncertainty: percentile bootstrap over the n December dates (B >= 10,000 resamples of date indices with replacement,
fixed RNG seed; the same index matrix for every comparison over the same date set). The intervals quantify variation
across evaluation days CONDITIONAL on the three trained RL seeds; they do not quantify uncertainty over hypothetical
future training seeds (see seed_variability.csv for between-seed spread). Deterministic references carry no run-to-run
uncertainty of their own.
p-values: NOT taken from the bootstrap distribution. A paired sign-flip randomization test on the date-level differences
(H0: the date-level differences are symmetric about zero; R = 10,000 random sign vectors, fixed seed; p = (1 + #{|mean
flipped| >= |mean observed|}) / (R + 1)). Holm correction is applied within prespecified families:
  primary          feeder scope, load factor 1.0, adjusted energy cost, the prespecified comparisons; one family per feeder
  secondary        feeder scope, one family per (feeder, load factor, metric) for every other load factor / metric
  per_topology     one family per (feeder, load factor, metric) over comparisons x topologies
  transfer         one family per (feeder, load factor, metric) over RL controllers x unseen topologies (TPk - TP1)
  per_seed         one family per (feeder, load factor, metric, seed); transfer_per_seed likewise
Holm's m counts only comparisons with a valid p-value (family_tests). A metric a controller does not record (the pure
zero-action reference has no terminal SoC shortfall; common.OPTIONAL_METRIC_COLUMNS) makes those comparisons 'not
estimable' (not_estimable_reason); they carry no p-value and do not enter any family. Every other column, and every
controller, seed, load factor, topology and December day, is required (the run stops otherwise).
F-MPC = the Gurobi results (--fmpc-backend gurobi, default): a folder whose fmpc_solves.csv does not show Gurobi on all
96 decisions of every episode stops the run (common.fmpc_backend_problems); inclusion.json lists the verified folders.
Effect sizes are always reported: mean and median paired difference, relative difference (% of B's mean), Cohen's dz
(mean / SD of date differences) and the fraction of dates on which A is better. A confidence interval containing zero is
not evidence of equivalence; non-inferiority is evaluated only for a margin given with --ni-margin (prespecified).
Outputs (new folder, never overwritten): <out-root>/bootstrap/<feeder>/<UTC>/ comparisons.csv, primary_table.csv,
transfer.csv, per_seed.csv, seed_variability.csv, inclusion.json, METHODS.md.
"""
import argparse
import zlib

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, TOPOLOGIES, write_json
from evaluation.evaluate import ZERO_LABEL
from evaluation.final_analyses.common import (CONTROLLERS, DISPLAY, FMPC_BACKEND, FMPC_BACKENDS,
                                              OPTIONAL_METRIC_COLUMNS, OUT_ROOT, RESULTS_ROOT, RL, SCALES, SEEDS,
                                              coverage_problems, guard_output, load_episodes, stamp)
from utility.run_manifest import reserve_directory

METRICS = dict(total_reward=('total_reward_mean', 'higher'), raw_energy_cost=('raw_energy_cost', 'lower'),
               adjusted_energy_cost=('adjusted_energy_cost', 'lower'),
               ac_feasible_rate=('final_network_feasibility_rate', 'higher'),
               voltage_deviation=('mean_voltage_deviation', 'lower'),
               terminal_soc_shortfall=('b3_terminal_shortfall_mean', 'lower'))
TRANSFER_METRICS = ('adjusted_energy_cost', 'savings_vs_zero', 'ac_feasible_rate', 'total_reward')
DERIVED = dict(savings_vs_zero='higher')            # S = C_zero - C_controller (adjusted cost, same convention)
COMPARISONS = (('mlp_a0', 'mlp_a1'), ('mlp_a0', 'mlp_a2'), ('mlp_a0', 'mlp_a3'), ('mlp_a1', 'mlp_a3'),
               ('mlp_a2', 'mlp_a3'), ('mlp_a0', 'pf_mpc'), ('mlp_a0', 'forecast_mpc'), ('mlp_a0', 'price_quantile_rbc'),
               ('mlp_a0', ZERO_LABEL))
PRIMARY_SCALE, PRIMARY_METRIC = 1.0, 'adjusted_energy_cost'
RNG_SEED = 20261110
KEYS = ['topology_case', 'scale', 'start_index', 'date']


def direction(metric):
    return DERIVED.get(metric) or METRICS[metric][1]


def values(units, controller, metric, seed=None):
    """[KEYS + value] of one controller and metric: seed mean (seed=None) or one seed. RL keys without all three
    seeds are dropped in the seed mean; deterministic controllers ignore `seed`."""
    if metric == 'savings_vs_zero':
        c = values(units, controller, 'adjusted_energy_cost', seed)
        z = values(units, ZERO_LABEL, 'adjusted_energy_cost')
        m = c.merge(z, on=KEYS, suffixes=('_c', '_z'), how='inner', validate='one_to_one')
        return m.assign(value=m.value_z - m.value_c)[KEYS + ['value']]
    col = METRICS[metric][0]
    u = units[units.controller == controller]
    if controller in RL:
        if seed is not None:
            u = u[u.seed == float(seed)]
            return u[KEYS].assign(value=u[col].to_numpy())
        g = u.groupby(KEYS)[col]
        out = pd.DataFrame(dict(value=g.mean(), n=g.count())).reset_index()
        return out[out.n == len(SEEDS)][KEYS + ['value']]
    return u[KEYS].assign(value=u[col].to_numpy())


class Resampler:
    """Common random numbers: one bootstrap index matrix and one sign-flip matrix per date set."""

    def __init__(self, resamples=10000, flips=10000, seed=RNG_SEED):
        self.B, self.R, self.seed, self.cache = int(resamples), int(flips), int(seed), {}

    def matrices(self, dates):
        key = tuple(dates)
        if key not in self.cache:
            rng = np.random.default_rng([self.seed, zlib.crc32(','.join(key).encode())])
            n = len(key)
            self.cache[key] = (rng.integers(0, n, size=(self.B, n)), rng.choice((-1.0, 1.0), size=(self.R, n)))
        return self.cache[key]


def paired_stats(d, dates, resampler, better):
    """Date-level paired differences d (A - B) -> effect sizes, percentile bootstrap CI and sign-flip p-value."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    out = dict(n_dates=n, mean_diff=np.nan, ci_low=np.nan, ci_high=np.nan, median_diff=np.nan, cohens_dz=np.nan,
               frac_dates_a_better=np.nan, p_signflip=np.nan)
    if n < 2:
        return out
    idx, flips = resampler.matrices(dates)
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = abs(d.mean())
    perm = np.abs((flips * d[None, :]).mean(axis=1))
    sd = d.std(ddof=1)
    favour = d < 0 if better == 'lower' else d > 0
    out.update(mean_diff=float(d.mean()), ci_low=float(lo), ci_high=float(hi), median_diff=float(np.median(d)),
               cohens_dz=float(d.mean() / sd) if sd > 0 else np.nan, frac_dates_a_better=float(favour.mean()),
               p_signflip=float((1 + np.sum(perm >= obs - 1e-12)) / (len(perm) + 1)))
    return out


def date_differences(a, b, scope, topologies):
    """Per-date paired differences A - B. scope 'feeder': mean over `topologies` of (A - B) per date (dates missing
    any topology dropped); otherwise the single topology `scope`."""
    m = a.merge(b, on=KEYS, suffixes=('_a', '_b'), how='inner', validate='one_to_one')
    m = m[np.isfinite(m.value_a) & np.isfinite(m.value_b)]
    if scope != 'feeder':
        m = m[m.topology_case == scope]
        m = m.sort_values('date')
        return m.date.tolist(), (m.value_a - m.value_b).to_numpy(), m.value_a.mean(), m.value_b.mean(), 0
    m = m[m.topology_case.isin(topologies)]
    counts = m.groupby('date').topology_case.nunique()
    complete = counts[counts == len(topologies)].index
    dropped = int(len(counts) - len(complete))
    m = m[m.date.isin(complete)]
    g = m.assign(diff=m.value_a - m.value_b).groupby('date')
    per = g[['diff', 'value_a', 'value_b']].mean().sort_index()
    return per.index.tolist(), per['diff'].to_numpy(), per.value_a.mean(), per.value_b.mean(), dropped


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def interpret(row, better):
    if not np.isfinite(row['ci_low']):
        return 'not estimable'
    if row['ci_low'] <= 0 <= row['ci_high']:
        return 'CI includes 0 (no detectable difference; not evidence of equivalence)'
    a_better = row['ci_high'] < 0 if better == 'lower' else row['ci_low'] > 0
    return 'first better (CI excludes 0)' if a_better else 'second better (CI excludes 0)'


def unavailable_reason(units, metric, *controllers):
    """Why a comparison has no estimate: a metric column a controller does not record (common.OPTIONAL_METRIC_COLUMNS),
    or too few paired dates."""
    col = METRICS.get(metric, (None,))[0]
    for c in controllers:
        if col in OPTIONAL_METRIC_COLUMNS.get(c, ()) and units.loc[units.controller == c, col].isna().all():
            return f'{col} is not recorded for {c}'
    return 'fewer than 2 paired dates'


def compare(units, feeder, resampler, scales, comparisons=COMPARISONS, metrics=tuple(METRICS), seed=None,
            scopes=('feeder', 'topology')):
    rows = []
    topologies = sorted(set(units.topology_case), key=lambda t: int(t[2:]))
    for scale in scales:
        u = units[units.scale == scale]
        for metric in metrics:
            better = direction(metric)
            for a_name, b_name in comparisons:
                if seed is not None and a_name not in RL and b_name not in RL:
                    continue
                a = values(u, a_name, metric, seed if a_name in RL else None)
                b = values(u, b_name, metric, seed if b_name in RL else None)
                for scope in [s for s in scopes if s == 'feeder'] + ([*topologies] if 'topology' in scopes else []):
                    dates, d, ma, mb, dropped = date_differences(a, b, scope, topologies)
                    r = dict(feeder=feeder, scale=scale, metric=metric, metric_column=METRICS.get(metric, ('derived',))[0],
                             better_direction=better, scope='feeder' if scope == 'feeder' else 'topology',
                             topology='all' if scope == 'feeder' else scope, comparison=f'{DISPLAY[a_name]} - {DISPLAY[b_name]}',
                             controller_a=a_name, controller_b=b_name, difference='A - B',
                             seed='mean_of_3' if seed is None else int(seed),
                             seeds_a=(len(SEEDS) if seed is None else 1) if a_name in RL else 'deterministic',
                             seeds_b=(len(SEEDS) if seed is None else 1) if b_name in RL else 'deterministic',
                             n_topologies=len(topologies) if scope == 'feeder' else 1, n_dates_dropped=dropped,
                             mean_a=ma, mean_b=mb, **paired_stats(d, dates, resampler, better))
                    r['rel_diff_pct_of_b'] = 100 * r['mean_diff'] / abs(mb) if mb and np.isfinite(mb) else np.nan
                    r['interpretation'] = interpret(r, better)
                    r['not_estimable_reason'] = unavailable_reason(u, metric, a_name, b_name) \
                        if not np.isfinite(r['mean_diff']) else ''
                    rows.append(r)
    return pd.DataFrame(rows)


def transfer(units, feeder, resampler, scales, seed=None, ni_margins=None):
    """TPk - TP1 for every RL controller and unseen topology k, same dates (and seed when per seed)."""
    rows = []
    unseen = [t for t in TOPOLOGIES[feeder] if t != 'TP1' and t in set(units.topology_case)]
    for scale in scales:
        u = units[units.scale == scale]
        for metric in TRANSFER_METRICS:
            better = direction(metric)
            for c in RL:
                v = values(u, c, metric, seed)
                base = v[v.topology_case == 'TP1'].drop(columns='topology_case')
                for tp in unseen:
                    k = v[v.topology_case == tp].drop(columns='topology_case')
                    m = k.merge(base, on=['scale', 'start_index', 'date'], suffixes=('_k', '_1'), validate='one_to_one')
                    m = m[np.isfinite(m.value_k) & np.isfinite(m.value_1)].sort_values('date')
                    d = (m.value_k - m.value_1).to_numpy()
                    r = dict(feeder=feeder, scale=scale, metric=metric, better_direction=better, controller=c,
                             display=DISPLAY[c], topology=tp, difference=f'{tp} - TP1',
                             seed='mean_of_3' if seed is None else int(seed), mean_tpk=m.value_k.mean(),
                             mean_tp1=m.value_1.mean(), **paired_stats(d, m.date.tolist(), resampler, better))
                    degradation_positive = better == 'lower'
                    r['degradation_detectable'] = (bool(r['ci_low'] > 0) if degradation_positive else bool(r['ci_high'] < 0)) \
                        if np.isfinite(r['ci_low']) else None
                    margin = (ni_margins or {}).get(metric)
                    if margin is not None and np.isfinite(r['ci_low']):
                        r['ni_margin'] = margin
                        r['non_inferior'] = bool(r['ci_high'] < margin) if degradation_positive else bool(r['ci_low'] > -margin)
                    rows.append(r)
    return pd.DataFrame(rows)


def assign_families(df, kind):
    if df.empty:
        return df
    if kind == 'compare':
        def fam(r):
            if r.scope == 'topology':
                return f'per_topology|{r.feeder}|x{r.scale}|{r.metric}'
            if r.seed != 'mean_of_3':
                return f'per_seed|{r.feeder}|x{r.scale}|{r.metric}|seed{r.seed}'
            if r.scale == PRIMARY_SCALE and r.metric == PRIMARY_METRIC:
                return f'primary|{r.feeder}'
            return f'secondary|{r.feeder}|x{r.scale}|{r.metric}'
    else:
        def fam(r):
            return (f'transfer|{r.feeder}|x{r.scale}|{r.metric}' if r.seed == 'mean_of_3'
                    else f'transfer_per_seed|{r.feeder}|x{r.scale}|{r.metric}|seed{r.seed}')
    df = df.copy()
    df['family'] = df.apply(fam, axis=1)
    df['family_role'] = df.family.str.split('|').str[0]
    df['p_holm'] = np.nan
    for _, idx in df.groupby('family').groups.items():
        df.loc[idx, 'p_holm'] = holm(df.loc[idx, 'p_signflip'].to_numpy())
    df['family_size'] = df.groupby('family').family.transform('size')
    df['family_tests'] = df.groupby('family').p_signflip.transform(lambda p: int(np.isfinite(p).sum()))  # Holm m
    return df


def seed_variability(units, feeder, scales):
    rows = []
    for scale in scales:
        u = units[units.scale == scale]
        for c in RL:
            for metric, (col, better) in METRICS.items():
                per = u[u.controller == c].groupby('seed')[col].mean()
                if per.empty:
                    continue
                rows.append(dict(feeder=feeder, scale=scale, controller=c, display=DISPLAY[c], metric=metric,
                                 better_direction=better, **{f'seed_{int(k)}_mean': v for k, v in per.items()},
                                 between_seed_mean=per.mean(), between_seed_sd=per.std(ddof=1) if len(per) > 1 else np.nan,
                                 n_seeds=int(per.size)))
    return pd.DataFrame(rows)


METHODS = __doc__


def run(feeder, results_root, out_root, resamples, flips, scales, ni_margins=None, time_series=None,
        allow_incomplete=False, fmpc_backend=FMPC_BACKEND):
    out_base = guard_output(out_root, results_root)
    columns = [c for c, _ in METRICS.values()]
    units, info = load_episodes(results_root, feeder, columns, scales=scales, time_series=time_series,
                                fmpc_backend=fmpc_backend)
    problems = coverage_problems(units, feeder, scales, CONTROLLERS, info)
    for p in problems:
        print('FINAL_BOOTSTRAP_MISSING', p, flush=True)
    if problems and not allow_incomplete:
        raise SystemExit(f'{len(problems)} missing prerequisites (listed above); the primary families would change. '
                         'Finish those runs, or pass --allow-incomplete for a provisional look.')
    info['coverage_problems'] = problems
    rs = Resampler(resamples, flips)
    main = compare(units, feeder, rs, scales)
    seeds = pd.concat([compare(units, feeder, rs, scales, seed=k, scopes=('feeder',)) for k in SEEDS], ignore_index=True)
    allc = assign_families(pd.concat([main, seeds], ignore_index=True), 'compare')
    tr = assign_families(pd.concat([transfer(units, feeder, rs, scales, None, ni_margins)] +
                                   [transfer(units, feeder, rs, scales, k, ni_margins) for k in SEEDS],
                                   ignore_index=True), 'transfer')
    out = reserve_directory(out_base / 'bootstrap' / feeder / stamp())
    allc.to_csv(out / 'comparisons.csv', index=False)
    allc[allc.family_role == 'primary'].to_csv(out / 'primary_table.csv', index=False)
    allc[allc.family_role == 'per_seed'].to_csv(out / 'per_seed.csv', index=False)
    tr.to_csv(out / 'transfer.csv', index=False)
    seed_variability(units, feeder, scales).to_csv(out / 'seed_variability.csv', index=False)
    counts = units.groupby(['controller', 'scale']).agg(rows=('start_index', 'size'),
                                                        seeds=('seed', 'nunique'), dates=('date', 'nunique'),
                                                        topologies=('topology_case', 'nunique')).reset_index()
    write_json(out / 'inclusion.json', dict(feeder=feeder, results_root=str(results_root), resamples=resamples,
                                            sign_flips=flips, rng_seed=RNG_SEED, scales=list(scales),
                                            ni_margins=ni_margins or {}, coverage=counts.to_dict('records'), **info))
    (out / 'METHODS.md').write_text('# Paired calendar-day bootstrap\n\n```\n' + METHODS + '\n```\n', encoding='utf-8')
    print('FINAL_BOOTSTRAP_DONE', feeder, out, f'comparisons={len(allc)}', f'transfer={len(tr)}', flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--results-root', default=RESULTS_ROOT)
    p.add_argument('--out-root', default=OUT_ROOT)
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--sign-flips', type=int, default=10000)
    p.add_argument('--scales', type=float, nargs='+', default=list(SCALES))
    p.add_argument('--ni-margin', nargs='+', default=[], metavar='METRIC=VALUE',
                   help='prespecified non-inferiority margins for the transfer analysis (absolute units)')
    p.add_argument('--time-series', help='processed series (default: the environment configuration)')
    p.add_argument('--allow-incomplete', action='store_true',
                   help='run despite missing controllers/seeds/topologies/days/columns (provisional; recorded)')
    p.add_argument('--fmpc-backend', choices=FMPC_BACKENDS, default=FMPC_BACKEND,
                   help='solver every F-MPC decision must show in fmpc_solves.csv (default gurobi; highs only to '
                        'reproduce the superseded HiGHS analysis)')
    a = p.parse_args(argv)
    if a.resamples < 10000 or a.sign_flips < 10000:
        p.error('use at least 10,000 bootstrap resamples and sign flips')
    margins = {}
    for item in a.ni_margin:
        k, _, v = item.partition('=')
        if k not in TRANSFER_METRICS:
            p.error(f'--ni-margin metric must be one of {TRANSFER_METRICS}')
        margins[k] = float(v)
    return run(a.feeder, a.results_root, a.out_root, a.resamples, a.sign_flips, a.scales, margins, a.time_series,
               a.allow_incomplete, a.fmpc_backend)


if __name__ == '__main__':
    main()
