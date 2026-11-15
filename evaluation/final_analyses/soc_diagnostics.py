"""Task 3: per-battery SoC depletion diagnostics of A3 (primary) and A0 (reference). Maintained entry point.

    python -u -m evaluation.final_analyses.soc_diagnostics --feeder ieee33 [--results-root ...] [--out-root ...]

No evaluation is rerun: the authoritative test step tables (step_metrics.csv) already hold, for every battery i, the
post-step SoC (soc_agent_<i>), the executed battery power (executed_battery_power_kw, JSON per step) and the terminal
shortfall (marl_terminal_soc_shortfall on the last step of a completed day). Battery parameters (SoC minimum, capacity,
bus) come from the environment's own battery models.
Definitions (per battery, per December day; days are separate episodes, streaks never cross midnight):
  near-minimum         post-step SoC <= soc_min_i + SOC_NEAR, SOC_NEAR = utility.feasibility_metrics.SOC_NEAR (the
                       existing B3 definition: 0.001 = 0.1 percentage point; not a new threshold)
  near_min_fraction    share of the day's 96 intervals that are near-minimum
  longest_streak       longest run of consecutive near-minimum intervals (intervals and hours, 15 min each)
  near_min_episodes    number of separate near-minimum runs in the day
  min_soc / mean_soc / end_soc   over the day's post-step SoC (start SoC reported separately)
  terminal_shortfall   environments.battery_feasibility.terminal_shortfall as recorded by the environment:
                       max(0, s_0 - s_T) / (s_0 - s_min), target = the start-of-day SoC (completed days only)
  charge/discharge_throughput_kwh  executed (post-backup) battery energy
A0 versus A3 per battery: paired by (load factor, topology, date, battery); primary = mean over the three seeds of each
controller, per-seed secondary = seed k vs seed k; calendar-day block bootstrap as in bootstrap.py (dates resampled
with all topologies together). Nothing about A3 is assumed: the tables report what the data show.
Outputs: <out-root>/soc_diagnostics/<feeder>/<UTC>/ battery_day.csv, seed_summary.csv, a0_vs_a3_battery.csv,
streak_distribution.csv, soc_profile.csv (time-of-day mean/p10/p90 per battery: figure-ready), inclusion.json.
"""
import argparse

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, write_json
from evaluation.final_analyses.bootstrap import Resampler, holm, paired_stats
from evaluation.final_analyses.common import (DISPLAY, OUT_ROOT, RESULTS_ROOT, SCALES, SEEDS, STEPS, guard_output,
                                              json_list, run_dirs, series_dates, stamp)
from utility.feasibility_metrics import SOC_NEAR
from utility.run_manifest import reserve_directory

CONTROLLERS = ('mlp_a3', 'mlp_a0')
DT_HOURS = 0.25
PAIRED_METRICS = dict(near_min_fraction='lower', longest_streak_hours='lower', near_min_episodes='lower',
                      min_soc='higher', mean_soc='higher', end_soc='higher', terminal_shortfall='lower',
                      discharge_throughput_kwh='none', charge_throughput_kwh='none')
STREAK_BINS = [0, 1, 5, 17, 49, 97]                       # intervals: 0 | 1-4 | 5-16 (<=4 h) | 17-48 (<=12 h) | 49-96
STREAK_LABELS = ['0', '1-4', '5-16', '17-48', '49-96']


def battery_table(feeder, time_series=None):
    """Per-battery parameters from the environment's own battery models (index i = agent i = soc_agent_<i>)."""
    from configs.experiments import feeder_config
    from environments.marl_env import make_marl_env
    env = make_marl_env(feeder_config(feeder, time_series, 'soc_feasible'), 'val', 42, 0.0, 1.0)
    return pd.DataFrame([dict(battery=i, bus=int(bus), soc_min=float(b.config.soc_min), soc_max=float(b.config.soc_max),
                              capacity_kwh=float(b.config.capacity_kwh))
                         for i, (b, bus) in enumerate(zip(env.batteries, env.dess_buses))])


def runs_of(mask):
    """(number of separate True runs, longest run) of a 1-D boolean array."""
    longest = count = run = 0
    for v in mask:
        if v:
            run += 1
            count += run == 1
            longest = max(longest, run)
        else:
            run = 0
    return count, longest


def day_metrics(g, batteries):
    """One controller-seed-scale-topology-day group of step rows -> one row per battery."""
    g = g[g.record_type == 'dispatch_attempt'] if 'record_type' in g else g
    g = g.sort_values('step')
    n = len(batteries)
    complete = len(g) == STEPS and (g.step.to_numpy() == np.arange(STEPS)).all()
    executed = json_list(g['executed_battery_power_kw'].tolist(), n) if 'executed_battery_power_kw' in g else \
        np.column_stack([g[f'dess_power_kw_agent_{i}'].to_numpy(float) for i in range(n)])
    last = g.iloc[-1]
    completed = str(last.get('marl_terminal_completed', False)).strip().lower() in ('true', '1', '1.0')
    shortfall = json_list([last.get('marl_terminal_soc_shortfall')], n)[0] if completed else np.full(n, np.nan)
    start = json_list([last.get('marl_start_of_day_soc')], n)[0] if 'marl_start_of_day_soc' in g else np.full(n, np.nan)
    rows = []
    for b in batteries.itertuples():
        s = g[f'soc_agent_{b.battery}'].to_numpy(dtype=float)
        near = s <= b.soc_min + SOC_NEAR
        count, longest = runs_of(near)
        p = executed[:, b.battery]
        rows.append(dict(battery=b.battery, bus=b.bus, soc_min=b.soc_min, capacity_kwh=b.capacity_kwh, steps=len(g),
                         day_complete=bool(complete), near_min_fraction=float(near.mean()) if len(s) else np.nan,
                         longest_streak_intervals=longest, longest_streak_hours=longest * DT_HOURS,
                         near_min_episodes=count, min_soc=float(np.min(s)) if len(s) else np.nan,
                         mean_soc=float(np.mean(s)) if len(s) else np.nan, end_soc=float(s[-1]) if len(s) else np.nan,
                         start_soc=float(start[b.battery]), terminal_shortfall=float(shortfall[b.battery]),
                         charge_throughput_kwh=float(np.nansum(np.maximum(-p, 0)) * DT_HOURS),
                         discharge_throughput_kwh=float(np.nansum(np.maximum(p, 0)) * DT_HOURS)))
    return rows


STEP_COLUMNS = ('topology_case', 'start_index', 'step', 'record_type', 'executed_battery_power_kw',
                'marl_terminal_soc_shortfall', 'marl_terminal_completed', 'marl_start_of_day_soc')


def read_steps(path, n):
    wanted = set(STEP_COLUMNS) | {f'soc_agent_{i}' for i in range(n)} | {f'dess_power_kw_agent_{i}' for i in range(n)}
    return pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)


def missing_inputs(results_root, feeder, n, scales):
    """Expected A0/A3 x seeds x load factors with the step-table columns this analysis needs; [] when complete."""
    runs = run_dirs(results_root, feeder, CONTROLLERS, scales)[0]
    have = {(c, int(k), s) for c, k, s, _ in runs}
    problems = [f'{c} seed {k} x{s}: no finished test run' for c in CONTROLLERS for k in SEEDS for s in scales
                if (c, k, s) not in have]
    need = [f'soc_agent_{i}' for i in range(n)] + ['marl_terminal_soc_shortfall', 'marl_terminal_completed', 'step']
    for c, k, s, d in runs:
        header = set(pd.read_csv(d / 'step_metrics.csv', nrows=0).columns)
        lost = [x for x in need if x not in header]
        if 'executed_battery_power_kw' not in header and f'dess_power_kw_agent_{n - 1}' not in header:
            lost.append('executed_battery_power_kw / dess_power_kw_agent_<i>')
        if lost:
            problems.append(f'{d}: step_metrics.csv lacks {lost}')
    return problems


def collect(results_root, feeder, batteries, scales, date_of):
    rows, profiles, used = [], [], []
    for c, seed, scale, d in run_dirs(results_root, feeder, CONTROLLERS, scales)[0]:
        steps = read_steps(d / 'step_metrics.csv', len(batteries))
        used.append(str(d))
        for (tp, start), g in steps.groupby(['topology_case', 'start_index']):
            for r in day_metrics(g, batteries):
                rows.append(dict(feeder=feeder, controller=c, display=DISPLAY[c], seed=int(seed), scale=scale,
                                 topology_case=tp, start_index=int(start), **r))
        st = steps[steps.record_type == 'dispatch_attempt'] if 'record_type' in steps else steps
        for i in range(len(batteries)):
            q = st.groupby(['topology_case', 'step'])[f'soc_agent_{i}'].agg(
                mean='mean', p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9)).reset_index()
            profiles.append(q.assign(feeder=feeder, controller=c, seed=int(seed), scale=scale, battery=i))
    table = pd.DataFrame(rows)
    if not table.empty:
        table['date'] = date_of(table.start_index)
    return table, (pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()), used


def seed_summary(day):
    keys = ['feeder', 'controller', 'display', 'seed', 'scale', 'battery', 'bus']
    agg = dict(days=('start_index', 'size'), near_min_fraction=('near_min_fraction', 'mean'),
               mean_longest_streak_hours=('longest_streak_hours', 'mean'),
               max_longest_streak_hours=('longest_streak_hours', 'max'), near_min_episodes=('near_min_episodes', 'mean'),
               min_soc=('min_soc', 'min'), mean_soc=('mean_soc', 'mean'), end_soc=('end_soc', 'mean'),
               terminal_shortfall=('terminal_shortfall', 'mean'), charge_throughput_kwh=('charge_throughput_kwh', 'mean'),
               discharge_throughput_kwh=('discharge_throughput_kwh', 'mean'),
               days_with_near_min=('near_min_episodes', lambda x: int((x > 0).sum())))
    by_tp = day.groupby(keys + ['topology_case']).agg(**agg).reset_index()
    allt = day.groupby(keys).agg(**agg).reset_index().assign(topology_case='all')
    return pd.concat([allt, by_tp], ignore_index=True)


def streak_distribution(day):
    d = day.assign(streak_bin=pd.cut(day.longest_streak_intervals, STREAK_BINS, right=False, labels=STREAK_LABELS))
    return (d.groupby(['feeder', 'controller', 'seed', 'scale', 'battery', 'streak_bin'], observed=False)
            .size().rename('battery_days').reset_index())


def a0_vs_a3(day, resampler):
    """Per battery: A3 - A0 (seed mean primary, seed k vs seed k secondary), feeder scope over all topologies."""
    rows = []
    topologies = sorted(set(day.topology_case), key=lambda t: int(t[2:]))
    keys = ['scale', 'topology_case', 'start_index', 'date', 'battery']
    for metric, better in PAIRED_METRICS.items():
        for seed in [None, *SEEDS]:
            sub = day if seed is None else day[day.seed == seed]
            v = sub.groupby(['controller'] + keys)[metric].agg(['mean', 'count']).reset_index()
            if seed is None:
                v = v[v['count'] == len(SEEDS)]
            a3, a0 = v[v.controller == 'mlp_a3'], v[v.controller == 'mlp_a0']
            m = a3.merge(a0, on=keys, suffixes=('_a3', '_a0'), validate='one_to_one')
            m = m[np.isfinite(m.mean_a3) & np.isfinite(m.mean_a0)]
            for (scale, battery), g in m.groupby(['scale', 'battery']):
                counts = g.groupby('date').topology_case.nunique()
                full = counts[counts == len(topologies)].index
                per = g[g.date.isin(full)].assign(diff=lambda x: x.mean_a3 - x.mean_a0).groupby('date')[
                    ['diff', 'mean_a3', 'mean_a0']].mean().sort_index()
                r = dict(scale=scale, battery=int(battery), metric=metric, better_direction=better,
                         difference='A3 - A0', seed='mean_of_3' if seed is None else seed, mean_a3=per.mean_a3.mean(),
                         mean_a0=per.mean_a0.mean(), n_dates_dropped=int(len(counts) - len(full)),
                         **paired_stats(per['diff'].to_numpy(), per.index.tolist(), resampler,
                                        better if better != 'none' else 'lower'))
                if better == 'none':
                    r['frac_dates_a_better'] = np.nan
                rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out['family'] = out.apply(lambda r: f'a3_vs_a0|x{r.scale}|{r.metric}|seed{r.seed}', axis=1)
        out['p_holm'] = np.nan
        for _, idx in out.groupby('family').groups.items():
            out.loc[idx, 'p_holm'] = holm(out.loc[idx, 'p_signflip'].to_numpy())
    return out


def run(feeder, results_root, out_root, scales, resamples=10000, time_series=None, allow_incomplete=False):
    base = guard_output(out_root, results_root)
    batteries = battery_table(feeder, time_series)
    problems = missing_inputs(results_root, feeder, len(batteries), scales)
    for p in problems:
        print('FINAL_SOC_MISSING', p, flush=True)
    if problems and not allow_incomplete:
        raise SystemExit(f'{len(problems)} missing prerequisites (listed above); pass --allow-incomplete for a '
                         'provisional look')
    _, date_of = series_dates(feeder, time_series)
    day, profile, used = collect(results_root, feeder, batteries, scales, date_of)
    if day.empty:
        raise SystemExit(f'no finished A0/A3 test runs under {results_root}/{feeder}')
    present = day.groupby('controller').seed.unique().apply(sorted).to_dict()
    out = reserve_directory(base / 'soc_diagnostics' / feeder / stamp())
    day.to_csv(out / 'battery_day.csv', index=False)
    seed_summary(day).to_csv(out / 'seed_summary.csv', index=False)
    streak_distribution(day).to_csv(out / 'streak_distribution.csv', index=False)
    a0_vs_a3(day, Resampler(resamples, resamples)).to_csv(out / 'a0_vs_a3_battery.csv', index=False)
    profile.to_csv(out / 'soc_profile.csv', index=False)
    write_json(out / 'inclusion.json', dict(feeder=feeder, runs=used, missing_prerequisites=problems,
                                            seeds_present={k: [int(s) for s in v] for k, v in present.items()},
                                            soc_near=SOC_NEAR, near_min_definition='post-step SoC <= soc_min + SOC_NEAR',
                                            batteries=batteries.to_dict('records'), scales=list(scales),
                                            incomplete_battery_days=int((~day.day_complete).sum())))
    print('FINAL_SOC_DIAGNOSTICS_DONE', feeder, out, f'battery_days={len(day)}', f'seeds={present}', flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--results-root', default=RESULTS_ROOT)
    p.add_argument('--out-root', default=OUT_ROOT)
    p.add_argument('--scales', type=float, nargs='+', default=list(SCALES))
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--time-series', help='processed series (default: the environment configuration)')
    p.add_argument('--allow-incomplete', action='store_true', help='run despite missing runs/columns (provisional)')
    a = p.parse_args(argv)
    if a.resamples < 10000:
        p.error('use at least 10,000 bootstrap resamples')
    return run(a.feeder, a.results_root, a.out_root, a.scales, a.resamples, a.time_series, a.allow_incomplete)


if __name__ == '__main__':
    main()
