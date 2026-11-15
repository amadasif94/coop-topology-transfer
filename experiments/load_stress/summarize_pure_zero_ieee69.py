"""Summarize the IEEE69 pure zero-action reference (0 kW, no backup) per load scale and topology (TP1-TP10).

    python -u -m experiments.load_stress.summarize_pure_zero_ieee69 \
        --root results/load_stress/ieee69/pure_zero_no_backup --output <new dir> \
        [--controller-root results/load_stress/ieee69]

Read only except for the new --output directory. Smoke runs are never used. Day-level values with day-bootstrap
95% CIs (one reference run: zero does not depend on training). Failed days are counted, never dropped.
Outputs:
  episode_metrics_long.csv            one row per scale x topology x day
  absolute_by_scale.csv               mean and CI per scale x topology x metric
  controllers_minus_pure_by_day.csv   A0/A1/A2 load-stress run minus pure zero, same scale and day; controller runs
  controllers_minus_pure_by_run.csv   are discovered by summarize_load_stress_ieee69.discover (FAIL/stale identity
  controllers_minus_pure_by_model.csv reports excluded); by_model uses the two-level run/day bootstrap
  inclusion.json                      included / missing scales, failed days, smoke and cross-check status
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.load_stress import summarize_load_stress as S33
from experiments.load_stress.evaluate_pure_zero_ieee69 import DAYS, SMOKE_SCALE, TOPOLOGIES, run_dir
from experiments.load_stress.pure_zero_env import LABEL
from experiments.load_stress.stress_env import ALLOWED_SCALES
from experiments.load_stress.summarize_load_stress_ieee69 import discover
from experiments.load_stress.summarize_pure_zero_ieee33 import episode_table
from experiments.no_shared_kpi.protocol import read_json, sha, write_json
from utility.run_manifest import reserve_directory

KEYS = S33.KEYS


def status_of(path):
    return read_json(path).get('status') if path.is_file() else 'NOT_PERFORMED'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--controller-root', type=Path, default=Path('results/load_stress/ieee69'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261028)
    a = p.parse_args(argv)
    included, missing, tables = [], [], []
    for s in ALLOWED_SCALES:
        d = run_dir(a.root, s)
        m = d / 'run_manifest.json'
        if not m.is_file() or read_json(m).get('status') != 'finished':
            missing.append(s)
            continue
        included.append(dict(scale=s, dir=str(d), manifest_sha256=sha(m), episodes=read_json(m)['pure_zero'].get('episodes'),
                             cross_check=status_of(d / 'cross_check_report.json')))
        tables.append(episode_table(d, s, LABEL))
    if not tables:
        raise SystemExit('No finished IEEE69 pure zero runs under ' + str(a.root))
    out = reserve_directory(a.output)
    rng = np.random.default_rng(a.seed)
    table = pd.concat(tables, ignore_index=True)
    table = table[table.topology_case.isin(TOPOLOGIES)]
    table['failed_day'] = table['failed_day'].astype(float)
    metrics = [c for c in table.columns if c not in ('scale', 'source', *KEYS)]
    table.to_csv(out / 'episode_metrics_long.csv', index=False)
    S33.summarize_deltas(table, metrics, ['scale', 'topology_case'], rng, a.bootstrap) \
        .rename(columns={'mean_delta': 'mean'}).to_csv(out / 'absolute_by_scale.csv', index=False)

    runs, excluded = discover(a.controller_root)
    scales = set(table.scale)
    deltas = []
    for item in [i for i in runs if i['scale'] in scales]:
        c = episode_table(item['dir'], item['scale'], item['model'])
        c['failed_day'] = c['failed_day'].astype(float)
        both = c.merge(table, on=['scale', *KEYS], suffixes=('', '__pure'), validate='one_to_one')
        d = both[['scale', *KEYS]].copy()
        for m in metrics:
            if m in c:
                d[m] = both[m] - both[m + '__pure']
        d['model'], d['run'], d['identity_check'] = item['model'], item['run'], item['identity_check']
        deltas.append(d)
    delta = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()
    delta.to_csv(out / 'controllers_minus_pure_by_day.csv', index=False)
    if len(delta):
        cols = [m for m in metrics if m in delta]
        S33.summarize_deltas(delta, cols, ['model', 'run', 'scale', 'topology_case'], rng, a.bootstrap) \
            .to_csv(out / 'controllers_minus_pure_by_run.csv', index=False)
        S33.model_level(delta, cols, rng, a.bootstrap, extra=('scale',)).to_csv(out / 'controllers_minus_pure_by_model.csv', index=False)
    expected = len(ALLOWED_SCALES) * len(TOPOLOGIES) * DAYS
    write_json(out / 'inclusion.json', dict(root=str(Path(a.root).resolve()), bootstrap=a.bootstrap, seed=a.seed,
        included=included, missing_scales=missing, episodes=int(len(table)), expected_episodes=expected,
        failed_days=int(table['failed_day'].sum()),
        smoke_check=status_of(run_dir(a.root, SMOKE_SCALE, smoke=True) / 'smoke_check.json'),
        controller_runs_compared=sorted({(i['model'], i['run'], i['scale']) for i in runs if i['scale'] in scales}),
        controller_runs_excluded=excluded,
        note='Pure zero-action reference (0 kW, no backup): not a controller; executed dispatch may violate limits.'))
    print('SUMMARY', out, json.dumps(dict(scales=len(included), episodes=int(len(table)), expected=expected,
          failed_days=int(table['failed_day'].sum()), controller_runs=len(deltas))), flush=True)


if __name__ == '__main__':
    main()
