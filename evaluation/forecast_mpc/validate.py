"""Forecast accuracy by horizon, persistence baselines and physical checks. Maintained entry point.

    python -u -m evaluation.forecast_mpc.validate --feeder ieee33 --stage oct nov     (validation months)
    python -u -m evaluation.forecast_mpc.validate --feeder ieee33 --stage dec         (December, reporting only)

Reads a complete forecast directory (evaluation.forecast_mpc.generate) and the environment's series (realized
nominal values; read here only to SCORE forecasts). For every issue step k and target step j >= k of every day the error
is forecast - actual; 'steps_ahead' = j - k + 1 (1 = the interval being dispatched, which is not observed at issuance).
Baselines use the same information: last_value = realized value of row t - 1 for every target row; same_time_yesterday
= realized value of row r - 96 (always < t). Feeder totals sum the buses. PV is also scored over daylight target rows
(realized feeder PV > 0). Physical checks: missing/nonfinite values, negative values, PV above its training maximum,
PV forecast at night (realized feeder PV = 0) and the clipping counts recorded at generation.

Stages 'oct' and 'nov' (each month scored with its own walk-forward models; reported per month and pooled) are the
accuracy gate before any controller run. Stage 'dec' may be scored afterwards for reporting only; the settings are
frozen and never changed from it.

Outputs (never overwritten): <root>/<feeder>/forecast_mpc/forecast_accuracy/<stage>[+<stage>]_<UTC>/
horizon_metrics.csv (column 'scope' = month or 'pooled'; per bus and totals x steps ahead), physical_checks.csv,
summary.json.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, MAINTAINED_PROTOCOL, maintained_output_root, write_json
from evaluation.forecast_mpc.common import (STAGES, STEPS, ForecastStore, forecasts_dir, load_series, stage_root,
                                           target_kind_bus, targets)
from utility.run_manifest import reserve_directory

DAYLIGHT_KW = 1e-6


def horizon_rows(forecast, actual, starts, name, kind, bus, mask_series=None):
    """forecast [days, 96, 96]; actual: realized series; mask_series: optional boolean series of scored target rows."""
    rows = []
    starts = np.asarray(starts, dtype=int)
    for h in range(STEPS):
        k = np.arange(STEPS - h)
        j = k + h
        target_rows = starts[:, None] + j[None, :]
        pred = forecast[:, k, j].astype(float)
        act = actual[target_rows]
        last = actual[starts[:, None] + k[None, :] - 1]
        yesterday = actual[target_rows - STEPS]
        mask = np.ones(pred.shape, dtype=bool) if mask_series is None else mask_series[target_rows]
        n = int(mask.sum())
        r = dict(name=name, kind=kind, bus=bus, steps_ahead=h + 1, n=n)
        for label, ref in (('', pred), ('_last_value', last), ('_same_time_yesterday', yesterday)):
            err = (ref - act)[mask]
            r[f'mae{label}'] = float(np.abs(err).mean()) if n else np.nan
            r[f'rmse{label}'] = float(np.sqrt((err ** 2).mean())) if n else np.nan
        r['bias'] = float((pred - act)[mask].mean()) if n else np.nan
        rows.append(r)
    return rows


def pooled(table):
    """Error pooled over every scored (day, issue step, target step) of a name: n-weighted means."""
    w = table['n'].to_numpy(dtype=float)
    out = dict(n=int(w.sum()))
    for m in ('mae', 'mae_last_value', 'mae_same_time_yesterday'):
        out[m] = float(np.nansum(table[m] * w) / w.sum()) if w.sum() else np.nan
    for m in ('rmse', 'rmse_last_value', 'rmse_same_time_yesterday'):
        out[m] = float(np.sqrt(np.nansum(table[m] ** 2 * w) / w.sum())) if w.sum() else np.nan
    for base in ('last_value', 'same_time_yesterday'):
        out[f'mae_skill_vs_{base}'] = 1.0 - out['mae'] / out[f'mae_{base}'] if out[f'mae_{base}'] else np.nan
    return out


def physical_checks(store, series):
    rows = []
    valid = np.triu(np.ones((STEPS, STEPS), dtype=bool))[None, :, :]
    pv_total = series.pv.sum(axis=1)
    night = pv_total <= DAYLIGHT_KW
    target_rows = store.day_start_index[:, None, None] + np.arange(STEPS)[None, None, :]
    night_mask = np.broadcast_to(night[target_rows], (len(store.day_start_index), STEPS, STEPS)) & valid
    for t in targets(store.feeder):
        kind, bus = target_kind_bus(t)
        f = (store.load if kind == 'load' else store.pv)[..., bus - 1]
        v = np.broadcast_to(valid, f.shape)
        rec = store.records.get(t, {})
        upper = rec.get('clip_upper')
        r = dict(target=t, kind=kind, bus=bus, values=int(v.sum()), nonfinite=int((~np.isfinite(f[v])).sum()),
                 negative=int((f[v] < 0).sum()),
                 above_training_max=int((f[v] > upper + 1e-9).sum()) if upper is not None else 0,
                 clipped_below=(rec.get('clipping') or {}).get('below_lower'),
                 clipped_above=(rec.get('clipping') or {}).get('above_upper'),
                 recursive_predictions=(rec.get('clipping') or {}).get('predictions'))
        if kind == 'pv':
            nf = f[night_mask]
            r.update(night_values=int(nf.size), night_mean_kw=float(nf.mean()) if nf.size else np.nan,
                     night_max_kw=float(nf.max()) if nf.size else np.nan,
                     night_nonzero_fraction=float((nf > 1e-6).mean()) if nf.size else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def validate(store, series):
    metrics = []
    for t in targets(store.feeder):
        kind, bus = target_kind_bus(t)
        f = (store.load if kind == 'load' else store.pv)[..., bus - 1]
        metrics += horizon_rows(f, series.target(t), store.day_start_index, t, kind, bus)
    daylight = series.pv.sum(axis=1) > DAYLIGHT_KW
    for kind in ('load', 'pv'):
        total = (store.load if kind == 'load' else store.pv).sum(axis=-1)
        metrics += horizon_rows(total, series.values(kind).sum(axis=1), store.day_start_index, f'{kind}_total', kind,
                                'total')
    metrics += horizon_rows(store.pv.sum(axis=-1), series.pv.sum(axis=1), store.day_start_index, 'pv_total_daylight',
                            'pv', 'total', mask_series=daylight)
    table = pd.DataFrame(metrics)
    summary = {name: pooled(table[table.name == name]) for name in ('load_total', 'pv_total', 'pv_total_daylight')}
    per_bus = {}
    for kind in ('load', 'pv'):
        sub = table[(table.kind == kind) & (table.bus != 'total')]
        per_bus[kind] = pooled(sub)
    return table, summary, per_bus


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--stage', required=True, nargs='+', choices=STAGES, help='one or more monthly stages')
    p.add_argument('--forecast-dirs', nargs='+', help='one per --stage (default <root>/<feeder>/forecast_mpc/forecasts/<stage>)')
    p.add_argument('--output-root', help='default results/<protocol>')
    p.add_argument('--time-series', help='processed series (default: the environment configuration)')
    a = p.parse_args(argv)
    stages = list(dict.fromkeys(a.stage))
    if 'dec' in stages and len(stages) > 1:
        p.error('score December (dec) on its own; it is never pooled with the validation months')
    if a.forecast_dirs and len(a.forecast_dirs) != len(stages):
        p.error('--forecast-dirs needs one directory per --stage')
    root = maintained_output_root(MAINTAINED_PROTOCOL, a.output_root)
    series = load_series(a.feeder, a.time_series)
    dirs = a.forecast_dirs or [forecasts_dir(root, a.feeder, s) for s in stages]
    stores = [ForecastStore(d, a.feeder, series_sha256=series.sha256, stage=s) for s, d in zip(stages, dirs)]
    tables, checks, summaries = [], [], {}
    scopes = [(s.stage, s) for s in stores] + ([('pooled', ForecastStore.merge(stores))] if len(stores) > 1 else [])
    for scope, store in scopes:
        table, summary, per_bus = validate(store, series)
        tables.append(table.assign(scope=scope))
        summaries[scope] = dict(days=len(store.day_start_index), first_day=store.day_start_time[0],
                                last_day=store.day_start_time[-1], feeder_totals=summary, per_bus_pooled=per_bus)
    for store in stores:
        checks.append(physical_checks(store, series).assign(stage=store.stage))
    checks = pd.concat(checks, ignore_index=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = reserve_directory(stage_root(root, a.feeder) / 'forecast_accuracy' / f'{"+".join(stages)}_{stamp}')
    pd.concat(tables, ignore_index=True).to_csv(out / 'horizon_metrics.csv', index=False)
    checks.to_csv(out / 'physical_checks.csv', index=False)
    problems = dict(nonfinite=int(checks.nonfinite.sum()), negative=int(checks.negative.sum()),
                    above_training_max=int(checks.above_training_max.sum()))
    write_json(out / 'summary.json', dict(
        feeder=a.feeder, stages=stages, forecast_dirs=[str(d) for d in dirs], series_sha256=series.sha256,
        purpose=('descriptive test-period accuracy; settings are frozen and never changed from it' if 'dec' in stages
                 else 'accuracy gate before controller runs (validation months, each with its own walk-forward models)'),
        scopes=summaries, physical_problems=problems,
        night_pv_max_kw=float(np.nanmax(checks.get('night_max_kw', pd.Series([np.nan])))),
        steps_ahead_note='1 = the interval being dispatched (not observed at issuance); 96 = 24 h at decision step 0'))
    for scope, info in summaries.items():
        for name, s in info['feeder_totals'].items():
            print(f'FMPC_ACCURACY {a.feeder} {scope} {name}: MAE {s["mae"]:.3f} kW (last value {s["mae_last_value"]:.3f}, '
                  f'same time yesterday {s["mae_same_time_yesterday"]:.3f}) RMSE {s["rmse"]:.3f} kW, n={s["n"]}',
                  flush=True)
    print('FMPC_PHYSICAL_CHECKS', problems, flush=True)
    print('FMPC_VALIDATE_DONE', out, flush=True)
    return out


if __name__ == '__main__':
    main()
