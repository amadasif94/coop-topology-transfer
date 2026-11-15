"""Causal recursive multi-step load/PV forecasts for every decision step of a period. Maintained entry point.

    python -u -m evaluation.forecast_mpc.generate --feeder ieee33 --stage oct [--chunk K --chunks N] [--max-days N]
    python -u -m evaluation.forecast_mpc.generate --feeder ieee33 --stage nov [--chunk K --chunks N]
    python -u -m evaluation.forecast_mpc.generate --feeder ieee33 --stage dec [--chunk K --chunks N]
    python -u -m evaluation.forecast_mpc.generate --feeder ieee33 --stage oct --status

Each monthly stage uses its own models (fitted on every row before the month) on that month's days only: 'oct' on
the October validation days, 'nov' on the November validation days, 'dec' on the December test days. The models are not
refitted within the month. For every day d (start row s) and decision step k = 0 .. 95 a forecast is ISSUED at row
t = s + k for the target rows t .. s + 95 (the rest of the day; the MPC horizon). Issuance information: realized
target values of rows < t only (the interval t itself has not happened yet, so it is forecast too); the model is then
applied recursively, each prediction (clipped to [0, PV training maximum] / [0, inf) for load) becoming history for
the next row. Calendar features and day-ahead prices of the same day are known in advance. Forecasts do not depend on
topology, load factor or any controller action, so one set per feeder and stage serves every F-MPC run; the load
factor is applied later by the controller.

Outputs (never overwritten): <root>/<feeder>/forecast_mpc/forecasts/<stage>[_first<N>]/<target>.npz
(forecast [days, 96 issue steps, 96 target steps], NaN below the diagonal; day_start_index; day_start_time) and
<target>.json (model record and predictor sha256, series sha256, information rule, clipping counts).
"""
import argparse
import time

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, MAINTAINED_PROTOCOL, maintained_output_root
from evaluation.forecast_mpc.common import (FORECAST_SCHEMA, INFORMATION_RULE, STAGE_MONTH, STAGE_SPLIT, STAGES, STEPS,
                                           stage_day_starts, dir_sha256, forecasts_dir, load_series, models_dir, print_status,
                                           read_json, save_forecast, select_targets, stage_status, target_kind_bus)
from evaluation.forecast_mpc.features import FEATURES, WINDOW, feature_frame


def recursive_forecast(y, price, timestamps, starts, predict, lower=0.0, upper=None, issue_steps=None):
    """Causal recursive forecasts of one target.

    y: realized target series (only rows < each issuance row are ever read); price: day-ahead price series;
    timestamps: row labels; starts: day start rows; predict: f(feature DataFrame) -> 1-D array.
    Returns (forecast [days, 96, 96] float32 with NaN where target step < issue step, clip counts).
    issue_steps (optional, default every step 0..95): only these decision steps are issued, e.g. [k] for the single
    forecast an online controller needs at step k (evaluation.final_analyses.timing); each issued forecast is computed
    exactly as in the full run (same history rule, recursion and clipping), the others stay NaN."""
    y, starts = np.asarray(y, dtype=float), np.asarray(starts, dtype=int)
    days = len(starts)
    steps = np.arange(STEPS) if issue_steps is None else np.asarray(sorted(set(int(k) for k in issue_steps)), dtype=int)
    if steps.size == 0 or steps.min() < 0 or steps.max() >= STEPS:
        raise ValueError(f'issue_steps must lie in 0..{STEPS - 1}')
    issue_day = np.repeat(np.arange(days), len(steps))
    issue_step = np.tile(steps, days)
    issue_row = starts[issue_day] + issue_step
    if issue_row.min() < WINDOW:
        raise ValueError('every issuance needs 672 observed rows of history')
    # history buffer: columns 0..671 = realized rows t-672 .. t-1; column 672 + h = prediction for row t + h
    buffer = np.full((len(issue_row), WINDOW + STEPS), np.nan)
    buffer[:, :WINDOW] = y[issue_row[:, None] - WINDOW + np.arange(WINDOW)[None, :]]
    out = np.full((days, STEPS, STEPS), np.nan, dtype=np.float32)
    clipped = dict(below_lower=0, above_upper=0, predictions=0)
    for h in range(STEPS - int(steps.min())):
        active = np.flatnonzero(issue_step + h <= STEPS - 1)          # target row stays within the issuing day
        p = WINDOW + h
        rows = issue_row[active] + h
        frame = feature_frame(buffer[active, p - WINDOW:p], timestamps[rows], price, rows)
        raw = np.asarray(predict(frame[list(FEATURES)]), dtype=float).reshape(-1)
        if raw.shape != (len(active),) or not np.isfinite(raw).all():
            raise ValueError(f'predictor returned {raw.shape} values (expected {len(active)}) or nonfinite values')
        clipped['below_lower'] += int((raw < lower).sum())
        clipped['above_upper'] += int((raw > upper).sum()) if upper is not None else 0
        clipped['predictions'] += int(raw.size)
        value = np.clip(raw, lower, upper if upper is not None else np.inf)
        buffer[active, p] = value
        out[issue_day[active], issue_step[active], issue_step[active] + h] = value
    return out, clipped


def load_predictor(model_path, record):
    """Read-only load of a trained predictor after checking it is the recorded one.

    The predictor always receives the full input schema (FEATURES); AutoGluon itself selects the columns it retained
    (a PV model drops the always-zero y_roll_min_96 / y_roll_min_672). The retained columns must be the recorded ones
    (records written before retained_features existed required all of FEATURES to be retained)."""
    if dir_sha256(model_path) != record['model_dir_sha256']:
        raise ValueError(f'{model_path} differs from the predictor recorded at training time')
    if list(record.get('input_features', FEATURES)) != list(FEATURES):
        raise ValueError(f'{model_path}: recorded input schema differs from the feature builder')
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(str(model_path))
    retained = sorted(predictor.feature_metadata_in.get_features())
    if retained != sorted(record.get('retained_features', FEATURES)):
        raise ValueError(f'{model_path}: retained features {retained} differ from the training record')
    features = list(FEATURES)

    def predict(frame):
        return np.asarray(predictor.predict(frame[features]), dtype=float)
    return predict


def generate_target(series, target, stage, root, out, max_days=None, skip_completed=False):
    if (out / f'{target}.json').exists():
        if skip_completed:
            print('FMPC_FORECAST_TARGET_SKIPPED (record exists)', target, flush=True)
            return None
        raise FileExistsError(f'{out / target}.json exists; forecasts are never overwritten')
    if (out / f'{target}.npz').exists():
        raise FileExistsError(f'{out / target}.npz exists without a record; move it away first')
    mdir = models_dir(root, series.feeder, stage)
    record = read_json(mdir / f'{target}.json')
    if (record.get('stage'), record.get('target'), record.get('feeder')) != (stage, target, series.feeder):
        raise ValueError(f'{mdir / target}.json is not the {stage} model of {series.feeder}/{target}')
    if record.get('series_sha256') != series.sha256:
        raise ValueError(f'{target}: model trained on another time series ({record.get("series_sha256")})')
    starts = stage_day_starts(series, stage)[:max_days or None]
    first_issue = series.timestamps[starts.min()]
    if first_issue < pd.Timestamp(record['cutoff_time']):
        raise ValueError(f'{target}: forecast period starts {first_issue}, before the model cutoff {record["cutoff_time"]}')
    if record['cutoff_time'] != str(pd.Timestamp(STAGE_MONTH[stage] + '-01')):
        raise ValueError(f'{target}: model cutoff {record["cutoff_time"]} is not the start of {STAGE_MONTH[stage]}')
    kind, _ = target_kind_bus(target)
    predict = load_predictor(mdir / target, record)
    started = time.perf_counter()
    forecast, clipped = recursive_forecast(series.target(target), series.price, series.timestamps, starts, predict,
                                           lower=record['clip_lower'], upper=record['clip_upper'])
    rec = dict(schema=FORECAST_SCHEMA, feeder=series.feeder, stage=stage, month=STAGE_MONTH[stage],
               period_split=STAGE_SPLIT[stage], target=target, kind=kind,
               days=int(len(starts)), first_day=str(series.timestamps[starts.min()]),
               last_day=str(series.timestamps[starts.max()]), max_days=max_days, issues_per_day=STEPS,
               horizon='rest of the issuing day (96 - k target rows at decision step k)',
               information_rule=INFORMATION_RULE, series_path=str(series.path), series_sha256=series.sha256,
               model_record=str(mdir / f'{target}.json'), model_dir_sha256=record['model_dir_sha256'],
               model_cutoff_time=record['cutoff_time'], autogluon_version=record.get('autogluon_version'),
               clip_lower=record['clip_lower'], clip_upper=record['clip_upper'], clipping=clipped,
               generation_seconds=time.perf_counter() - started)
    save_forecast(out, target, forecast, starts, series.timestamps[starts], rec)
    print('FMPC_FORECAST_TARGET_DONE', series.feeder, stage, target, f'days={len(starts)}',
          f'clipped={clipped["below_lower"] + clipped["above_upper"]}/{clipped["predictions"]}',
          f'{rec["generation_seconds"]:.0f}s', flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--targets', nargs='+')
    p.add_argument('--chunk', type=int)
    p.add_argument('--chunks', type=int)
    p.add_argument('--max-days', type=int, help='smoke: first N days only (separate <stage>_first<N> folder)')
    p.add_argument('--skip-completed', action='store_true')
    p.add_argument('--status', action='store_true', help='report completeness and exit (writes nothing)')
    p.add_argument('--output-root', help='default results/<protocol>')
    p.add_argument('--time-series', help='processed series (default: the environment configuration)')
    a = p.parse_args(argv)
    if a.max_days is not None and a.max_days < 1:
        p.error('--max-days must be positive')
    root = maintained_output_root(MAINTAINED_PROTOCOL, a.output_root)
    names = select_targets(a.feeder, a.targets, a.chunk, a.chunks)
    out = forecasts_dir(root, a.feeder, a.stage, a.max_days)
    if a.status:
        ok = print_status(stage_status(out, a.feeder, names, kind='forecast'), 'FMPC_FORECAST_STATUS')
        raise SystemExit(0 if ok else 1)
    series = load_series(a.feeder, a.time_series)
    out.mkdir(parents=True, exist_ok=True)
    print('FMPC_FORECAST_START', a.feeder, a.stage, f'targets={len(names)}', f'month={STAGE_MONTH[a.stage]}',
          out, flush=True)
    for t in names:
        generate_target(series, t, a.stage, root, out, a.max_days, a.skip_completed)
    print('FMPC_FORECAST_DONE', a.feeder, a.stage, f'targets={len(names)}', out, flush=True)
    return out


if __name__ == '__main__':
    main()
