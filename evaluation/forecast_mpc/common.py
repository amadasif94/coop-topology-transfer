"""Shared definitions of the F-MPC pipeline: stages, paths, the environment's own series, forecast files. Maintained.

Data: the forecasting models train on the processed time series the environment itself reads
(configs <feeder>_config paths.processed_time_series, loaded with the environment's own loader and split rules), at
nominal load (load_scaling_factor 1.0). Forecasts are therefore in the current data version; the only transformation
applied later is the load-stress factor lambda (load only, never PV), exactly as DESSEnv._get_profiles does.

Monthly expanding-window walk-forward stages (one set of models per monthly cutoff; never retrained within a month):
    oct  fitted on 2018-01-01 .. 09-30, forecasts the October validation days
    nov  fitted on 2018-01-01 .. 10-31, forecasts the November validation days
    dec  fitted on 2018-01-01 .. 11-30, forecasts the December test days
Fitting and AutoGluon's internal model selection use only rows before the stage's month.

Layout (never overwritten):
    <root>/<feeder>/forecast_mpc/models/<stage>/<target>/          AutoGluon predictor
    <root>/<feeder>/forecast_mpc/models/<stage>/<target>.json      training record (exclusive creation)
    <root>/<feeder>/forecast_mpc/forecasts/<stage>[_first<N>]/<target>.{npz,json}
    <root>/<feeder>/forecast_mpc/forecast_accuracy/<stage>[+<stage>]_<UTC>/
targets: load_node_1 .. load_node_N, pv_node_1 .. pv_node_N (N = 33 or 69; bus i <-> environment bus index i - 1).
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.experiments import ROOT, feeder_config, sha, source_inventory, write_json
from evaluation.forecast_mpc.features import WINDOW
from utility.data_loader import (load_processed_node_time_series, split_time_series_by_date,
                                 validate_time_series_resolution)

FORECAST_SCHEMA = 'graph-marl-dess-forecast/2'           # /2: monthly walk-forward stages (oct, nov, dec)
MODEL_SCHEMA = 'graph-marl-dess-forecast-model/2'
LABEL = 'forecast_mpc'
STAGES = ('oct', 'nov', 'dec')
STAGE_MONTH = dict(oct='2018-10', nov='2018-11', dec='2018-12')       # the month a stage forecasts (its cutoff = 1st)
STAGE_SPLIT = dict(oct='val', nov='val', dec='test')                  # environment split holding that month's days
SPLIT_STAGES = dict(val=('oct', 'nov'), test=('dec',))                # stages whose forecasts an F-MPC split uses
KINDS = ('load', 'pv')
NUM_BUSES = dict(ieee33=33, ieee69=69)
STEPS = 96
# Frozen before any training; never tuned on December (or on any evaluation result).
AUTOGLUON_SETTINGS = dict(label='y', eval_metric='mean_absolute_error', presets='medium_quality', time_limit_s=300,
                          cleanup='delete_models(models_to_keep="best") then save_space()')
INFORMATION_RULE = ('forecast issued at decision row t for target rows t .. day end: realized target values of rows < t '
                    'only (row t is not yet observed), predictions for rows t .. r-1, calendar features of row r, '
                    'day-ahead prices of rows r, r-1, r-96 (same day or past)')
FMPC_SOURCES = ('evaluation/forecast_mpc/__init__.py', 'evaluation/forecast_mpc/common.py',
                'evaluation/forecast_mpc/features.py', 'evaluation/forecast_mpc/train.py',
                'evaluation/forecast_mpc/generate.py', 'evaluation/forecast_mpc/validate.py',
                'evaluation/forecast_mpc/controller.py', 'evaluation/forecast_mpc/run.py',
                'evaluation/reference_runs.py', 'evaluation/oracle/solver.py', 'evaluation/oracle/network.py',
                'evaluation/oracle/run.py')


def code_inventory(extra=FMPC_SOURCES):
    return dict(source_inventory(), **{p: sha(ROOT / p) for p in extra})


def targets(feeder):
    n = NUM_BUSES[feeder]
    return [f'load_node_{i}' for i in range(1, n + 1)] + [f'pv_node_{i}' for i in range(1, n + 1)]


def target_kind_bus(target):
    kind, _, bus = target.partition('_node_')
    if kind not in KINDS or not bus.isdigit():
        raise ValueError(f'unknown target {target!r}')
    return kind, int(bus)


def select_targets(feeder, names=None, chunk=None, chunks=None):
    """All targets, an explicit subset, or contiguous chunk `chunk` of `chunks` (Slurm array task)."""
    every = targets(feeder)
    if names:
        unknown = sorted(set(names) - set(every))
        if unknown:
            raise ValueError(f'unknown targets {unknown}')
        return [t for t in every if t in names]
    if chunk is None and chunks is None:
        return every
    if chunk is None or chunks is None or not 0 <= int(chunk) < int(chunks):
        raise ValueError('--chunk K needs --chunks N with 0 <= K < N')
    return [str(t) for t in np.array_split(np.array(every), int(chunks))[int(chunk)]]


def stage_root(root, feeder):
    return Path(root) / feeder / LABEL


def models_dir(root, feeder, stage):
    return stage_root(root, feeder) / 'models' / stage


def forecasts_dir(root, feeder, stage, max_days=None):
    return stage_root(root, feeder) / 'forecasts' / (f'{stage}_first{max_days}' if max_days else stage)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dir_sha256(path):
    """sha256 over the sorted relative paths and contents of every file below `path`, except AutoGluon's own log files
    (a 'logs' folder or *.log), which AutoGluon may append to without changing the model."""
    path, h = Path(path), hashlib.sha256()
    files = [p for p in path.rglob('*') if p.is_file() and p.suffix != '.log'
             and 'logs' not in p.relative_to(path).parts]
    for f in sorted(files):
        h.update(f.relative_to(path).as_posix().encode() + b'\0' + file_sha256(f).encode() + b'\n')
    return h.hexdigest()


@dataclass
class Series:
    """The environment's own processed series at nominal scale (row index = environment row index)."""
    feeder: str
    path: Path
    sha256: str
    timestamps: pd.DatetimeIndex        # timezone-naive (the series' UTC labels, as the earlier forecasting code)
    load: np.ndarray                    # [rows, buses] kW
    pv: np.ndarray                      # [rows, buses] kW
    price: np.ndarray                   # [rows] $/MWh (day-ahead)
    splits: dict                        # train/val/test row indices (utility.data_loader.split_time_series_by_date)
    scaling: dict

    def values(self, kind):
        return dict(load=self.load, pv=self.pv)[kind]

    def target(self, name):
        kind, bus = target_kind_bus(name)
        return self.values(kind)[:, bus - 1]


def load_series(feeder, time_series=None):
    config = feeder_config(feeder, time_series, experiment='soc_feasible')
    path = Path(config['paths']['processed_time_series']).resolve()
    ts = load_processed_node_time_series(path, NUM_BUSES[feeder], allow_empty=False)
    validate_time_series_resolution(ts, config)
    splits = {k.replace('_indices', ''): np.asarray(v, dtype=int)
              for k, v in split_time_series_by_date(ts, config).items()}
    scaling = dict(config.get('data_scaling', {}))
    factors = {k: float(scaling.get(k, 1.0)) for k in ('load_scaling_factor', 'pv_scaling_factor', 'price_scaling_factor')}
    if factors['load_scaling_factor'] != 1.0 or factors['pv_scaling_factor'] != 1.0:
        raise ValueError(f'the nominal feeder configuration must have load/PV scaling 1.0, got {factors}')
    df = ts['df']
    stamps = pd.DatetimeIndex(df[ts['datetime_col']]).tz_convert(None)
    return Series(feeder=feeder, path=path, sha256=file_sha256(path), timestamps=stamps,
                  load=df[ts['load_cols']].to_numpy(dtype=float) * factors['load_scaling_factor'],
                  pv=df[ts['renewable_cols']].to_numpy(dtype=float) * factors['pv_scaling_factor'],
                  price=df[ts['price_col']].to_numpy(dtype=float) * factors['price_scaling_factor'],
                  splits=splits, scaling=factors)


def day_starts(series, split):
    """Rows at 00:00 whose 96 rows all lie in `split` (DESSEnv._build_episode_start_indices)."""
    rows = series.splits[split]
    inside = set(rows.tolist())
    t = series.timestamps
    return np.array([r for r in rows if t[r].hour == 0 and t[r].minute == 0 and r + STEPS - 1 < len(t)
                     and all(j in inside for j in range(r, r + STEPS))], dtype=int)


def stage_day_starts(series, stage):
    """Environment day starts (of the stage's split) that fall in the stage's month."""
    starts = day_starts(series, STAGE_SPLIT[stage])
    months = series.timestamps[starts].strftime('%Y-%m')
    out = starts[np.asarray(months == STAGE_MONTH[stage])]
    if out.size == 0:
        raise ValueError(f'no {STAGE_SPLIT[stage]} days in {STAGE_MONTH[stage]}')
    return out


def cutoff_time(series, stage):
    """First timestamp NOT available to a stage's training: 00:00 on the 1st of its forecast month."""
    return pd.Timestamp(STAGE_MONTH[stage] + '-01')


def training_rows(series, stage):
    """Target rows used to fit a stage's models: every row before its month, from row 672 on (full history
    window). Expanding window from the first row of the series; never a row of the forecast month or later."""
    cutoff = cutoff_time(series, stage)
    t = series.timestamps
    before = np.flatnonzero(np.asarray(t < cutoff))
    if before.size == 0 or before.min() != 0 or np.any(np.diff(before) != 1):
        raise ValueError('the rows before the cutoff must be one contiguous block from the first row')
    if t[before.max() + 1] != cutoff:
        raise ValueError(f'the series has no row at the cutoff {cutoff}')
    test = series.splits['test']
    if test.size and before.max() >= test.min():
        raise ValueError('training rows reach the December test split')
    return before[before >= WINDOW]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_exclusive_bytes(path, data):
    with Path(path).open('xb') as f:
        f.write(data)


def save_forecast(directory, target, forecast, starts, start_times, record):
    """forecast [days, 96 issue steps, 96 target steps] (NaN where target step < issue step) + its JSON record."""
    buf = io.BytesIO()
    np.savez_compressed(buf, forecast=np.asarray(forecast, dtype=np.float32), day_start_index=np.asarray(starts),
                        day_start_time=np.asarray([str(t) for t in start_times]))
    write_exclusive_bytes(Path(directory) / f'{target}.npz', buf.getvalue())
    record = dict(record, npz_sha256=file_sha256(Path(directory) / f'{target}.npz'))
    write_json(Path(directory) / f'{target}.json', record)
    return record


class ForecastStore:
    """All targets of one stage's forecast directory (complete, same series, same days), in bus order."""

    def __init__(self, directory, feeder, series_sha256=None, stage=None):
        self.directory, self.feeder = Path(directory), feeder
        if not self.directory.is_dir():
            raise FileNotFoundError(f'forecast directory {self.directory} does not exist')
        names = targets(feeder)
        missing = [t for t in names if not (self.directory / f'{t}.json').is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} forecast files missing in {self.directory}, e.g. {missing[:3]}')
        n = NUM_BUSES[feeder]
        self.records, arrays, starts, times = {}, dict(load=[None] * n, pv=[None] * n), None, None
        for t in names:
            rec = read_json(self.directory / f'{t}.json')
            npz = self.directory / f'{t}.npz'
            if rec.get('schema') != FORECAST_SCHEMA or rec.get('feeder') != feeder or rec.get('target') != t:
                raise ValueError(f'{npz}: record does not describe {feeder}/{t}')
            if file_sha256(npz) != rec.get('npz_sha256'):
                raise ValueError(f'{npz}: file changed after it was written')
            if series_sha256 is not None and rec.get('series_sha256') != series_sha256:
                raise ValueError(f'{npz}: forecasts were made from another time series ({rec.get("series_sha256")})')
            if stage is not None and rec.get('stage') != stage:
                raise ValueError(f'{npz}: stage {rec.get("stage")!r}, expected {stage!r}')
            with np.load(npz, allow_pickle=False) as z:
                f, s, st = z['forecast'], z['day_start_index'], z['day_start_time']
            if starts is None:
                starts, times = s, st
            elif not (np.array_equal(starts, s) and np.array_equal(times, st)):
                raise ValueError(f'{npz}: covers other days than the other targets')
            kind, bus = target_kind_bus(t)
            arrays[kind][bus - 1] = f
            self.records[t] = rec
        stages = {r['stage'] for r in self.records.values()}
        shas = {r['series_sha256'] for r in self.records.values()}
        if len(stages) != 1 or len(shas) != 1:
            raise ValueError(f'mixed stages {stages} or series {shas} in {self.directory}')
        self.stage, self.series_sha256 = stages.pop(), shas.pop()
        self.load = np.stack(arrays['load'], axis=-1)        # [days, 96, 96, buses]
        self.pv = np.stack(arrays['pv'], axis=-1)
        self.day_start_index = np.asarray(starts, dtype=int)
        self.day_start_time = [str(x) for x in times]
        if self.stage in STAGE_MONTH:
            outside = [x for x in self.day_start_time if not x.startswith(STAGE_MONTH[self.stage])]
            if outside:
                raise ValueError(f'{self.directory}: stage {self.stage} forecasts days outside {STAGE_MONTH[self.stage]}: '
                                 f'{outside[:3]}')
        self._day = {int(s): d for d, s in enumerate(self.day_start_index)}

    @classmethod
    def merge(cls, stores):
        """One store over the days of several monthly stores (e.g. oct + nov for the validation split); every day comes
        from exactly one stage, and all stores were made from the same series."""
        stores = list(stores)
        shas = {s.series_sha256 for s in stores}
        feeders = {s.feeder for s in stores}
        if len(shas) != 1 or len(feeders) != 1:
            raise ValueError(f'cannot merge forecasts of different series {shas} or feeders {feeders}')
        starts = np.concatenate([s.day_start_index for s in stores])
        if len(set(starts.tolist())) != len(starts):
            raise ValueError('the same day is forecast by more than one store')
        self = cls.__new__(cls)
        self.feeder, self.series_sha256 = feeders.pop(), shas.pop()
        self.directory = [s.directory for s in stores]
        self.stage = '+'.join(s.stage for s in stores)
        self.records = {f'{s.stage}/{t}': r for s in stores for t, r in s.records.items()}
        self.load = np.concatenate([s.load for s in stores], axis=0)
        self.pv = np.concatenate([s.pv for s in stores], axis=0)
        self.day_start_index = starts.astype(int)
        self.day_start_time = [x for s in stores for x in s.day_start_time]
        self.stage_of_day = {int(d): s.stage for s in stores for d in s.day_start_index}
        self._day = {int(s): d for d, s in enumerate(self.day_start_index)}
        return self

    def target_record(self, target):
        """The record of `target` (of the first stage when merged)."""
        return self.records.get(target) or next((r for k, r in self.records.items() if k.endswith('/' + target)), {})

    @classmethod
    def from_arrays(cls, feeder, load, pv, day_start_index, day_start_time, stage='test-arrays', series_sha256=None):
        """In-memory store (tests); arrays [days, 96, 96, buses]."""
        self = cls.__new__(cls)
        self.directory, self.feeder, self.records = None, feeder, {}
        self.stage, self.series_sha256 = stage, series_sha256
        self.load, self.pv = np.asarray(load, dtype=float), np.asarray(pv, dtype=float)
        self.day_start_index = np.asarray(day_start_index, dtype=int)
        self.day_start_time = [str(x) for x in day_start_time]
        self._day = {int(s): d for d, s in enumerate(self.day_start_index)}
        return self

    def has_day(self, start_index):
        return int(start_index) in self._day

    def window(self, start_index, step):
        """Nominal load and PV forecasts issued at decision step `step` of the day starting at `start_index`, for
        target steps step .. 95: two arrays [96 - step, buses]. Raises if the day or any value is missing."""
        if int(start_index) not in self._day:
            raise KeyError(f'no forecast for the day starting at row {start_index}')
        if not 0 <= int(step) < STEPS:
            raise ValueError(f'step {step} outside 0..95')
        d, k = self._day[int(start_index)], int(step)
        load, pv = self.load[d, k, k:], self.pv[d, k, k:]
        if not (np.isfinite(load).all() and np.isfinite(pv).all()):
            raise ValueError(f'missing forecast values for day {start_index} step {step}')
        return load.astype(float), pv.astype(float)

    def summary(self):
        directory = ([str(d) for d in self.directory] if isinstance(self.directory, list)
                     else str(self.directory) if self.directory else None)
        return dict(directory=directory, stage=self.stage, feeder=self.feeder,
                    series_sha256=self.series_sha256, days=len(self.day_start_index),
                    targets=len(self.records) or None,
                    model_dir_sha256={t: r.get('model_dir_sha256') for t, r in self.records.items()} or None)


def stage_status(directory, feeder, names=None, kind='model'):
    """Completeness of a stage directory: complete targets (record present), partial (artifact without record), missing."""
    directory = Path(directory)
    done, partial, missing = [], [], []
    for t in names or targets(feeder):
        artifact = directory / (t if kind == 'model' else f'{t}.npz')
        if (directory / f'{t}.json').is_file():
            done.append(t)
        elif artifact.exists():
            partial.append(t)
        else:
            missing.append(t)
    return dict(directory=str(directory), complete=len(done), partial=partial, missing=missing,
                total=len(names or targets(feeder)))


def print_status(status, marker):
    ok = not status['partial'] and not status['missing']
    print(f"{marker} {'COMPLETE' if ok else 'INCOMPLETE'} {status['complete']}/{status['total']} in {status['directory']}"
          + (f"; partial (artifact without record; move it away before re-running): {status['partial']}"
             if status['partial'] else '')
          + (f"; missing: {status['missing'][:10]}{' ...' if len(status['missing']) > 10 else ''}"
             if status['missing'] else ''), flush=True)
    return ok
