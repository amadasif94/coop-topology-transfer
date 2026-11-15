"""Synthetic inputs for the F-MPC / PQ-RBC tests: a smooth 2018 series in the environment's file format and a stub in
place of AutoGluon. No research data is read and no model is trained (the stub 'fit' only records its input)."""
import json
from pathlib import Path
import sys
import types
from unittest import mock

import numpy as np
import pandas as pd

ROWS = 365 * 96


def write_synthetic_series(feeder, path, seed=0):
    """Full 2018 year, 15-minute rows, the processed-series columns of the environment (UTC labels as the real file)."""
    from configs.experiments import feeder_config
    cfg = feeder_config(feeder)
    base = pd.read_csv(cfg['paths']['bus_data']).sort_values('bus')['load_kw'].to_numpy(dtype=float)
    n = len(base)
    rng = np.random.default_rng(seed)
    t = pd.date_range('2018-01-01', periods=ROWS, freq='15min', tz='UTC')
    slot = (t.hour * 4 + t.minute // 15).to_numpy()
    day = np.arange(ROWS) // 96
    shape = 0.55 + 0.30 * np.exp(-((slot - 76) / 14.0) ** 2) + 0.12 * np.exp(-((slot - 32) / 10.0) ** 2)
    weekly = 1.0 - 0.06 * np.isin(t.dayofweek.to_numpy(), (5, 6))
    load = base[None, :] * (shape * weekly)[:, None] * (1.0 + 0.03 * rng.standard_normal((ROWS, n)))
    load = np.clip(load, 0.0, None)
    sun = np.clip(np.sin(np.pi * (slot - 24) / 48.0), 0.0, None) * ((slot >= 24) & (slot <= 72))
    cloud = 0.4 + 0.6 * rng.random(365)[day]
    pv = 40.0 * sun[:, None] * cloud[:, None] * (0.8 + 0.4 * rng.random(n))[None, :]
    hourly = 30.0 + 14.0 * np.sin(2 * np.pi * (np.arange(365 * 24) % 24 - 7) / 24.0) + 4.0 * rng.standard_normal(365 * 24)
    price = np.repeat(np.clip(hourly, 5.0, None).round(2), 4)
    df = pd.DataFrame({'date_time': t.strftime('%Y-%m-%d %H:%M:%S+00:00')})
    for i in range(n):
        df[f'active_power_node_{i + 1}'] = load[:, i]
    for i in range(n):
        df[f'renewable_active_power_node_{i + 1}'] = pv[:, i]
    df['price'] = price
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return Path(path)


class _Features:
    def __init__(self, names):
        self.names = list(names)

    def get_features(self):
        return list(self.names)


class _Metric:
    name = 'mean_absolute_error'


class _Learner:
    def __init__(self, rows):
        self._pre_X_rows = rows


class StubPredictor:
    """Stands in for autogluon.tabular.TabularPredictor. RULE decides predict(): 'lag96' -> y_lag_96, 'lag1' -> y_lag_1.

    Like AutoGluon, fit() removes useless input columns (a single value in the training data) unless DROP_USELESS is
    False; EXTRA_DROP removes further (informative) columns to test the refusal. original_features is the input schema,
    feature_metadata_in the retained one, and predict() accepts the full input frame but reads retained columns only."""
    RULE = 'lag96'
    DROP_USELESS = True
    EXTRA_DROP = ()
    fits = []
    problem_type = 'regression'
    eval_metric = _Metric()

    def __init__(self, label='y', eval_metric=None, path=None, verbosity=2):
        self.label, self.path, self.rule = label, Path(path) if path else None, StubPredictor.RULE
        self.original, self.features, self._learner = None, None, None

    def fit(self, train_data, time_limit=None, presets=None):
        self.original = [c for c in train_data.columns if c != self.label]
        useless = [c for c in self.original if StubPredictor.DROP_USELESS and train_data[c].nunique() == 1]
        self.features = [c for c in self.original if c not in useless and c not in StubPredictor.EXTRA_DROP]
        self._learner = _Learner(len(train_data))
        self.path.mkdir(parents=True)
        info = dict(rule=self.rule, original=self.original, features=self.features, rows=len(train_data),
                    y_max=float(train_data[self.label].max()), time_limit=time_limit, presets=presets, useless=useless)
        (self.path / 'predictor.pkl').write_text(json.dumps(info))
        for name, text in (('learner.pkl', 'stub learner'), ('metadata.json', '{}'), ('version.txt', 'stub')):
            (self.path / name).write_text(text)                  # the files of a finished AutoGluon save
        (self.path / 'models' / 'Stub').mkdir(parents=True)
        (self.path / 'models' / 'Stub' / 'model.pkl').write_text('stub model')
        StubPredictor.fits.append(info)
        return self

    @property
    def original_features(self):
        return list(self.original)

    def leaderboard(self, silent=True):
        return pd.DataFrame(dict(model=['Stub'], score_val=[-1.0]))

    @property
    def model_best(self):
        return 'Stub'

    def delete_models(self, models_to_keep='best', dry_run=False):
        return None

    def save_space(self):
        return None

    @property
    def feature_metadata_in(self):
        return _Features(self.features)

    @classmethod
    def load(cls, path):
        info = json.loads((Path(path) / 'predictor.pkl').read_text())       # raises on a corrupted file
        self = cls(path=path)
        self.rule, self.original, self.features = info['rule'], info['original'], info['features']
        self._learner = _Learner(info['rows'])
        return self

    def predict(self, X):
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f'missing retained features {missing}')
        X = X[self.features]                                                 # AutoGluon selects its retained columns
        column = dict(lag96='y_lag_96', lag1='y_lag_1')[self.rule]
        return pd.Series(X[column].to_numpy(dtype=float))


def fake_autogluon():
    """Context manager: `from autogluon.tabular import TabularPredictor` returns StubPredictor."""
    package, tabular = types.ModuleType('autogluon'), types.ModuleType('autogluon.tabular')
    tabular.TabularPredictor, tabular.__version__ = StubPredictor, 'stub'
    package.tabular = tabular
    return mock.patch.dict(sys.modules, {'autogluon': package, 'autogluon.tabular': tabular})


def actual_forecast_arrays(series, starts, kind):
    """'Perfect' forecasts [days, 96, 96, buses]: the realized nominal values (tests of the optimisation path only)."""
    values = series.values(kind)
    out = np.full((len(starts), 96, 96, values.shape[1]), np.nan)
    for d, s in enumerate(starts):
        for k in range(96):
            out[d, k, k:] = values[s + k:s + 96]
    return out
