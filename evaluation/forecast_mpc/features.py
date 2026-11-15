"""Causal feature builder of the F-MPC forecasting models. Maintained.

Adapted from experiments/mpc_smpc/forecasting/build_33bus_forecasts.py (build_features_for_target), which is NOT
imported (it creates results/mpc_smpc folders at import time and needs AutoGluon). Same lag set, rolling windows, time-
of-day/day-of-week cycles and price features; documented changes:
  * 'month' and 'dayofyear' are removed: the training rows never contain October-December values, so tree models would
    have to extrapolate them. The cyclic day-of-year terms doy_sin / doy_cos stay.
  * the target-specific column prefix (e.g. load_node_1_lag_1) is replaced by the generic 'y_' (one model per target).

For a target row r every history feature uses ONLY the target series at rows r-672 .. r-1 (the 'window'), so a
feature row never contains the value it predicts or anything later. Training and recursive forecasting call the SAME
function (history_features) on a [n, 672] window matrix; in recursive forecasting the window entries at rows >= the
issuance row are earlier PREDICTIONS, never realized values. Price features use the day-ahead price of row r (known
for the whole day before it starts) and of rows r-1 and r-96 (past).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TARGET_LAGS = (1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 672)
ROLL_WINDOWS = (4, 12, 24, 96, 672)
WINDOW = 672                       # rows of history a feature row needs (one week of 15-minute rows)
CALENDAR_FEATURES = ('hour', 'minute', 'dow', 'is_weekend', 'slot_sin', 'slot_cos', 'dow_sin', 'dow_cos', 'doy_sin',
                     'doy_cos')
PRICE_FEATURES = ('price', 'price_lag_1', 'price_lag_96')
HISTORY_FEATURES = tuple([f'y_lag_{k}' for k in TARGET_LAGS] +
                         [f'y_roll_{s}_{w}' for w in ROLL_WINDOWS for s in ('mean', 'std', 'min', 'max')] +
                         ['lag_96_minus_lag_672'])
FEATURES = CALENDAR_FEATURES + PRICE_FEATURES + HISTORY_FEATURES
LABEL = 'y'


def history_features(window):
    """window: [n, 672]; window[:, WINDOW - k] is the target value k rows before the target row (k = 1 .. 672)."""
    window = np.asarray(window, dtype=float)
    if window.ndim != 2 or window.shape[1] != WINDOW:
        raise ValueError(f'history window must have shape [n, {WINDOW}], got {window.shape}')
    if not np.isfinite(window).all():
        raise ValueError('nonfinite value in a history window')
    out = {f'y_lag_{k}': window[:, WINDOW - k] for k in TARGET_LAGS}
    for w in ROLL_WINDOWS:
        seg = window[:, WINDOW - w:]                                        # rows r-w .. r-1
        out[f'y_roll_mean_{w}'] = seg.mean(axis=1)
        out[f'y_roll_std_{w}'] = seg.std(axis=1, ddof=1)                    # pandas rolling std convention
        out[f'y_roll_min_{w}'] = seg.min(axis=1)
        out[f'y_roll_max_{w}'] = seg.max(axis=1)
    out['lag_96_minus_lag_672'] = out['y_lag_96'] - out['y_lag_672']
    return out


def calendar_features(timestamps):
    """Calendar features of the TARGET rows (timezone-naive timestamps, the environment's own labels)."""
    t = pd.DatetimeIndex(timestamps)
    if t.tz is not None:
        raise ValueError('calendar features expect timezone-naive timestamps')
    hour, minute, dow = t.hour.to_numpy(), t.minute.to_numpy(), t.dayofweek.to_numpy()
    slot = hour * 4 + minute // 15
    doy = t.dayofyear.to_numpy()
    return dict(hour=hour, minute=minute, dow=dow, is_weekend=np.isin(dow, (5, 6)).astype(int),
                slot_sin=np.sin(2 * np.pi * slot / 96), slot_cos=np.cos(2 * np.pi * slot / 96),
                dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7),
                doy_sin=np.sin(2 * np.pi * doy / 366), doy_cos=np.cos(2 * np.pi * doy / 366))


def price_features(price, rows):
    rows = np.asarray(rows, dtype=int)
    if rows.size and rows.min() < 96:
        raise ValueError('price_lag_96 needs rows >= 96')
    price = np.asarray(price, dtype=float)
    return dict(price=price[rows], price_lag_1=price[rows - 1], price_lag_96=price[rows - 96])


def feature_frame(window, timestamps, price, rows):
    """Feature rows (columns FEATURES, in order) for target rows `rows` with history `window` [len(rows), 672]."""
    rows = np.asarray(rows, dtype=int)
    window = np.asarray(window, dtype=float)
    if len(window) != len(rows) or len(timestamps) != len(rows):
        raise ValueError('window, timestamps and rows must have the same length')
    columns = dict(calendar_features(timestamps), **price_features(price, rows), **history_features(window))
    return pd.DataFrame({name: columns[name] for name in FEATURES})


def training_window(y, rows):
    """Realized history windows y[r-672 : r] of target rows `rows` (each r >= 672); a view, nothing is copied."""
    y = np.asarray(y, dtype=float)
    rows = np.asarray(rows, dtype=int)
    if rows.size and (rows.min() < WINDOW or rows.max() >= len(y)):
        raise ValueError(f'training rows must lie in [{WINDOW}, {len(y)})')
    return sliding_window_view(y, WINDOW)[rows - WINDOW]
