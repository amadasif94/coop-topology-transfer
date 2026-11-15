"""Reporting/acceptance tolerances, independent of the AC solver tolerance.

Voltage: 1e-6 p.u. outside the physical limit. Current: 1e-6 of EACH
effective branch rating (0.000255 A at 255 A), invariant to the current base.
Raw excesses are retained; only classification applies this tolerance.
"""
import numpy as np

SHARED_NUMERICAL_TOLERANCE = 1e-6
VOLTAGE_TOLERANCE_PU = SHARED_NUMERICAL_TOLERANCE
CURRENT_RELATIVE_TOLERANCE = SHARED_NUMERICAL_TOLERANCE
# Dimensionless comparison of already-normalized backup scores; not solver precision.
BACKUP_SCORE_TOLERANCE = SHARED_NUMERICAL_TOLERANCE


def voltage_excess(voltage, v_min=.95, v_max=1.05):
    voltage = np.asarray(voltage, dtype=float)
    return np.maximum(v_min-voltage, 0), np.maximum(voltage-v_max, 0)


def current_overloaded(current, limits):
    current, limits = np.asarray(current), np.asarray(limits)
    return np.abs(current) > limits * (1 + CURRENT_RELATIVE_TOLERANCE)


def feasible(voltage, current, limits, converged, v_min=.95, v_max=1.05,
             voltage_enabled=True, current_enabled=True):
    if not converged or not np.isfinite(voltage).all() or not np.isfinite(current).all():
        return False
    if current_enabled and limits is not None:
        limits = np.asarray(limits, dtype=float)
        if limits.shape != np.asarray(current).shape or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("Invalid effective branch-current ratings")
    low, high = voltage_excess(voltage, v_min, v_max)
    return bool((not voltage_enabled or np.all(np.maximum(low, high) <= VOLTAGE_TOLERANCE_PU))
                and (not current_enabled or (limits is not None and
                     not np.any(current_overloaded(current, limits)))))
