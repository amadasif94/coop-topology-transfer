"""Local voltage envelope in kW. No network model or battery state mutation."""
import numpy as np


def validate_parameters(w, delta_c, v_min, v_max):
    if not np.all(np.isfinite([w, delta_c, v_min, v_max])):
        raise ValueError("DOE parameters must be finite.")
    if w < 0 or delta_c <= 0 or v_min >= v_max:
        raise ValueError("DOE requires w >= 0, delta_c > 0 and v_min < v_max.")
    # The preventive margins of width w sit inside [v_min, v_max] at both limits. With
    # w >= half the band they overlap, so no legal voltage permits both full charge and
    # full discharge. delta_c has no physical upper bound (larger = gentler correction).
    # 1e-12 absorbs float rounding: (1.05 - 0.95) / 2 == 0.05000000000000004.
    if w >= (v_max - v_min) / 2 - 1e-12:
        raise ValueError(f"DOE preventive width w={w} must be < (v_max - v_min)/2 = {(v_max - v_min) / 2}.")


DOE_KEYS = ("doe_enabled", "doe_w", "doe_delta_c", "pf_failure_mode", "doe_penalty_coef")


def validate_doe_config(doe, v_min, v_max):
    """Reject DOE configurations that would silently change behaviour.

    Unknown keys (typos), non-boolean doe_enabled (bool('False') is True), non-numeric or
    boolean numbers, unknown PF-failure modes, and a DOE penalty with DOE disabled.
    Parameters are validated whether or not DOE is enabled.
    """
    unknown = sorted(set(doe) - set(DOE_KEYS))
    if unknown:
        raise ValueError(f"Unknown DOE configuration key(s) {unknown}; allowed: {list(DOE_KEYS)}")
    enabled = doe.get("doe_enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError(f"doe_enabled must be a boolean, got {enabled!r} ({type(enabled).__name__})")
    for key, default in (("doe_w", 0.015), ("doe_delta_c", 0.010), ("doe_penalty_coef", 0.0)):
        value = doe.get(key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{key} must be a real number, got {value!r}")
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value!r}")
    if doe.get("pf_failure_mode", "strict") not in ("strict", "logged_fallback"):
        raise ValueError("Unknown DOE pf_failure_mode")
    if not bool(enabled) and float(doe.get("doe_penalty_coef", 0.0)) > 0:
        raise ValueError("doe_penalty_coef > 0 requires doe_enabled=True")
    validate_parameters(float(doe.get("doe_w", 0.015)), float(doe.get("doe_delta_c", 0.010)), v_min, v_max)


def apply_voltage_doe(power_kw, voltage_pu, charge_rating_kw,
                      discharge_rating_kw, available_charge_kw,
                      available_discharge_kw, *, w, delta_c,
                      v_min=0.95, v_max=1.05, pf_converged=True):
    """Filter physically limited proposals; return powers and per-battery flags.

    Correction is an absolute minimum magnitude, never an increment. Shortfall
    is physical unavailability at this layer, before the AC backup acts.
    Invalid measurement raises before any battery state can be committed.
    """
    validate_parameters(w, delta_c, v_min, v_max)
    arrays = [np.asarray(a, dtype=float) for a in (
        power_kw, voltage_pu, charge_rating_kw, discharge_rating_kw,
        available_charge_kw, available_discharge_kw)]
    p, v, cr, dr, ac, ad = arrays
    if p.ndim != 1 or any(a.shape != p.shape for a in arrays):
        raise ValueError("DOE inputs must be equal-length one-dimensional arrays.")
    if not pf_converged or not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("DOE requires finite inputs and converged pre-action PF.")
    if any(np.any(a < 0) for a in (cr, dr, ac, ad)):
        raise ValueError("Battery ratings and available powers must be nonnegative.")
    ac, ad = np.minimum(ac, cr), np.minimum(ad, dr)
    p = np.clip(p, -ac, ad)
    out = p.copy()
    legal = (v >= v_min) & (v <= v_max)
    if w > 0:
        c = cr * np.clip((v - v_min) / w, 0, 1)
        d = dr * np.clip((v_max - v) / w, 0, 1)
        out[legal] = np.clip(p, -c, d)[legal]
    low, high = v < v_min, v > v_max
    required = np.zeros_like(p)
    required[low] = dr[low] * np.clip((v_min - v[low]) / delta_c, 0, 1)
    required[high] = cr[high] * np.clip((v[high] - v_max) / delta_c, 0, 1)
    out[low] = np.maximum(p[low], np.minimum(required[low], ad[low]))
    out[high] = np.minimum(p[high], -np.minimum(required[high], ac[high]))
    shortfall = np.zeros_like(p)
    shortfall[low] = np.maximum(required[low] - ad[low], 0)
    shortfall[high] = np.maximum(required[high] - ac[high], 0)
    changed = np.abs(out - p) > 1e-9
    return out, {
        "doe_preventive_clipped": legal & changed,
        "doe_corrective_active": low | high,
        "doe_corrective_intervened": (low | high) & changed,
        "doe_corrective_required_kw": required,
        "doe_corrective_shortfall_kw": shortfall,
    }
