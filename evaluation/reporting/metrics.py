"""Observed-decision denominators; no reward-to-dollar conversion.

Missing = absent/null/NaN. Invalid = malformed or infinite. Both are excluded
from numerical denominators and reported separately. A malformed array invalidates
the whole row; individual missing entries invalidate only those battery decisions.
"""
import json
import numpy as np
import pandas as pd


def arrays(values, n):
    data, missing, invalid = [], [], []
    for value in values:
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            a = np.full(n, np.nan)
        else:
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                a = np.full(n, np.nan) if parsed is None else np.asarray(parsed, dtype=float).reshape(-1)
                if len(a) != n:
                    raise ValueError('wrong battery count')
            except (ValueError, TypeError, json.JSONDecodeError):
                a = np.full(n, np.inf)
        data.append(a); missing.append(np.isnan(a)); invalid.append(np.isinf(a))
    return (np.asarray(data, dtype=float).reshape(-1, n),
            np.asarray(missing, dtype=bool).reshape(-1, n),
            np.asarray(invalid, dtype=bool).reshape(-1, n))


def column(frame, key, n):
    return arrays(frame[key] if key in frame else [None] * len(frame), n)


def ratio(num, den):
    return float(num / den) if den > 0 else float('nan')


def power_metrics(frame, n):
    requested, rm, ri = column(frame, 'policy_requested_power_kw', n)
    physical, pm, pi = column(frame, 'physical_limited_power_kw', n)
    invalid = ri | pi
    missing = (rm | pm) & ~invalid
    valid = ~(invalid | missing)
    delta = np.abs(requested[valid] - physical[valid])
    numerator, denominator = float(delta.sum()), float(np.abs(requested[valid]).sum())
    return dict(battery_decisions=int(valid.size), power_valid_decisions=int(valid.sum()),
        power_missing_decisions=int(missing.sum()), power_invalid_decisions=int(invalid.sum()),
        requested_abs_kw_sum=denominator, removed_abs_kw_sum=numerator,
        material_corrected_decisions=int((delta > 1.0).sum()),
        requested_power_removed_fraction=ratio(numerator, denominator),
        material_correction_fraction=ratio((delta > 1.0).sum(), valid.sum()),
        mean_soc_projection_kw=ratio(numerator, valid.sum()),
        power_denominator_zero=bool(valid.any() and denominator == 0))


def observed_rate(values):
    x = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
    valid = np.isfinite(x) & np.isin(x, [0, 1])
    return ratio(x[valid].sum(), valid.sum()), int(valid.sum()), int((~valid).sum())


def episode_metrics(frame, batteries, price_by_start, expected_steps=96, dt=.25):
    """One row even for reset failure. Costs undefined unless all day data valid.

    Adjusted cost = raw cost - mean(actual day price)*(E_end-E_start)/1000.
    This is stored-side inventory valuation, not an efficiency-adjusted recharge.
    price_by_start is extracted from the evaluated env's actual scaled profile.
    """
    n = len(batteries)
    caps = np.array([b['capacity_kwh'] for b in batteries])
    mins = np.array([b['soc_min'] for b in batteries])
    rows = []
    for (tp, ep, start), all_rows in frame.groupby(['topology_case', 'episode_id', 'start_index'], dropna=False):
        g = all_rows[all_rows.record_type == 'dispatch_attempt'] if 'record_type' in all_rows else all_rows
        g = g.sort_values('step')
        if g.step.duplicated().any():
            raise ValueError('Duplicate decision within episode')
        failure_flags = pd.to_numeric(all_rows.get('episode_failed', pd.Series([np.nan])), errors='coerce')
        failed = bool(failure_flags.eq(1).any())
        failure_known = bool(failure_flags.isin([0, 1]).all())
        completed_flag = all_rows.get('episode_completed', pd.Series([False])).eq(True).all()
        complete = bool(failure_known and not failed and completed_flag and len(g) == expected_steps
                        and set(g.step) == set(range(expected_steps)))
        r = dict(topology_case=tp, episode_id=ep, start_index=int(start), intended_days=1,
                 completed_days=int(complete), failed_days=int(failed), incomplete_days=int(not complete),
                 failure_status_unknown_days=int(not failure_known),
                 observed_steps=len(g), **power_metrics(g, n))
        prices = np.asarray(price_by_start[int(start)], float)
        if len(prices) != expected_steps:
            raise ValueError('Price schedule is not a complete day')
        stamp = all_rows.get('date_time', pd.Series([None])).iloc[0]
        r['date'] = str(stamp)[:10] if pd.notna(stamp) else None
        grid = pd.to_numeric(g.get('grid_import_kw', pd.Series(index=g.index, dtype=float)), errors='coerce').to_numpy(float)
        durations = pd.to_numeric(g.get('delta_t_hours', pd.Series(dt, index=g.index)), errors='coerce').to_numpy(float)
        cost_ok = complete and np.isfinite(grid).all() and np.isfinite(prices).all() and np.allclose(durations, dt)
        cost = float(np.sum(np.maximum(grid, 0) * prices * durations / 1000)) if cost_ok else np.nan
        initial = column(g.iloc[:1], 'soc_before', n)[0][0] if len(g) else np.full(n, np.nan)
        terminal = column(g.iloc[-1:], 'soc_after', n)[0][0] if complete else np.full(n, np.nan)
        initial[~np.isfinite(initial)] = np.nan
        terminal[~np.isfinite(terminal)] = np.nan
        e0, e1 = float(initial @ caps), float(terminal @ caps)
        shortfall = np.divide(np.maximum(0, initial-terminal), initial-mins,
                              out=np.zeros(n), where=initial-mins > 1e-9)
        if not (np.isfinite(initial).all() and np.isfinite(terminal).all()):
            shortfall[:] = np.nan
        rate, valid, unknown = observed_rate(g.get('final_network_feasible', [np.nan] * len(g)))
        adjusted = cost-float(prices.mean())*(e1-e0)/1000 if np.isfinite(prices).all() else np.nan
        r.update(raw_energy_cost=cost, adjusted_energy_cost=adjusted,
                 initial_stored_energy_kwh=e0, terminal_stored_energy_kwh=e1,
                 terminal_soc_shortfall=float(np.mean(shortfall)), network_feasibility_rate=rate,
                 network_valid_steps=valid, network_unknown_steps=unknown)
        for i in range(n):
            r[f'initial_soc_{i}'], r[f'terminal_soc_{i}'] = initial[i], terminal[i]
        rows.append(r)
    return pd.DataFrame(rows)


METRICS = ['raw_energy_cost', 'adjusted_energy_cost', 'terminal_soc_shortfall',
           'terminal_stored_energy_kwh', 'requested_power_removed_fraction',
           'material_correction_fraction', 'network_feasibility_rate']


def summary(frame):
    """Explicit coverage for each surviving-day mean; ratios use pooled sums."""
    out = {k: int(frame[k].sum()) for k in ('intended_days', 'completed_days', 'failed_days', 'incomplete_days',
        'power_valid_decisions', 'power_missing_decisions', 'power_invalid_decisions')}
    out['failure_status_unknown_days'] = int(frame['failure_status_unknown_days'].sum())
    for name in METRICS:
        x = pd.to_numeric(frame[name], errors='coerce'); valid = np.isfinite(x)
        out[name+'_observed_day_mean'] = float(x[valid].mean()) if valid.any() else np.nan
        out[name+'_valid_days'] = int(valid.sum())
        out[name+'_missing_days'] = int((~valid).sum())
    out['requested_power_removed_fraction_pooled'] = ratio(frame.removed_abs_kw_sum.sum(), frame.requested_abs_kw_sum.sum())
    out['material_correction_fraction_pooled'] = ratio(frame.material_corrected_decisions.sum(), frame.power_valid_decisions.sum())
    return out
