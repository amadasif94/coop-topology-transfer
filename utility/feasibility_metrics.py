"""B3 per-episode diagnostics from an evaluation step_metrics frame (identical definitions for every controller).

All rates are over dispatch agent-steps (num_agents x attempted steps) of the episode, so MLP and GNN controllers
share denominators. Battery SoC projection (requested -> SoC-feasible power) is kept separate from the network
backup (SoC-feasible -> executed power).
  b3_feasibility_loss            mean over agent-steps of (a - clip(a, l, h))^2, raw submitted action a
  b3_outside_bounds_fraction     share of agent-steps with a outside [l, h]
  b3_mean_soc_projection_kw      mean |requested - SoC-feasible| kW
  b3_material_clipping_fraction  share of agent-steps with |requested - SoC-feasible| > MATERIAL_KW
  b3_boundary_occupancy          share of agent-steps whose post-step SoC is within SOC_EDGE of s_min or s_max
  b3_max_infeasible_discharge_streak  longest run of consecutive steps (any one agent) with a > h
  b3_mean_backup_change_kw       mean |executed - SoC-feasible| kW (network backup only)
  b3_backup_activation_rate      share of steps with a backup activation
  b3_terminal_shortfall_mean / _max, b3_terminal_penalty, start/terminal SoC per agent (completed days only)
  b3_soc_near_min_fraction / _near_max_fraction  share of agent-steps with post-step SoC within SOC_NEAR of s_min / s_max
  b3_max_soc_boundary_streak     longest run of consecutive steps (any one agent) within SOC_NEAR of either SoC limit
  b3_charge_throughput_kwh / b3_discharge_throughput_kwh  executed (post-backup) battery energy over the day, all agents
  action_mapping                 'rated' or 'soc_feasible' (marl_action_mapping; a label, so not b3_-prefixed, because
                                 evaluation.evaluate.write_tables averages every b3_* column)
  b3_max_soc_boundary_streak_agent_<i>  the same streak for battery i alone
  b3_near_min_step_fraction      share of agent-steps whose PRE-action SoC (soc_before) is within SOC_NEAR of s_min
  b3_near_min_mean_raw_action_agent_<i>  mean raw actor action a of battery i over its near-s_min steps (NaN if none)
  b3_near_min_positive_raw_action_fraction  share of near-s_min agent-steps with a > 0 (raw action, NOT a kW request)
  b3_near_min_discharge_request_fraction    share of near-s_min agent-steps whose MAPPED request is a discharge
                                 (> MATERIAL_KW); expected 0 under soc_feasible (no discharge is available at s_min)
  action_mapping_version         'affine_v2' (current soc_feasible), 'zero_preserving_v1' (earlier soc_feasible runs)
                                 or 'rated' (label, never averaged)
Under soc_feasible mapping policy_requested_power_kw is the SOC-feasible request P_mapped, so b3_mean_soc_projection_kw
measures only the limiter's correction of P_mapped (expected ~0), and b3_mean_backup_change_kw the network backup.
Under the AFFINE soc_feasible mapping a raw action a = 0 is the midpoint of the available interval, not 0 kW, and a > 0
does not imply a discharge: kW quantities (marl_power_soc_mapped_kw, executed_battery_power_kw) carry the physical
meaning; raw-action columns describe what the actor output.
"""
import json

import numpy as np
import pandas as pd

MATERIAL_KW = 1.0
SOC_EDGE = 1e-6
SOC_NEAR = 1e-3        # "at or near" a SoC limit (0.1 percentage point)


def _matrix(series, n):
    rows = []
    for v in series:
        a = np.asarray(json.loads(v) if isinstance(v, str) else (v if v is not None else [np.nan] * n), dtype=float)
        rows.append(a.reshape(-1) if a.size == n else np.full(n, np.nan))
    return np.vstack(rows) if rows else np.zeros((0, n))


def _streak(mask):
    best = 0
    for column in mask.T:
        run = 0
        for value in column:
            run = run + 1 if value else 0
            best = max(best, run)
    return best


def episode_diagnostics(frame, env):
    n = env.num_agents
    soc_min = np.array([b.config.soc_min for b in env.batteries])
    soc_max = np.array([b.config.soc_max for b in env.batteries])
    rows = []
    for (tp, ep, start), g in frame.groupby(['topology_case', 'episode_id', 'start_index']):
        g = g[g['record_type'] == 'dispatch_attempt'].sort_values('step') if 'record_type' in g else g.sort_values('step')
        r = dict(topology_case=tp, episode_id=int(ep), start_index=int(start))
        raw = _matrix(g['policy_proposal_action_normalized'], n) if 'policy_proposal_action_normalized' in g else None
        if raw is not None and 'marl_feasible_action_lower' in g:
            lo, hi = _matrix(g['marl_feasible_action_lower'], n), _matrix(g['marl_feasible_action_upper'], n)
            v = raw - np.clip(raw, lo, hi)
            r.update(b3_feasibility_loss=float(np.nanmean(v ** 2)), b3_outside_bounds_fraction=float(np.nanmean(v != 0)),
                     b3_max_infeasible_discharge_streak=_streak(raw > hi))
        if 'policy_requested_power_kw' in g and 'physical_limited_power_kw' in g:
            proj = np.abs(_matrix(g['policy_requested_power_kw'], n) - _matrix(g['physical_limited_power_kw'], n))
            r.update(b3_mean_soc_projection_kw=float(np.nanmean(proj)),
                     b3_material_clipping_fraction=float(np.nanmean(proj > MATERIAL_KW)))
            if 'executed_battery_power_kw' in g:
                backup = np.abs(_matrix(g['executed_battery_power_kw'], n) - _matrix(g['physical_limited_power_kw'], n))
                r['b3_mean_backup_change_kw'] = float(np.nanmean(backup))
        if 'soc_after' in g:
            soc = _matrix(g['soc_after'], n)
            r['b3_boundary_occupancy'] = float(np.nanmean((soc <= soc_min + SOC_EDGE) | (soc >= soc_max - SOC_EDGE)))
            near_min, near_max = soc <= soc_min + SOC_NEAR, soc >= soc_max - SOC_NEAR
            r.update(b3_soc_near_min_fraction=float(np.nanmean(near_min)), b3_soc_near_max_fraction=float(np.nanmean(near_max)),
                     b3_max_soc_boundary_streak=_streak(near_min | near_max))
        if 'executed_battery_power_kw' in g:
            executed = _matrix(g['executed_battery_power_kw'], n)
            dt = float(env.delta_t_hours)
            r.update(b3_charge_throughput_kwh=float(np.nansum(np.maximum(-executed, 0.0)) * dt),
                     b3_discharge_throughput_kwh=float(np.nansum(np.maximum(executed, 0.0)) * dt))
        if 'soc_after' in g:
            for i in range(n):                                   # longest SoC-boundary streak of EACH battery
                r[f'b3_max_soc_boundary_streak_agent_{i}'] = _streak((near_min | near_max)[:, i:i + 1])
        if 'soc_before' in g:
            low = _matrix(g['soc_before'], n) <= soc_min + SOC_NEAR      # pre-action SoC at/near the minimum
            r['b3_near_min_step_fraction'] = float(np.nanmean(low))
            if raw is not None:
                # Raw actor action a while the battery starts the step near s_min (per battery; NaN if never).
                for i in range(n):
                    r[f'b3_near_min_mean_raw_action_agent_{i}'] = (float(np.nanmean(raw[low[:, i], i]))
                                                                    if low[:, i].any() else np.nan)
                r['b3_near_min_positive_raw_action_fraction'] = (float(np.nanmean(raw[low] > 0)) if low.any() else np.nan)
            mapped_col = 'marl_power_soc_mapped_kw' if 'marl_power_soc_mapped_kw' in g else 'policy_requested_power_kw'
            if mapped_col in g:
                # Discharge REQUEST in kW (after the action mapping) near s_min. Under the affine mapping a > 0 is not
                # a discharge request (at s_min every a maps to <= 0 kW), so this kW-based measure is the unambiguous one.
                mapped = _matrix(g[mapped_col], n)
                r['b3_near_min_discharge_request_fraction'] = (float(np.nanmean(mapped[low] > MATERIAL_KW))
                                                               if low.any() else np.nan)
        if 'marl_action_mapping' in g and len(g):
            r['action_mapping'] = str(g['marl_action_mapping'].iloc[0])     # label, outside the numeric b3_* family
        if 'marl_action_mapping_version' in g and len(g):
            r['action_mapping_version'] = str(g['marl_action_mapping_version'].iloc[0])     # label (never averaged)
        if 'backup_activated' in g:
            r['b3_backup_activation_rate'] = float(g['backup_activated'].astype(float).mean())
        last = g.iloc[-1] if len(g) else None
        if last is not None and bool(last.get('marl_terminal_completed', False)):
            d = _matrix([last['marl_terminal_soc_shortfall']], n)[0]
            r.update(b3_terminal_shortfall_mean=float(np.mean(d)), b3_terminal_shortfall_max=float(np.max(d)),
                     b3_terminal_penalty=float(last['marl_terminal_penalty']))
            start_soc = _matrix([last['marl_start_of_day_soc']], n)[0]
            for i in range(n):
                r[f'b3_start_soc_agent_{i}'] = float(start_soc[i])
                r[f'b3_terminal_soc_agent_{i}'] = float(last[f'soc_agent_{i}'])
        rows.append(r)
    return pd.DataFrame(rows)
