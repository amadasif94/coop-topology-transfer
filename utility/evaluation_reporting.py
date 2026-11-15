"""Additive evaluation accounting. Failed resets are event rows, not dispatches.

Rates use explicitly reported measured-step denominators; PF counts sum per-step
failed solves (not cumulative counters). Episode completeness requires a valid
full horizon. Legacy economic columns remain unchanged for valid episodes.
"""
import json
import re
import numpy as np
import pandas as pd
from training.transition_semantics import finite_tree
from environments.directional_backup import evaluation_fields


def pf_exception_info(exc, stage):
    """Recognize numerical PF exceptions; unrelated programming errors propagate."""
    if not isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)) and not re.search(
            r'\bpf\b|power.?flow|converg|singular', str(exc), re.I):
        raise exc
    return dict(episode_failed=True, observation_valid=False, transition_valid=False,
                converged=False, feasible=False, final_network_feasible=False,
                terminated=False, truncated=True, dispatch_executed=False,
                pf_failure_stage=stage, pf_failure_reason=str(exc), evaluation_exception=True)


def reset_for_evaluation(env, start_index, seed):
    # Random-baseline actions come from env.action_space.sample(), whose RNG env.reset()
    # does not seed; seed it per episode so 'random' evaluations are reproducible.
    action_space = getattr(env, 'action_space', None)
    if seed is not None and hasattr(action_space, 'seed'):
        action_space.seed(int(seed))
    old_starts = env.episode_start_indices
    env.episode_start_indices = [int(start_index)]
    try:
        return env.reset(seed=seed)
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        info = pf_exception_info(exc, 'reset')
        info['pf_failed_solve_count'] = getattr(env, 'pf_failed_solve_count', None)
        return {}, info
    finally:
        env.episode_start_indices = old_starts


def observation_ready(obs, info):
    return bool(not info.get('episode_failed', False) and info.get('observation_valid', True)
                and info.get('converged', True) and finite_tree(obs))


def failure_row(env, info, episode_id, start_index, policy=None, topology_case='TP1', controller=None):
    # NaN economics must not appear to be a measured zero-cost successful day.
    names = ('reward_mean reward_sum grid_import_kw grid_import_mwh energy_cost curtailment_kw '
             'curtailment_mwh voltage_deviation grid_stress min_voltage_pu max_voltage_pu '
             'max_line_current_pu max_voltage_violation max_line_current_violation throughput_mwh solve_time_sec').split()
    row = dict.fromkeys(names, np.nan)
    row.update(episode_id=int(episode_id), start_index=int(start_index), step=-1,
        date_time=info.get('date_time'), topology_case=topology_case, system=f'IEEE{env.num_buses}',
        policy=policy, controller=controller, feasible=False, converged=False,
        infeasible_action=None, record_type='reset_failure', episode_failed=True,
        observation_valid=False, dispatch_executed=False, transition_valid=False,
        terminated=False, truncated=True, final_network_feasible=False,
        pf_failure_stage=info.get('pf_failure_stage') or 'reset',
        evaluation_failure_reason=info.get('failure_reason') or 'invalid_reset_observation',
        step_pf_failed_solves=info.get('pf_failed_solve_count'),
        expected_episode_steps=env.episode_length, successful_evaluation_step=False)
    for key, value in evaluation_fields(info).items():
        if value is not None and key not in ('episode_failed', 'observation_valid', 'final_network_feasible'):
            row[key] = value
    for key in ('backup_activation_count', 'backup_penalty_applied_count',
                'backup_penalty_applied', 'zero_battery_feasible',
                'zero_battery_score', 'backup_initial_score'):
        row.setdefault(key, None)
    return finish_episode([row], env)


def _timestep_hours(env):
    value = getattr(env, 'delta_t_hours', None)
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else np.nan


def step_fields(info, env, terminated, truncated, next_obs):
    row = evaluation_fields(info)
    invalid = not observation_ready(next_obs, info)
    row.update(record_type='dispatch_attempt', terminated=bool(terminated), truncated=bool(truncated),
        episode_failed=bool(info.get('episode_failed', False) or invalid),
        observation_valid=not invalid, expected_episode_steps=env.episode_length,
        successful_evaluation_step=bool(info.get('dispatch_executed', False) and info.get('converged', False)),
        pf_failure_stage=info.get('pf_failure_stage') or ('next_observation' if invalid else ''),
        evaluation_failure_reason=('invalid_next_observation' if invalid else ''),
        delta_t_hours=_timestep_hours(env))  # F8.2 duration weighting
    if 'physical_limited_power_kw' in info:
        row['physical_limited_action_normalized'] = json.dumps(
            env._power_to_action(np.asarray(info['physical_limited_power_kw'])).tolist())
    return row


def finish_episode(rows, env):
    attempted = sum(r.get('record_type') == 'dispatch_attempt' for r in rows)
    successful = sum(bool(r.get('successful_evaluation_step', False)) for r in rows)
    failed = any(bool(r.get('episode_failed', False)) for r in rows)
    completed = bool(not failed and attempted == env.episode_length
                     and (rows[-1].get('terminated', False) or rows[-1].get('truncated', False)))
    for row in rows:
        row.update(episode_completed=completed, episode_incomplete=not completed,
                   episode_successful_steps=successful, episode_attempted_steps=attempted)
    return rows


def _numeric(df, key):
    return pd.to_numeric(df[key], errors='coerce') if key in df else pd.Series(np.nan, index=df.index)


def eligible_checkpoint_rows(df):
    """Never call incomplete/unknown runs safe or reclassify raw excesses.

    Feasibility is the environment's tolerance-aware classification. Raw maxima
    cannot be rethresholded without the applicable per-branch ratings.
    """
    required = ('intended_episodes', 'completed_episodes', 'failed_episodes',
                'incomplete_episodes', 'final_network_feasibility_rate', 'pf_failed_solves')
    if any(k not in df for k in required):
        return df.iloc[0:0].copy()
    keep = ((_numeric(df, 'intended_episodes') > 0)
            & (_numeric(df, 'completed_episodes') == _numeric(df, 'intended_episodes'))
            & (_numeric(df, 'failed_episodes') == 0) & (_numeric(df, 'incomplete_episodes') == 0)
            & (_numeric(df, 'final_network_feasibility_rate') == 1)
            & (_numeric(df, 'pf_failed_solves') == 0))
    return df.loc[keep].copy()


EPISODE_KEYS = ('system', 'topology_case', 'policy', 'controller', 'episode_id', 'start_index')


def _violation_duration_summary(df):
    """F8.2: duration of EXECUTED-state violations from evaluated steps (not just counts).

    *_violation_steps counts evaluated steps whose executed state violates; *_violation_hours
    weights each such step by its timestep length (delta_t_hours); longest_*_run_* is the
    longest consecutive violating stretch within one episode (runs never cross episodes).
    Steps without a measured executed state (failed dispatch) are excluded, not assumed safe.
    """
    out = {}
    dt = _numeric(df, 'delta_t_hours')
    keys = [k for k in EPISODE_KEYS if k in df]
    for kind in ('network', 'voltage', 'current'):
        col = f'executed_{kind}_violation'
        flags = _numeric(df, col)
        measured = int(flags.notna().sum())
        out[f'{kind}_violation_measured_steps'] = measured
        if not measured:
            for name in ('steps', 'hours', 'longest_run_steps', 'longest_run_hours'):
                out[f'{kind}_violation_{name}'] = np.nan
            continue
        hit = flags == 1
        out[f'{kind}_violation_steps'] = int(hit.sum())
        out[f'{kind}_violation_hours'] = float(dt.where(hit, 0.0).sum())
        longest_steps, longest_hours = 0, 0.0
        groups = df.groupby(keys, dropna=False, sort=False).groups.values() if keys else [df.index]
        for index in groups:
            run_s, run_h = 0, 0.0
            for i in index:
                if hit.loc[i]:
                    run_s, run_h = run_s + 1, run_h + float(dt.loc[i]) if np.isfinite(dt.loc[i]) else run_h
                else:
                    run_s, run_h = 0, 0.0
                longest_steps, longest_hours = max(longest_steps, run_s), max(longest_hours, run_h)
        out[f'{kind}_violation_longest_run_steps'] = longest_steps
        out[f'{kind}_violation_longest_run_hours'] = longest_hours
    return out


def diagnostic_summary(df, episode=False):
    out = {}
    def metric(key, label):
        x = _numeric(df, key)
        n = int(x.notna().sum())
        out[label+'_measured_steps'] = n
        out[label+'_count'] = float(x.sum(min_count=1)) if n else np.nan
        out[label+'_rate'] = float(x.mean()) if n else np.nan
    # Fractions of measured steps: physical clipping, DOE change, backup.
    metric('physical_clipping_any', 'physical_clipping')
    metric('doe_intervened', 'doe_intervention')
    metric('backup_activated', 'backup_activation')
    metric('backup_penalty_applied', 'backup_penalty_applied')
    # Legacy infeasible_requested_count is retained; directional mode counts activations.
    out['infeasible_requested_count_semantics'] = 'legacy: directional mode counts backup activations'
    causes = df.get('backup_cause', pd.Series(index=df.index, dtype=object))
    known = causes.isin(('none', 'action_induced', 'preexisting_not_worsened',
                        'preexisting_worsened', 'unknown'))
    n = int(known.sum())
    out['backup_cause_measured_steps'] = n
    # Rates use all steps with a reported cause (including none and unknown).
    for cause in ('none', 'action_induced', 'preexisting_not_worsened',
                  'preexisting_worsened', 'unknown'):
        count = int((causes[known] == cause).sum())
        out['backup_cause_'+cause+'_count'] = count if n else np.nan
        out['backup_cause_'+cause+'_rate'] = count/n if n else np.nan
    metric('major_backup_penalty_applied', 'major_backup_penalty')
    metric('final_network_feasible', 'final_network_feasibility')
    out.update(_violation_duration_summary(df))
    # F8.6 mean per-step timings (ms) and F8.4 peak loading; NaN when not recorded.
    for col in ('time_policy_ms', 'time_env_step_ms', 'time_doe_ms', 'time_backup_search_ms',
                'time_pf_ms', 'time_diagnostic_pf_ms', 'time_env_other_ms'):
        out['mean_' + col] = float(_numeric(df, col).mean()) if col in df else np.nan
    out['max_executed_line_loading_pct'] = float(_numeric(df, 'executed_max_line_loading_pct').max()) \
        if 'executed_max_line_loading_pct' in df else np.nan
    metric('executed_violating_bus_has_dess', 'executed_violation_at_dess_bus')
    metric('physical_proposal_feasible', 'physical_proposal_feasibility')
    pf = _numeric(df, 'backup_pf_evaluations')
    out.update(mean_backup_pf_evaluations=pf.mean(), max_backup_pf_evaluations=pf.max(),
        major_backup_penalty_amount=_numeric(df, 'major_backup_penalty_amount').sum(min_count=1),
        pf_failed_solves=_numeric(df, 'step_pf_failed_solves').sum(min_count=1))
    best, conv = _numeric(df, 'backup_best_effort'), _numeric(df, 'converged')
    measured = best.notna() & conv.notna()
    out['converged_infeasible_best_effort_steps'] = int(((best == 1) & (conv == 1)).sum()) if measured.any() else np.nan
    penalized = _numeric(df, 'major_backup_penalty_applied') == 1
    valid = _numeric(df, 'backup_zero_valid') == 1
    known = penalized & valid
    preexisting = known & (_numeric(df, 'backup_zero_infeasible') == 1)
    out.update(penalized_steps_with_valid_zero_reference=int(known.sum()),
        penalized_steps_with_preexisting_infeasibility=int(preexisting.sum()),
        penalized_steps_without_valid_zero_reference=int((penalized & ~valid).sum()),
        preexisting_infeasibility_percent_of_valid_penalized_steps=(100*preexisting.sum()/known.sum() if known.any() else np.nan))
    if episode:
        complete = bool(df['episode_completed'].iloc[-1])
        failed = bool((_numeric(df, 'episode_failed') == 1).any())
        out.update(intended_episodes=1, completed_episodes=int(complete), failed_episodes=int(failed),
                   incomplete_episodes=int(not complete), successful_evaluation_steps=int(df['episode_successful_steps'].iloc[-1]),
                   attempted_evaluation_steps=int(df['episode_attempted_steps'].iloc[-1]))
    return out


def augment_summaries(step_df, episode_summary, aggregate_summary):
    keys = [k for k in ('system', 'topology_case', 'policy', 'controller', 'episode_id', 'start_index') if k in episode_summary]
    extras = []
    for values, group in step_df.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        extras.append(dict(zip(keys, values), **diagnostic_summary(group, episode=True)))
    if not extras: return episode_summary, aggregate_summary
    episode_summary = episode_summary.merge(pd.DataFrame(extras), on=keys, validate='one_to_one')
    # Complete-episode reward/economic totals exist only for completed, failure-free
    # episodes. Failed or incomplete episodes (including partial days that ended in a PF
    # failure and its nonconvergence penalty) get NaN totals; step rows are unchanged.
    economic = [c for c in episode_summary if any(x in c for x in ('total_reward', 'energy_cost', 'grid_import_mwh', 'curtailment_mwh', 'throughput_mwh'))]
    economic_valid = (episode_summary['completed_episodes'] == 1) & (episode_summary['failed_episodes'] == 0)
    episode_summary['economic_totals_valid'] = economic_valid
    episode_summary.loc[~economic_valid, economic] = np.nan
    if 'steps' in episode_summary:
        episode_summary['steps'] = episode_summary['attempted_evaluation_steps']
    group_keys = [k for k in keys if k not in ('episode_id', 'start_index')]
    aggregates = []
    for values, group in step_df.groupby(group_keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        stats = diagnostic_summary(group)
        mask = pd.Series(True, index=episode_summary.index)
        aggregate_mask = pd.Series(True, index=aggregate_summary.index)
        for key, value in zip(group_keys, values):
            mask &= episode_summary[key].eq(value)
            aggregate_mask &= aggregate_summary[key].eq(value)
        episodes = episode_summary.loc[mask]
        # Do not let pandas' pre-augmentation empty sums make reset failures
        # look like zero-cost days in the existing aggregate economics.
        for column in aggregate_summary:
            if column.startswith(('mean_', 'std_')):
                operation, source = column.split('_', 1)
                source = source if source in episodes else 'total_'+source
                if source in economic:
                    value = episodes[source].mean() if operation == 'mean' else episodes[source].std()
                    aggregate_summary.loc[aggregate_mask, column] = value
        completed = episodes[episodes['completed_episodes'] == 1]
        for column in economic:
            stats['completed_only_mean_'+column] = completed[column].mean()
        # Denominator of every economic mean/std above (NaN-masked episodes excluded).
        stats['economic_valid_episodes'] = int(episodes['economic_totals_valid'].sum())
        stats['economic_excluded_episodes'] = int((~episodes['economic_totals_valid']).sum())
        last = group.groupby(keys, dropna=False).tail(1)
        stats.update(intended_episodes=len(last), completed_episodes=int(last['episode_completed'].sum()),
                     failed_episodes=int(group.groupby(keys, dropna=False)['episode_failed'].max().sum()),
                     incomplete_episodes=int(last['episode_incomplete'].sum()),
                     successful_evaluation_steps=int(last['episode_successful_steps'].sum()),
                     attempted_evaluation_steps=int(last['episode_attempted_steps'].sum()))
        aggregates.append(dict(zip(group_keys, values), **stats))
    aggregate_summary = aggregate_summary.merge(pd.DataFrame(aggregates), on=group_keys, validate='one_to_one')
    return episode_summary, aggregate_summary
