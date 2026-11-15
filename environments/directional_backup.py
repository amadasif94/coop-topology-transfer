"""Bounded, centralized AC-verified heuristic; not an optimizer or safety guarantee."""
import numpy as np
from utility.feasibility import VOLTAGE_TOLERANCE_PU, current_overloaded, BACKUP_SCORE_TOLERANCE


DEFAULTS = dict(backup_mode="legacy_scaling", pf_budget=20, refinement_steps=4,
                voltage_scale_pu=0.01, relative_current_scale=0.05,
                residual_penalty_enabled=False, residual_penalty_coefficient=0.0)


def settings(config):
    cfg = dict(DEFAULTS, **config)
    if cfg['backup_mode'] not in ('legacy_scaling', 'directional_incremental'):
        raise ValueError('Unknown backup_mode')
    if not isinstance(cfg['pf_budget'], int) or not 2 <= cfg['pf_budget'] <= 20:
        raise ValueError('pf_budget must be an integer from 2 to 20')
    if not isinstance(cfg['refinement_steps'], int) or not 0 <= cfg['refinement_steps'] <= 4:
        raise ValueError('refinement_steps must be an integer from 0 to 4')
    for key in ('voltage_scale_pu', 'relative_current_scale'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(key + ' must be finite and positive')
    if not np.isfinite(cfg['residual_penalty_coefficient']) or cfg['residual_penalty_coefficient'] < 0:
        raise ValueError('Residual penalty coefficient must be finite and nonnegative')
    return cfg


def valid_pf(pf):
    return bool(pf['converged']) and all(
        np.all(np.isfinite(pf[k])) for k in ('voltage_pu', 'line_current_pu',
                                           'grid_import_kw', 'grid_import_kvar'))


def violation_metrics(pf, limits, *, v_min=.95, v_max=1.05,
                      voltage_enabled=True, current_enabled=True,
                      voltage_scale_pu=.01, relative_current_scale=.05):
    """Score=max(max voltage excess/.01, max relative current excess/.05).

    Shared utility.feasibility tolerances define zero classified violations.
    Normalizations .01 p.u. and .05 relative overload are design choices. The mean
    normalized violation is a secondary tie breaker, not added to the score.
    Invalid/nonconverged PF has infinite score, never feasible.
    """
    v = np.asarray(pf.get('voltage_pu_raw', pf['voltage_pu']), dtype=float)
    current = np.abs(np.asarray(pf.get('line_current_pu_raw', pf['line_current_pu']), dtype=float))
    if current_enabled:
        if limits is None:
            raise ValueError('Directional backup requires effective line ratings')
        limits = np.asarray(limits, dtype=float)
        if limits.shape != current.shape or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
            raise ValueError('Invalid effective line ratings')
    good = valid_pf(pf)
    low, high = np.maximum(v_min-v, 0), np.maximum(v-v_max, 0)
    raw_low, raw_high = low.copy(), high.copy()
    low[low <= VOLTAGE_TOLERANCE_PU] = 0
    high[high <= VOLTAGE_TOLERANCE_PU] = 0
    overload = np.maximum(current-limits, 0) if current_enabled else np.zeros_like(current)
    raw_overload = overload.copy()
    if current_enabled:
        overload[~current_overloaded(current, limits)] = 0
    vr = np.maximum(low, high) / voltage_scale_pu if voltage_enabled else np.zeros_like(v)
    ir = overload / limits / relative_current_scale if current_enabled else np.zeros_like(current)
    score = max(float(np.max(vr, initial=0)), float(np.max(ir, initial=0))) if good else float('inf')
    return dict(score=score, secondary=float(np.mean(np.r_[vr, ir])) if good else float('inf'),
                valid=good, feasible=good and score == 0,
                raw_max_voltage_violation=float(np.max(np.maximum(raw_low, raw_high), initial=0)),
                raw_max_current_violation=float(np.max(raw_overload, initial=0)),
                min_voltage=float(np.min(v)), max_voltage=float(np.max(v)),
                max_line_current=float(np.max(current, initial=0)),
                undervoltage_count=int(np.count_nonzero(low)) if voltage_enabled else 0,
                overvoltage_count=int(np.count_nonzero(high)) if voltage_enabled else 0,
                overloaded_lines=int(np.count_nonzero(overload)),
                voltage_severity=float(np.max(vr, initial=0)) if good else float('inf'),
                undervoltage_severity=float(np.max(low/voltage_scale_pu, initial=0)) if good and voltage_enabled else 0.0,
                overvoltage_severity=float(np.max(high/voltage_scale_pu, initial=0)) if good and voltage_enabled else 0.0,
                current_severity=float(np.max(ir, initial=0)) if good else float('inf'))


def residual_penalty(final_score, zero_score, cfg):
    """Separate incremental residual charge; unavailable references incur none."""
    if not cfg['residual_penalty_enabled'] or not np.isfinite([final_score, zero_score]).all():
        return 0.0
    return -cfg['residual_penalty_coefficient'] * max(0.0, final_score-zero_score)


def search(initial, initial_power, charge_available, discharge_available, evaluate,
           measure, cfg):
    """evaluate(power)->candidate; initial PF already paid for, count starts at 1.

    Explore BOTH directions at all four fractions, even after finding feasibility.
    Refine only tested feasible brackets; never infer feasibility from direction.
    Cache identical powers. Keep the best measured candidate, never accept a
    worse trial. Bisection can miss disconnected feasible islands; no monotonic
    response or globally minimal correction is assumed.

    F12: every entry records its role ('initial', 'zero_reference', 'directional'),
    that it was physically evaluated (one AC power flow solved for exactly this
    power, or the caller's already-solved initial candidate), and the validity and
    feasibility measured on THAT solve. A cache hit returns the same evaluated entry
    (its status is never re-derived) and records the extra role. Only evaluated,
    valid entries are eligible for dispatch; the zero reference is eligible exactly
    when it is such an entry. "Fallback to zero" means the backup moved a nonzero
    proposal to an executed all-zero battery dispatch, not merely power == 0.
    """
    start = np.asarray(initial_power, dtype=float)
    entries = []
    cache = {}
    count = 1
    cache_hits = 0

    def remember(power, candidate, direction, alpha, role):
        metrics = measure(candidate['pf_result'])
        candidate['feasible'] = metrics['feasible']
        item = dict(power=np.array(power, copy=True), candidate=candidate,
                    metrics=metrics, direction=direction, alpha=alpha,
                    role=role, roles=[role], evaluated=True)
        entries.append(item)
        cache[tuple(np.round(power, 10))] = item
        return item

    first = remember(start, initial, 'unchanged', 0.0, 'initial')

    def trial(power, direction, alpha, role='directional'):
        nonlocal count, cache_hits
        key = tuple(np.round(power, 10))
        if key in cache:
            cache_hits += 1
            cache[key]['roles'].append(role)
            return cache[key]
        if count >= cfg['pf_budget']:
            return None
        count += 1
        return remember(power, evaluate(power), direction, alpha, role)

    def eligible(e):
        return bool(e['evaluated'] and e['metrics']['valid'])

    zero = None
    if not first['metrics']['feasible']:
        zero = trial(np.zeros_like(start), 'zero_reference', 1.0, role='zero_reference')
        # Voltage severity determines which direction gets budget priority.
        # Both directions are still tested for mixed violations/reverse flow.
        m = first['metrics']
        prefer_charge = (m['overvoltage_severity'] > m['undervoltage_severity']
                         if m['voltage_severity'] >= m['current_severity'] else
                         initial['pf_result']['grid_import_kw'] < 0)
        targets = [('discharge', np.asarray(discharge_available)),
                   ('charge', -np.asarray(charge_available))]
        if prefer_charge:
            targets.reverse()
        brackets = []
        # Interleave directions so a reduced explicit budget cannot consume
        # every trial on the first direction.
        tested = {name: [(0.0, first)] for name, _ in targets}
        for alpha in (.25, .5, .75, 1.0):
            for name, target in targets:
                item = trial(start + alpha*(target-start), name, alpha)
                if item is not None:
                    tested[name].append((alpha, item))
        for name, target in targets:
            feasible = [(a, x) for a, x in tested[name] if x['metrics']['feasible']]
            if feasible:
                hi, _ = min(feasible, key=lambda pair: pair[0])
                lo = max(a for a, x in tested[name] if a < hi)
                brackets.append([name, target, lo, hi])
        for _ in range(cfg['refinement_steps']):
            for bracket in brackets:
                name, target, lo, hi = bracket
                mid = (lo+hi)/2
                item = trial(start+mid*(target-start), name, mid)
                if item is None:
                    continue
                bracket[3 if item['metrics']['feasible'] else 2] = mid

    good = [e for e in entries if eligible(e)]
    feasible = [e for e in good if e['metrics']['feasible']]
    distance = lambda e: float(np.sum(np.abs(e['power']-start)))
    best = (min(feasible, key=distance) if feasible else
            min(good, key=lambda e: (e['metrics']['score'], e['metrics']['secondary'], distance(e))) if good else None)
    fm = best['metrics'] if best else first['metrics']
    zs = zero['metrics']['score'] if zero else float('nan')
    improved_zero = bool(np.isfinite(zs) and fm['score'] < zs)
    outcome = 'fixed' if feasible else 'partly fixed' if improved_zero else 'not resolved'
    diag = dict(backup_mode='directional_incremental', backup_pf_evaluations=count,
                backup_additional_pf_evaluations=count-1, backup_initial_score=first['metrics']['score'],
                backup_final_score=fm['score'], backup_zero_evaluated=zero is not None,
                backup_zero_score=zs, backup_improved_vs_zero=improved_zero,
                backup_not_worse_vs_zero=bool(np.isfinite(zs) and fm['score'] <= zs),
                backup_zero_comparison_valid=bool(np.isfinite(zs) and np.isfinite(fm['score'])),
                backup_fully_feasible=bool(feasible), backup_best_effort=bool(best and not feasible),
                backup_outcome=outcome, backup_direction=best['direction'] if best else 'none',
                backup_initial_power_kw=start.copy(),
                backup_adjustment_kw=best['power']-start if best else np.zeros_like(start),
                backup_residual_penalty=residual_penalty(fm['score'], zs, cfg),
                backup_failure_reason='' if best else 'no_valid_converged_candidate')
    diag['backup_invalid_pf_evaluations'] = sum(not e['metrics']['valid'] for e in entries)
    diag['backup_pf_failure_reasons'] = '|'.join(sorted({
        e['candidate']['pf_result'].get('failure_reason', 'nonconverged_or_nonfinite_power_flow')
        for e in entries if not e['metrics']['valid']}))
    for prefix, metrics in [('initial', first['metrics']), ('final', fm)]:
        for key in ('min_voltage', 'max_voltage', 'max_line_current', 'undervoltage_count',
                    'overvoltage_count', 'overloaded_lines'):
            diag['backup_'+prefix+'_'+key] = metrics[key]
    # F12: candidate status for reporting (dispatch selection is unchanged).
    executed_zero = bool(best and np.all(np.asarray(best['candidate'].get('dess_power_kw', best['power'])) == 0))
    diag.update(backup_selected_role=best['role'] if best else 'none',
                backup_selected_roles='|'.join(best['roles']) if best else '',
                backup_selected_evaluated=bool(best and best['evaluated']),
                backup_selected_executed_zero=executed_zero,
                backup_cache_hits=cache_hits,
                backup_zero_reference_role=('|'.join(zero['roles']) if zero else ''),
                backup_eligible_candidates=len(good),
                backup_evaluated_candidates=sum(e['evaluated'] for e in entries))
    if best:
        candidate = dict(best['candidate'])
        candidate.update(correction_applied=not first['metrics']['feasible'],
                         correction_iterations=int(np.ceil(best['alpha']/.25)),
                         correction_fallback_to_zero=bool(not first['metrics']['feasible'] and executed_zero
                                                          and np.any(start != 0)))
        return candidate, diag
    return None, diag


def legacy_diagnostics(initial, final, measure):
    """Read-only summary; no extra PF, reference solve or changed dispatch."""
    im, fm = measure(initial['pf_result']), measure(final['pf_result'])
    # F12: a zero fallback whose power flow failed (an unevaluated placeholder) is not a zero reference.
    zero = (final if final.get('correction_fallback_to_zero', False) and valid_pf(final['pf_result'])
            else initial if np.all(initial['dess_power_kw'] == 0) else None)
    zs = measure(zero['pf_result'])['score'] if zero is not None else float('nan')
    improved = bool(np.isfinite(zs) and fm['score'] < zs)
    result = dict(backup_initial_score=im['score'], backup_final_score=fm['score'],
        backup_zero_evaluated=zero is not None, backup_zero_score=zs,
        backup_zero_comparison_valid=bool(np.isfinite(zs) and np.isfinite(fm['score'])),
        backup_improved_vs_zero=improved, backup_fully_feasible=fm['feasible'],
        backup_not_worse_vs_zero=bool(np.isfinite(zs) and fm['score'] <= zs),
        backup_best_effort=False, backup_outcome='fixed' if fm['feasible'] else 'partly fixed' if improved else 'not resolved',
        backup_direction='legacy_scaling' if final.get('correction_applied', False) else 'unchanged',
        backup_residual_penalty=0.0)
    for prefix, metrics in [('initial', im), ('final', fm)]:
        for key in ('min_voltage', 'max_voltage', 'max_line_current', 'undervoltage_count',
                    'overvoltage_count', 'overloaded_lines'):
            result['backup_'+prefix+'_'+key] = metrics[key]
    return result


def backup_attribution(info):
    """Read-only cause of INITIAL DOE-conditioned infeasibility, not final quality.

    Scores already incorporate physical feasibility tolerances. One shared
    dimensionless tolerance is used for zero and initial-minus-zero comparisons.
    Missing/invalid references are unknown, never assumed safe. No extra PF.
    """
    activated = info.get('backup_activated')
    zero = info.get('backup_zero_score')
    valid = bool(info.get('backup_zero_evaluated', False) and zero is not None
                 and np.isfinite(zero))
    zero_feasible = bool(zero <= BACKUP_SCORE_TOLERANCE) if valid else None
    if activated is not None and not activated:
        cause = 'none'
    elif activated is None or not valid:
        cause = 'unknown'
    elif zero_feasible:
        cause = 'action_induced'
    else:
        initial = info.get('backup_initial_score')
        if initial is None or np.isnan(initial):
            cause = 'unknown'
        else:
            cause = ('preexisting_worsened' if initial > zero + BACKUP_SCORE_TOLERANCE
                     else 'preexisting_not_worsened')
    return dict(backup_cause=cause, zero_battery_feasible=zero_feasible,
                zero_battery_score=zero if zero is not None else None)


def evaluation_fields(info):
    """Flat CSV-compatible additive fields; safe for older environment outputs."""
    import json
    # F8 diagnostics: reward_term_*/reward_terms_total, time_*, physical_proposal_*,
    # doe_candidate_* and executed_* violation/loading fields, pf_solves_in_step.
    prefixes = ('backup_', 'pf_failure_', 'doe_', 'marl_', 'major_backup_',
                'reward_term', 'time_', 'physical_proposal_', 'executed_', 'pf_solves_in_step')
    keys = [k for k in info if k.startswith(prefixes)]
    keys += [k for k in ('episode_failed', 'failed_episode_count', 'observation_valid', 'transition_valid',
                        'dispatch_executed', 'observation_pf_evaluations', 'step_total_pf_evaluations',
                        'next_observation_pf_failed', 'pf_failed_solve_count', 'step_pf_failed_solves',
                        'terminated', 'truncated', 'bootstrap_unavailable_reason',
                        'pre_action_pf_failure', 'final_network_feasible',
                        'pre_action_local_voltage_pu', 'physical_limited_power_kw',
                        'soc_before', 'soc_after', 'final_voltage_pu', 'final_line_current_pu',
                        'final_battery_power_kw', 'physical_clipped', 'physical_clipped_kw',
                        'physical_clipping_any', 'physical_clipped_kw_total') if k in info]
    result = {k: json.dumps(info[k].tolist()) if isinstance(info[k], np.ndarray) else info[k] for k in keys}
    result.update(backup_attribution(info))
    result['backup_penalty_applied'] = info.get('backup_penalty_applied')
    # Step-level counts are 0/1 indicators, not cumulative episode totals.
    for source, name in [('backup_activated', 'backup_activation_count'),
                         ('backup_penalty_applied', 'backup_penalty_applied_count')]:
        result[name] = int(info[source]) if info.get(source) is not None else None
    result['infeasible_requested_count_semantics'] = (
        'legacy: counts backup activations in directional_incremental mode')
    # Additive stage labels. Legacy requested_action means the pre-backup,
    # DOE-conditioned candidate when DOE is enabled; never relabel it as policy.
    aliases = dict(policy_proposal_action_normalized='marl_raw_action',
        policy_prepared_action_normalized='marl_prepared_action',
        policy_requested_power_kw='marl_requested_power_kw',
        physical_limited_power_kw='physical_limited_power_kw',
        doe_action_normalized='doe_conditioned_action', doe_power_kw='doe_power_kw',
        backup_selected_action_normalized='accepted_action',
        executed_battery_power_kw='final_battery_power_kw')
    for label, source in aliases.items():
        value = info.get(source)
        result[label] = json.dumps(np.asarray(value).tolist()) if value is not None else None
    result['requested_action_semantics'] = 'pre_backup_candidate_DOE_conditioned_when_enabled'
    if 'doe_power_kw' in info and 'physical_limited_power_kw' in info:
        delta = np.asarray(info['doe_power_kw']) - np.asarray(info['physical_limited_power_kw'])
        result['doe_power_adjustment_kw'] = json.dumps(delta.tolist())
    else:
        result['doe_power_adjustment_kw'] = None
    indicators = ('doe_preventive_clipped', 'doe_corrective_intervened')
    result['doe_intervened'] = (bool(any(np.any(info[k]) for k in indicators))
                              if all(k in info for k in indicators) else None)
    # Stable schema for unavailable/older-controller outputs: missing, never zero.
    for key in ('doe_enabled', 'backup_activated', 'backup_direction', 'backup_initial_score',
                'backup_zero_score', 'backup_final_score', 'backup_pf_evaluations', 'backup_pf_budget',
                'backup_fully_feasible', 'backup_best_effort', 'backup_outcome',
                'major_backup_penalty_applied', 'major_backup_penalty_amount',
                'backup_zero_valid', 'backup_zero_infeasible', 'pf_failure_count',
                'step_pf_failed_solves', 'pf_failure_stage', 'episode_failed',
                'observation_valid', 'final_network_feasible'):
        result.setdefault(key, None)
    for stage in ('initial', 'final'):
        for kind in ('voltage', 'line_current'):
            source = ('requested_max_' if stage == 'initial' else 'max_') + kind + '_violation'
            result[f'backup_{stage}_max_{kind}_violation'] = info.get(source)
    return result
