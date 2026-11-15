"""Explicit in-memory stress fixture; real PF, unchanged DOE/search/reward.

No CSV changes. Offline scenario construction is outside the search budget.
Uses empty batteries (charging available) and a tighter upstream current rating
to require a large reduction of DOE-conditioned charging, exercising major penalty.
"""
from unittest.mock import patch
import numpy as np
from environments.voltage_doe import apply_voltage_doe
from environments.directional_backup import evaluation_fields


def exercise_interventions(env):
    env.reset(seed=321)
    for b in env.batteries: b.soc = b.config.soc_min
    p, q = env.grid.get_base_load_kw().astype(float), env.grid.get_base_load_kvar().astype(float)
    pv = np.zeros(env.num_buses)
    lo, hi = 0., 2.
    for _ in range(20):
        mid = (lo+hi)/2
        zero = env._evaluate_candidate_action(np.zeros(env.num_agents), p*mid, q*mid, pv)
        if zero['pf_result']['voltage_pu'].min() < .958: hi = mid
        else: lo = mid
    p, q = p*hi, q*hi
    zero = env._evaluate_candidate_action(np.zeros(env.num_agents), p, q, pv)
    cr = np.array([b.config.max_charge_kw for b in env.batteries])
    dr = np.array([b.config.max_discharge_kw for b in env.batteries])
    ac = np.array([b.get_available_charge_kw() for b in env.batteries])
    ad = np.array([b.get_available_discharge_kw() for b in env.batteries])
    power, diag = apply_voltage_doe(-np.minimum(cr, ac), zero['pf_result']['voltage_pu'][env.dess_buses],
        cr, dr, ac, ad, w=env.doe_w, delta_c=env.doe_delta_c)
    assert np.any(diag['doe_preventive_clipped'])
    candidate = env._evaluate_candidate_action(env._power_to_action(power), p, q, pv)
    z = zero['pf_result']['line_current_pu'][0]
    c = candidate['pf_result']['line_current_pu'][0]
    assert c > z
    limits = env.grid.get_line_current_limits_pu().copy()
    limits[0] = z + .05*(c-z)
    price, date = env._get_profiles(env.current_index)[3:]
    env.last_pf_result = zero['pf_result']
    env.current_kpis = env._compute_kpis(zero['pf_result'], price, 0.)
    env.previous_kpis = dict(env.current_kpis)
    original_reward = env._calculate_rewards
    observed = {}
    def check_reward(**kwargs):
        actual = original_reward(**kwargs)
        base = original_reward(**dict(kwargs, infeasible_action=False, backup_penalty_applied=False))
        observed['penalty'] = bool(kwargs['backup_penalty_applied'])
        if observed['penalty']:
            np.testing.assert_allclose(actual-base, env.reward_cfg['infeasible_action_penalty'], atol=1e-4)
        return actual
    try:
        with patch.object(env.grid, 'line_current_limits_pu', limits), \
             patch.object(env, '_get_profiles', return_value=(p,q,pv,price,date)), \
             patch.object(env, '_calculate_rewards', side_effect=check_reward):
            _, reward, term, trunc, info = env.step(-np.ones(env.num_agents))
        assert info['doe_enabled'] and np.any(info['doe_preventive_clipped'])
        assert info['backup_activated'] and info['major_backup_penalty_applied'] and observed['penalty']
        assert info['backup_pf_evaluations'] <= 20 and info['step_pf_failed_solves'] == 0
        assert np.isfinite(reward).all() and info['final_network_feasible']
        assert not np.allclose(info['marl_prepared_action'], info['doe_conditioned_action'])
        assert not np.allclose(info['doe_conditioned_action'], info['accepted_action'])
        return dict(fixture='synthetic_upstream_rating', **evaluation_fields(info))
    finally:
        env.reset(seed=321)
