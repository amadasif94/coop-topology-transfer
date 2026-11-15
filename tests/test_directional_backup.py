"""Deterministic heuristic tests plus real AC tests on BOTH local feeder files.

Fixtures explicitly use in-memory ratings/scenarios; never modify production CSVs.
No trained models, checkpoints or training runs are used.
"""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.directional_backup import settings, violation_metrics, search, residual_penalty
from environments.dess_env import DESSEnv
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG


def pf(v, current=(.5,), converged=True):
    return dict(voltage_pu=np.array(v, dtype=float), line_current_pu=np.array(current, dtype=float),
                converged=converged, grid_import_kw=1., grid_import_kvar=0.)


def run_rule(fn, start=0., low=75., high=75., budget=20):
    calls = []
    def evaluate(p):
        calls.append(p.copy())
        return dict(pf_result=fn(float(p[0])), accepted_action=p/75,
                    feasible=False, dess_power_kw=p.copy())
    initial = evaluate(np.array([start]))
    cfg = settings(dict(backup_mode='directional_incremental', pf_budget=budget))
    best, diag = search(initial, [start], [low], [high], evaluate,
        lambda result: violation_metrics(result, [1]), cfg)
    assert diag['backup_pf_evaluations'] == len(calls)
    assert len(calls) <= budget
    return best, diag, calls


class Rules(unittest.TestCase):
    def test_undervoltage_and_overvoltage(self):
        for sign in (1, -1):
            base = .94 if sign == 1 else 1.06
            best, diag, _ = run_rule(lambda p: pf([base + p*.0004]))
            self.assertTrue(best['feasible'])
            self.assertEqual(diag['backup_direction'], 'discharge' if sign == 1 else 'charge')
            self.assertLessEqual(abs(best['dess_power_kw'][0]), 27)

    def test_soc_saturation_best_effort_not_zero(self):
        best, diag, _ = run_rule(lambda p: pf([.93+p*.0001]), start=10, high=20)
        self.assertFalse(best['feasible'])
        self.assertEqual(best['dess_power_kw'][0], 20)
        self.assertTrue(diag['backup_best_effort'])
        self.assertEqual(diag['backup_outcome'], 'partly fixed')

    def test_opposing_voltage_and_simultaneous_current(self):
        for current in (False, True):
            fn = lambda p: pf([.94+p*.0002, 1.055+p*.0002],
                              [1.1+p*.003 if current else .5])
            best, diag, _ = run_rule(fn)
            self.assertLessEqual(diag['backup_final_score'], diag['backup_initial_score'])
            self.assertFalse(best['feasible'])

    def test_current_load_and_reverse_flow(self):
        for sign in (-1, 1):
            best, diag, _ = run_rule(lambda p: pf([1], [1.1+sign*p*.004]))
            self.assertTrue(best['feasible'])
            self.assertEqual(diag['backup_direction'], 'discharge' if sign == -1 else 'charge')

    def test_nonmonotonic_feasible_island(self):
        # 50% is feasible; 25%, 75%,100% are not. All coarse points must be checked.
        best, diag, _ = run_rule(lambda p: pf([1 if 33 <= p <= 40 else .94]))
        self.assertTrue(best['feasible'])
        self.assertGreaterEqual(best['dess_power_kw'][0], 33)
        self.assertLessEqual(best['dess_power_kw'][0], 40)

    def test_no_valid_pf_and_budget_cache(self):
        for budget in (2, 7, 20):
            best, diag, calls = run_rule(lambda p: pf([np.nan], converged=False), budget=budget)
            self.assertIsNone(best)
            self.assertEqual(diag['backup_outcome'], 'not resolved')
            self.assertEqual(len(calls), len({tuple(p) for p in calls}))
        with self.assertRaises(ValueError): settings(dict(pf_budget=21))

    def test_score_all_network_constraints_and_invalid(self):
        self.assertAlmostEqual(violation_metrics(pf([.94, 1.07], [1.15]), [1])['score'], 3)
        self.assertEqual(violation_metrics(pf([1], [.5]), [1])['score'], 0)
        for result in (pf([1], converged=False), pf([np.nan]), pf([1], [np.inf])):
            self.assertFalse(violation_metrics(result, [1])['feasible'])
            self.assertTrue(np.isinf(violation_metrics(result, [1])['score']))

    def test_zero_reference_attribution(self):
        cfg = settings(dict(residual_penalty_enabled=True, residual_penalty_coefficient=2))
        self.assertEqual(residual_penalty(2, 1, cfg), -2)
        self.assertEqual(residual_penalty(.5, 1, cfg), 0)
        self.assertEqual(residual_penalty(2, np.inf, cfg), 0)
        self.assertEqual(residual_penalty(2, 1, settings({})), 0)
        best, diag, _ = run_rule(lambda p: pf([.94-p*.0002]), start=30, low=0, high=0)
        self.assertEqual(best['dess_power_kw'][0], 0)
        self.assertTrue(diag['backup_zero_evaluated'])
        self.assertEqual(diag['backup_final_score'], diag['backup_zero_score'])


def fixture_config(original):
    c = copy.deepcopy(original)
    net = c['network']
    net['line_current_limit_mode'] = 'global_pu'
    net['global_line_current_limit_pu'] = 255/(net['s_base_kva']/(np.sqrt(3)*net['v_base_kv']))
    c['backup'] = dict(backup_mode='directional_incremental')
    c['doe']['pf_failure_mode'] = 'logged_fallback'
    return c


class RealAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [DESSEnv(fixture_config(c), seed=123) for c in (IEEE33_CONFIG, IEEE69_CONFIG)]

    def setUp(self):
        for env in self.envs:
            env.doe_enabled = False
            env.doe_pf_failure_mode = 'logged_fallback'
            env.reset(seed=123)

    def test_real_ac_uv_ov_stress(self):
        for env in self.envs:
            p = env.grid.get_base_load_kw().astype(float)
            q = env.grid.get_base_load_kvar().astype(float)
            def initial(scale, over):
                pv = np.zeros(env.num_buses)
                if over:
                    pv[env.dess_buses] = scale*75
                    load, reactive = p*.3, q*.3
                else:
                    load, reactive = p*scale, q*scale
                # Existing curtailment remains active. For the overvoltage stress
                # fixture only, allow export so PV can produce the desired stress.
                return load, reactive, pv
            for over in (False, True):
                old_export = env.grid_export_limit_kw
                try:
                    if over: env.grid_export_limit_kw = 100000.
                    lo, hi = 0., 20. if over else 2.
                    # Offline scenario construction is outside the backup budget.
                    for _ in range(18):
                        mid = (lo+hi)/2
                        load, reactive, pv = initial(mid, over)
                        c = env._evaluate_candidate_action(np.zeros(env.num_agents), load, reactive, pv)
                        v = c['pf_result']['voltage_pu']
                        stressed = np.max(v) > 1.051 if over else np.min(v) < .949
                        if stressed: hi = mid
                        else: lo = mid
                    load, reactive, pv = initial(hi, over)
                    first = env._evaluate_candidate_action(np.zeros(env.num_agents), load, reactive, pv)
                    before = env._snapshot_battery_states()
                    calls = []
                    def evaluate(power):
                        calls.append(power.copy())
                        return env._evaluate_candidate_action(env._power_to_action(power), load, reactive, pv)
                    best, diag = search(first, np.zeros(env.num_agents),
                        [b.get_available_charge_kw() for b in env.batteries],
                        [b.get_available_discharge_kw() for b in env.batteries],
                        evaluate, env._measure_violation, env.backup_cfg)
                    self.assertEqual(before, env._snapshot_battery_states())
                    self.assertLessEqual(len(calls)+1, 20)
                    self.assertLess(diag['backup_final_score'], diag['backup_initial_score'])
                    self.assertIsNotNone(best)
                    print('REAL_AC_STRESS', env.num_buses, 'OV' if over else 'UV',
                          diag['backup_initial_score'], diag['backup_final_score'], diag['backup_pf_evaluations'])
                finally:
                    env.grid_export_limit_kw = old_export

    def test_real_short_rollouts_and_pf_count(self):
        for env in self.envs:
            env.doe_enabled = True
            obs, _ = env.reset(seed=123)
            for i in range(8):
                pre = obs['agent_obs'][:, 3].copy()
                with patch.object(env.power_flow, 'run_power_flow', wraps=env.power_flow.run_power_flow) as solver:
                    obs, reward, term, trunc, info = env.step(np.full(env.num_agents, .9*(-1)**i))
                self.assertLessEqual(info['backup_pf_evaluations'], 20)
                self.assertEqual(solver.call_count, info['backup_pf_evaluations']+int(not term and not trunc))
                self.assertEqual(solver.call_count, info['step_total_pf_evaluations'])
                np.testing.assert_array_equal(pre, info['pre_action_local_voltage_pu'])
                self.assertTrue(np.isfinite(reward).all())
                self.assertTrue(np.isfinite(obs['agent_obs']).all())
                self.assertFalse(term)

    def test_real_ac_forward_and_reverse_current(self):
        for env in self.envs:
            p = env.grid.get_base_load_kw().astype(float)*.25
            q = env.grid.get_base_load_kvar().astype(float)*.25
            saved_limits = env.grid.line_current_limits_pu.copy()
            saved_export = env.grid_export_limit_kw
            try:
                env.grid_export_limit_kw = 100000.
                for reverse in (False, True):
                    pv = np.zeros(env.num_buses)
                    if reverse: pv[env.dess_buses] = (p.sum()+400)/env.num_agents
                    initial = env._evaluate_candidate_action(np.zeros(env.num_agents), p, q, pv)
                    # Deliberately lower ONLY the upstream branch rating in this
                    # controlled thermal fixture. Production ratings unchanged.
                    env.grid.line_current_limits_pu = saved_limits.copy()
                    env.grid.line_current_limits_pu[0] = initial['pf_result']['line_current_pu'][0]*.98
                    first = env._evaluate_candidate_action(np.zeros(env.num_agents), p, q, pv)
                    best, diag = search(first, np.zeros(env.num_agents),
                        [b.get_available_charge_kw() for b in env.batteries],
                        [b.get_available_discharge_kw() for b in env.batteries],
                        lambda power: env._evaluate_candidate_action(env._power_to_action(power), p, q, pv),
                        env._measure_violation, env.backup_cfg)
                    self.assertTrue(best['feasible'])
                    self.assertEqual(diag['backup_direction'], 'charge' if reverse else 'discharge')
                    self.assertLess(best['pf_result']['line_current_pu'][0], first['pf_result']['line_current_pu'][0])
                    print('REAL_AC_CURRENT', env.num_buses, 'reverse' if reverse else 'load',
                          diag['backup_initial_score'], diag['backup_final_score'], diag['backup_pf_evaluations'])
            finally:
                env.grid.line_current_limits_pu = saved_limits
                env.grid_export_limit_kw = saved_export

    def test_pre_action_strict_and_logged_recovery(self):
        for env in self.envs:
            env.doe_enabled = True
            env.doe_pf_failure_mode = 'strict'
            env.last_pf_result['converged'] = False
            before = env._snapshot_battery_states()
            with self.assertRaises(ValueError): env.step(np.zeros(env.num_agents))
            self.assertEqual(before, env._snapshot_battery_states())
            env.doe_pf_failure_mode = 'logged_fallback'
            obs, reward, term, trunc, info = env.step(np.zeros(env.num_agents))
            self.assertTrue(info['doe_skipped'])
            self.assertTrue(info['pre_action_pf_failure'])
            self.assertFalse(term)
            self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(np.isfinite(obs['agent_obs']).all())
            self.assertEqual(info['pf_failure_count'], 1)

    def test_pf_exceptions_restore_states_and_terminate(self):
        for env in self.envs:
            env.doe_enabled = True
            before = env._snapshot_battery_states()
            with patch.object(env.power_flow, 'run_power_flow', side_effect=RuntimeError('injected failure')) as solver:
                obs, reward, term, trunc, info = env.step(np.ones(env.num_agents))
            self.assertFalse(term)
            self.assertTrue(trunc)
            self.assertFalse(info['dispatch_executed'])
            self.assertFalse(info['final_network_feasible'])
            self.assertEqual(before, env._snapshot_battery_states())
            self.assertTrue(np.isfinite(obs['agent_obs']).all())
            self.assertLessEqual(solver.call_count, 20)
            self.assertEqual(info['step_pf_failed_solves'], solver.call_count)
            self.assertEqual(info['pf_failure_count'], 1)
            self.assertEqual(info['pf_failure_stage'], 'candidate_search')
            self.assertFalse(info['transition_valid'])
            np.testing.assert_allclose(reward, env.reward_cfg['nonconvergence_penalty'])

    def test_legacy_logged_recovery_and_strict_candidate_exception(self):
        env = self.envs[0]
        mode, robust = env.backup_mode, env._robust_pf
        try:
            env.backup_mode = 'legacy_scaling'
            env.doe_enabled = True
            env.last_pf_result['converged'] = False
            _, reward, term, _, info = env.step(np.zeros(env.num_agents))
            self.assertTrue(info['doe_skipped'])
            self.assertFalse(term)
            env.reset(seed=123)
            before = env._snapshot_battery_states()
            env._robust_pf = False
            p, q, pv, _, _ = env._get_profiles(env.current_index)
            with patch.object(env.power_flow, 'run_power_flow', side_effect=RuntimeError('strict')):
                with self.assertRaises(RuntimeError):
                    env._evaluate_candidate_action(np.ones(env.num_agents), p, q, pv)
            self.assertEqual(before, env._snapshot_battery_states())
        finally:
            env.backup_mode, env._robust_pf = mode, robust

    def test_reset_and_next_observation_failure(self):
        for env in self.envs:
            env.doe_enabled = True
            with patch.object(env.power_flow, 'run_power_flow', side_effect=RuntimeError('reset failure')):
                obs, info = env.reset(seed=123)
            self.assertTrue(info['episode_failed'])
            self.assertFalse(info['observation_valid'])
            self.assertTrue(np.isfinite(obs['agent_obs']).all())
            before = env._snapshot_battery_states()
            _, _, term, trunc, info = env.step(np.zeros(env.num_agents))
            self.assertFalse(term)
            self.assertTrue(trunc)
            self.assertEqual(before, env._snapshot_battery_states())
            env.reset(seed=123)
            original = env.power_flow.run_power_flow
            count = [0]
            def fail_next(**kw):
                count[0] += 1
                if count[0] == 2: raise RuntimeError('next observation failure')
                return original(**kw)
            with patch.object(env.power_flow, 'run_power_flow', side_effect=fail_next):
                obs, reward, term, trunc, info = env.step(np.zeros(env.num_agents))
            self.assertFalse(term)
            self.assertTrue(trunc)
            self.assertTrue(info['next_observation_pf_failed'])
            self.assertFalse(info['observation_valid'])
            self.assertTrue(np.isfinite(obs['agent_obs']).all())

    def test_time_limit_has_actual_next_timestep_for_bootstrap(self):
        for env in self.envs:
            old_length = env.episode_length
            try:
                env.episode_length = 1
                before_index = env.current_index
                obs, _, term, trunc, info = env.step(np.zeros(env.num_agents))
                self.assertFalse(term)
                self.assertTrue(trunc)
                self.assertTrue(info['transition_valid'])
                self.assertEqual(env.current_index, before_index + 1)
                np.testing.assert_array_equal(obs['agent_obs'][:, 3],
                    env.last_pf_result['voltage_pu'][env.dess_buses])
            finally:
                env.episode_length = old_length

    def test_time_limit_does_not_cross_data_split(self):
        for env in self.envs:
            old_length, old_splits = env.episode_length, env.splits
            try:
                env.episode_length = 1
                env.splits = dict(env.splits)
                env.splits[env.mode + '_indices'] = [env.current_index]
                before_index = env.current_index
                _, _, term, trunc, info = env.step(np.zeros(env.num_agents))
                self.assertFalse(term)
                self.assertTrue(trunc)
                self.assertFalse(info['transition_valid'])
                self.assertEqual(info['bootstrap_unavailable_reason'], 'end_of_data_split')
                self.assertEqual(env.current_index, before_index)
                self.assertFalse(info['episode_failed'])
            finally:
                env.episode_length, env.splits = old_length, old_splits

    def test_intervention_csv_preserves_failure_and_action_diagnostics(self):
        import io
        import pandas as pd
        from environments.directional_backup import evaluation_fields
        for env in self.envs:
            limits = env.grid.get_line_current_limits_pu().copy()
            try:
                # Explicit synthetic overload fixture; CSV files are untouched.
                env.grid.set_line_current_limits_pu(limits * .1)
                _, _, _, _, info = env.step(-np.ones(env.num_agents))
                self.assertTrue(info['backup_activated'])
                row = evaluation_fields(info)
                loaded = pd.read_csv(io.StringIO(pd.DataFrame([row]).to_csv(index=False)))
                for column in ('backup_mode', 'backup_initial_score', 'backup_zero_score',
                               'backup_final_score', 'marl_raw_action', 'doe_power_kw',
                               'final_battery_power_kw', 'major_backup_penalty_amount',
                               'terminated', 'truncated', 'transition_valid'):
                    self.assertIn(column, loaded.columns)
                    self.assertFalse(loaded[column].isna().any())
                self.assertTrue(bool(loaded['backup_zero_valid'].iloc[0]))
            finally:
                env.grid.set_line_current_limits_pu(limits)

    def test_major_penalty_preserved_no_doe_penalty(self):
        for env in self.envs:
            p, q, pv, price, date = env._get_profiles(env.current_index)
            first = env._evaluate_candidate_action(np.zeros(env.num_agents), p, q, pv)
            env.last_pf_result = first['pf_result']
            env.current_kpis = env._compute_kpis(first['pf_result'], price, first['curtailment_kw'])
            env.previous_kpis = dict(env.current_kpis)
            soc = np.array([b.soc for b in env.batteries])
            args = dict(price=price, dess_power_kw=first['dess_power_kw'], soc_before=soc, soc_after=soc)
            r0 = env._calculate_rewards(**args, infeasible_action=False)
            r1 = env._calculate_rewards(**args, infeasible_action=True)
            np.testing.assert_allclose(r1-r0, env.reward_cfg['infeasible_action_penalty'], atol=1e-4)
            env.doe_enabled = True
            np.testing.assert_array_equal(r0, env._calculate_rewards(**args, infeasible_action=False))

    def test_real_soc_bound_and_minor_penalty_configuration(self):
        for env in self.envs:
            p = env.grid.get_base_load_kw().astype(float)
            q = env.grid.get_base_load_kvar().astype(float)
            pv = np.zeros(env.num_buses)
            for b in env.batteries: b.soc = b.config.soc_min
            before = env._snapshot_battery_states()
            first = env._evaluate_candidate_action(np.zeros(env.num_agents), p, q, pv)
            best, diag = search(first, np.zeros(env.num_agents),
                [b.get_available_charge_kw() for b in env.batteries],
                [b.get_available_discharge_kw() for b in env.batteries],
                lambda power: env._evaluate_candidate_action(env._power_to_action(power), p, q, pv),
                env._measure_violation, env.backup_cfg)
            self.assertEqual(before, env._snapshot_battery_states())
            self.assertTrue(np.all(best['dess_power_kw'][env.dess_buses] <= 0))
            self.assertFalse(best['feasible'])
            self.assertEqual(diag['backup_outcome'], 'not resolved')
            # Construct a slight real undervoltage needing <50% available power.
            env.reset(seed=123)
            lo, hi = 0., 2.
            for _ in range(18):
                mid = (lo+hi)/2
                c = env._evaluate_candidate_action(np.zeros(env.num_agents), p*mid, q*mid, pv)
                if np.min(c['pf_result']['voltage_pu']) < .949: hi = mid
                else: lo = mid
            original_profiles = env._get_profiles
            _, _, _, price, date = original_profiles(env.current_index)
            original_flag = env.penalize_minor_corrections
            try:
                for flag in (False, True):
                    env.reset(seed=123)
                    env.penalize_minor_corrections = flag
                    with patch.object(env, '_get_profiles', return_value=(p*hi, q*hi, pv, price, date)), \
                         patch.object(env, '_calculate_rewards', wraps=env._calculate_rewards) as reward:
                        _, _, _, _, info = env.step(np.zeros(env.num_agents))
                    # Small correction is still reported, but every directional
                    # backup activation is penalized regardless of move size.
                    self.assertTrue(info['minor_corrected_action'])
                    self.assertTrue(info['backup_activated'])
                    self.assertTrue(info['major_infeasible_action'])
                    self.assertTrue(reward.call_args.kwargs['infeasible_action'])
                    self.assertEqual(info['backup_residual_penalty'], 0)
            finally:
                env.penalize_minor_corrections = original_flag


if __name__ == '__main__':
    unittest.main()
