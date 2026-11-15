"""Read-only attribution and exact reward/dispatch regression against pre-edit source."""
import copy
import io
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environments.dess_env import DESSEnv
from environments.directional_backup import backup_attribution, evaluation_fields
from utility.feasibility import BACKUP_SCORE_TOLERANCE as TOL
from utility.evaluation_reporting import diagnostic_summary
from test_directional_backup import fixture_config, IEEE33_CONFIG, IEEE69_CONFIG, run_rule, pf
from test_experiment_integration import evaluator, EVALUATORS
from ac_intervention_fixture import exercise_interventions
from f3_obs import assert_agent_obs_equal_except_f3_curtailment


class Attribution(unittest.TestCase):
    def test_reward_reads_explicit_flag_and_preserves_nonconvergence_early_return(self):
        env = SimpleNamespace(reward_cfg=dict(w_grid=0, w_local=0, w_kpi=0,
            use_kpi_reward=False, infeasible_action_penalty=-77, nonconvergence_penalty=-123),
            last_pf_result={'converged': True}, current_kpis=dict(grid_import_kw=0,
            voltage_deviation=0, curtailment_kw=0), delta_t_hours=.25,
            num_agents=1, dess_buses=[0])
        kw = dict(price=0, dess_power_kw=np.zeros(1), soc_before=np.zeros(1),
                  soc_after=np.zeros(1))
        np.testing.assert_array_equal(DESSEnv._calculate_rewards(env, **kw,
            infeasible_action=True, backup_penalty_applied=False), [0])
        np.testing.assert_array_equal(DESSEnv._calculate_rewards(env, **kw,
            infeasible_action=False, backup_penalty_applied=True), [-77])
        env.last_pf_result['converged'] = False
        np.testing.assert_array_equal(DESSEnv._calculate_rewards(env, **kw,
            infeasible_action=True, backup_penalty_applied=True), [-123])

    def test_only_zero_reference_fails(self):
        best, diag, _ = run_rule(
            lambda p: pf([np.nan], converged=False) if p == 0 else pf([.94+p*.0004]),
            start=10)
        self.assertTrue(best['feasible'])
        self.assertTrue(diag['backup_zero_evaluated'])
        fields = evaluation_fields(dict(diag, backup_activated=True))
        self.assertEqual(fields['backup_cause'], 'unknown')
        self.assertIsNone(fields['zero_battery_feasible'])

    def test_cause_initial_not_final_and_boundaries(self):
        def cause(zero, initial, **kw):
            return backup_attribution(dict(backup_activated=True, backup_zero_evaluated=True,
                backup_zero_score=zero, backup_initial_score=initial, backup_final_score=0, **kw))
        self.assertEqual(cause(0, 3)['backup_cause'], 'action_induced')
        self.assertEqual(cause(TOL, 3)['backup_cause'], 'action_induced')
        self.assertEqual(cause(2, 2 + TOL/2)['backup_cause'], 'preexisting_not_worsened')
        self.assertEqual(cause(2, 2 + 2*TOL)['backup_cause'], 'preexisting_worsened')
        self.assertFalse(cause(2, 3)['zero_battery_feasible'])
        self.assertEqual(cause(float('inf'), 3)['backup_cause'], 'unknown')
        self.assertEqual(cause(float('nan'), 3)['backup_cause'], 'unknown')
        self.assertEqual(backup_attribution(dict(backup_activated=True))['backup_cause'], 'unknown')
        self.assertEqual(backup_attribution(dict(backup_activated=False))['backup_cause'], 'none')

    def test_counts_rates_and_csv(self):
        rows = []
        for active, zero, initial, penalty in [(False, None, 0, False),
                (True, 0, 1, True), (True, 2, 1, True), (True, 2, 3, True),
                (True, float('inf'), float('inf'), False)]:
            rows.append(evaluation_fields(dict(backup_activated=active,
                backup_zero_evaluated=zero is not None, backup_zero_score=zero,
                backup_initial_score=initial, backup_penalty_applied=penalty)))
        frame = pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)))
        stats = diagnostic_summary(frame)
        self.assertEqual(stats['backup_activation_count'], 4)
        self.assertEqual(stats['backup_penalty_applied_count'], 3)
        for cause in ('none', 'action_induced', 'preexisting_not_worsened',
                      'preexisting_worsened', 'unknown'):
            self.assertEqual(stats['backup_cause_'+cause+'_count'], 1)
            self.assertEqual(stats['backup_cause_'+cause+'_rate'], .2)
        self.assertEqual(stats['backup_cause_measured_steps'], 5)


class RealAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('BACKUP_ATTRIBUTION_BASELINE_SOURCE')
        if path:
            source = Path(path).read_text(encoding='utf-8-sig')
        else:
            archive = ROOT.parent/'analysis/backup_attribution_step2/before.zip'
            with zipfile.ZipFile(archive) as z:
                source = z.read('environments/dess_env.py').decode('utf-8-sig')
        scope = dict(__name__='pre_attribution_env', __file__=str(ROOT/'environments/dess_env.py'))
        exec(compile(source, scope['__file__'], 'exec'), scope)
        cls.BeforeEnv = scope['DESSEnv']

    def test_doe_disabled_200_steps_matches_prechange_both_feeders(self):
        for cfg in (IEEE33_CONFIG, IEEE69_CONFIG):
            c = fixture_config(cfg)
            c['doe']['doe_enabled'] = False
            before, after = self.BeforeEnv(copy.deepcopy(c), seed=123), DESSEnv(copy.deepcopy(c), seed=123)
            before.reset(seed=123); after.reset(seed=123)
            rng = np.random.default_rng(123)
            for t in range(200):
                action = rng.uniform(-1, 1, after.num_agents).astype(np.float32)
                a, b = before.step(action.copy()), after.step(action.copy())
                np.testing.assert_array_equal(a[1], b[1])
                for key in a[0]:
                    if key == 'agent_obs':
                        # F5: grid_stress (last column) removed from actor observations.
                        self.assertIn(a[0][key].shape[1], (b[0][key].shape[1], b[0][key].shape[1] + 1))
                        # F3: only the feeder-curtailment scale changed (same physical value).
                        assert_agent_obs_equal_except_f3_curtailment(
                            b[0][key], a[0][key][:, :b[0][key].shape[1]], after, before)
                    else:
                        np.testing.assert_array_equal(a[0][key], b[0][key])
                self.assertEqual(a[2:4], b[2:4])
                for key in ('physical_clipped_kw', 'final_battery_power_kw', 'soc_after'):
                    np.testing.assert_array_equal(a[4][key], b[4][key])
                self.assertEqual(b[4]['backup_penalty_applied'], a[4]['major_backup_penalty_applied'])
                if b[2] or b[3]:
                    before.reset(seed=124+t); after.reset(seed=124+t)

    def test_failed_search_unknown_no_double_penalty_and_reward_equivalence(self):
        for cfg in (IEEE33_CONFIG, IEEE69_CONFIG):
            c = fixture_config(cfg)
            for enabled in (False, True):
                c['doe']['doe_enabled'] = enabled
                results = []
                for klass in (self.BeforeEnv, DESSEnv):
                    env = klass(copy.deepcopy(c), seed=123); env.reset(seed=123)
                    states = env._snapshot_battery_states()
                    with patch.object(env.power_flow, 'run_power_flow', side_effect=RuntimeError('PF injected failure')):
                        result = env.step(np.ones(env.num_agents))
                    self.assertEqual(states, env._snapshot_battery_states())
                    results.append(result)
                np.testing.assert_array_equal(results[0][1], results[1][1])
                _, reward, term, trunc, info = results[1]
                self.assertFalse(term); self.assertTrue(trunc)
                self.assertFalse(info['backup_penalty_applied'])
                self.assertEqual(info['backup_cause'], 'unknown')
                self.assertIsNone(info['zero_battery_feasible'])
                np.testing.assert_array_equal(reward, np.full(env.num_agents,
                    env.reward_cfg.get('nonconvergence_penalty', -10000), dtype=np.float32))

    def test_doe_and_backup_reward_equivalence_and_csv_all_marl(self):
        for cfg in (IEEE33_CONFIG, IEEE69_CONFIG):
            c = fixture_config(cfg); c['doe']['doe_enabled'] = True
            before, after = self.BeforeEnv(copy.deepcopy(c), seed=123), DESSEnv(copy.deepcopy(c), seed=123)
            # Compare the pre-change reward method on precisely the same committed state.
            original = after._calculate_rewards
            def compare(**kw):
                actual = original(**kw)
                old_kw = dict(kw); old_kw.pop('backup_penalty_applied', None)
                expected = self.BeforeEnv._calculate_rewards(after, **old_kw)
                np.testing.assert_array_equal(actual, expected)
                return actual
            with patch.object(after, '_calculate_rewards', side_effect=compare):
                info = exercise_interventions(after)
            self.assertTrue(info['backup_penalty_applied'])
            self.assertEqual(info['backup_cause'], 'action_induced')
        for path in EVALUATORS[:8]:
            scope = evaluator(path)
            with patch.object(after, 'episode_length', 1):
                kwargs = dict(env=after, agent=None, policy='zero', episode_id=0,
                    start_index=after.episode_start_indices[0], seed=123)
                if 'run_tp_' in path: kwargs['topology_case'] = 'TP1'
                rows = scope['evaluate_one_episode'](**kwargs)
            frame = pd.DataFrame(rows)
            ep, agg = scope['summarize'](frame)
            for output in (frame, ep, agg):
                for column in ('backup_activation_count', 'backup_penalty_applied_count'):
                    self.assertIn(column, output)
            for output in (ep, agg):
                self.assertIn('backup_cause_none_rate', output)
            self.assertIn('infeasible_requested_count', ep)

    def test_real_preexisting_violation_and_reference_failure(self):
        env = DESSEnv(fixture_config(IEEE69_CONFIG), seed=123)
        env.doe_enabled = False; env.reset(seed=123)
        for b in env.batteries: b.soc = b.config.soc_min
        p, q = env.grid.get_base_load_kw()*1.2, env.grid.get_base_load_kvar()*1.2
        _, _, pv, price, date = env._get_profiles(env.current_index)
        with patch.object(env, '_get_profiles', return_value=(p, q, np.zeros_like(pv), price, date)):
            _, _, _, _, info = env.step(np.zeros(env.num_agents))
        self.assertTrue(info['backup_activated'])
        self.assertFalse(info['zero_battery_feasible'])
        self.assertGreater(info['zero_battery_score'], TOL)
        self.assertEqual(info['backup_cause'], 'preexisting_not_worsened')


if __name__ == '__main__':
    unittest.main()
