"""Real AC integration plus actual evaluator-function CSV tests, without torch.

AST loading excludes model/optimizer imports only; tested evaluator bodies are
compiled unchanged. Synthetic in-memory ratings are explicitly labeled.
"""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environments.dess_env import DESSEnv
from environments.primary_marl import primary_config, validate_primary_env, validate_network
from environments.directional_backup import search, evaluation_fields
from utility.run_manifest import RunManifest
from utility.evaluation_reporting import (observation_ready, failure_row, step_fields,
                                        finish_episode, augment_summaries, diagnostic_summary,
                                        reset_for_evaluation, pf_exception_info)
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from test_directional_backup import fixture_config
from utility.feasibility import current_overloaded
from ac_intervention_fixture import exercise_interventions


EVALUATORS = ['scripts/run_eval_ieee33.py', 'scripts/run_eval_ieee69.py',
 'experiments/no_gnn_maddpg/run_eval_mlp_ieee33.py', 'experiments/no_gnn_maddpg/run_eval_mlp_ieee69.py',
 'experiments/topology_generalization/run_tp_eval.py', 'experiments/topology_generalization/run_tp_eval_ieee69.py',
 'experiments/no_gnn_maddpg/run_tp_eval_mlp.py', 'experiments/no_gnn_maddpg/run_tp_eval_mlp_ieee69.py',
 'experiments/mpc_smpc/optimization/rolling_horizon_eval.py',
 'experiments/mpc_smpc/optimization/rolling_horizon_eval_ieee69.py']


def evaluator(path):
    source = ast.parse((ROOT/path).read_text())
    names = {'reset_at_episode_start', 'get_action', 'evaluate_one_episode', 'summarize',
             'summarize_results', 'build_step_row', 'get_soc_vector', 'normalize_topology_case',
             'build_timestamp_index_map', 'resolve_time_index', 'choose_available_horizon'}
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(globals(), MLP_POLICY_NAME='mlp', cfg=SimpleNamespace(), Optional=__import__('typing').Optional,
                 Dict=dict, Any=__import__('typing').Any, ScenarioData=object, PointForecastData=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/path), 'exec'), scope)
    return scope


def synthetic_config(original):
    return primary_config(fixture_config(original))


class Reporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DESSEnv(synthetic_config(IEEE33_CONFIG), seed=123)

    def test_incorrect_instantiated_modes_rejected(self):
        env = self.env
        validate_primary_env(env, 'synthetic')
        for name, value in [('doe_enabled', False), ('backup_mode', 'legacy_scaling'),
                            ('doe_pf_failure_mode', 'strict')]:
            with patch.object(env, name, value):
                with self.assertRaises(ValueError): validate_primary_env(env, 'synthetic')
        with patch.dict(env.backup_cfg, pf_budget=19):
            with self.assertRaises(ValueError): validate_primary_env(env, 'synthetic')

    def test_manifests_actual_config_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            run = RunManifest(out, 'GCN', 123, 'train')
            run.add_environment(self.env, scenario='synthetic', intended_episodes=2)
            data = json.loads((out/'run_manifest.json').read_text())
            actual = data['environments'][0]
            self.assertEqual(actual['doe']['enabled'], True)
            self.assertEqual(actual['backup']['pf_budget'], self.env.backup_cfg['pf_budget'])
            self.assertEqual(actual['reward'], self.env.reward_cfg)
            self.assertAlmostEqual(actual['min_current_rating_a'], 255)
            self.assertEqual(actual['scenario'], 'synthetic')
            with self.assertRaises(FileExistsError): RunManifest(out, 'GCN', 123, 'train')

    def test_optimization_battery_compatibility(self):
        from utility.run_manifest import validate_optimization_compatibility
        from experiments.mpc_smpc.optimization import baseline_config
        validate_optimization_compatibility(self.env, baseline_config)
        with patch.object(baseline_config, 'ETA_CH', .8):
            with self.assertRaises(ValueError): validate_optimization_compatibility(self.env, baseline_config)

    def test_all_evaluators_failed_reset_no_controller_call(self):
        for path in EVALUATORS:
            with self.subTest(path=path):
                scope = evaluator(path)
                env = Mock(episode_length=96, num_agents=9)
                scope['reset_at_episode_start'] = Mock(return_value=({'x': np.array([np.nan])},
                    dict(episode_failed=True, observation_valid=False, pf_failure_stage='reset', pf_failed_solve_count=1)))
                agent = Mock()
                args = dict(env=env, episode_id=0, start_index=0, seed=123)
                if 'rolling_horizon' in path:
                    args.update(controller='mpc', scenario_data=None, point_data=None, horizon_t=4, n_scenarios=1)
                else:
                    args.update(agent=agent, policy='maddpg')
                    if 'run_tp_' in path: args['topology_case'] = 'TP1'
                rows = scope['evaluate_one_episode'](**args)
                agent.select_action.assert_not_called()
                env.step.assert_not_called()
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]['episode_failed'])
                self.assertEqual(rows[0]['episode_successful_steps'], 0)
                ep, agg = scope.get('summarize', scope.get('summarize_results'))(pd.DataFrame(rows))
                self.assertEqual(agg['failed_episodes'].iloc[0], 1)
                self.assertEqual(agg['completed_episodes'].iloc[0], 0)

    def test_strict_pf_reset_exception_recorded_and_indices_restored(self):
        env = Mock(episode_start_indices=[1, 2], pf_failed_solve_count=1)
        env.reset.side_effect = RuntimeError('injected PF convergence failure')
        obs, info = reset_for_evaluation(env, 1, 123)
        self.assertFalse(observation_ready(obs, info))
        self.assertTrue(info['episode_failed'])
        self.assertEqual(env.episode_start_indices, [1, 2])
        env.reset.side_effect = ValueError('unrelated configuration bug')
        with self.assertRaises(ValueError): reset_for_evaluation(env, 1, 123)

    def test_attribution_denominator_and_missing_values(self):
        df = pd.DataFrame(dict(major_backup_penalty_applied=[True, True, True, False],
            backup_zero_valid=[True, True, False, True], backup_zero_infeasible=[True, False, None, True]))
        stats = diagnostic_summary(df)
        self.assertEqual(stats['preexisting_infeasibility_percent_of_valid_penalized_steps'], 50)
        self.assertEqual(stats['penalized_steps_without_valid_zero_reference'], 1)
        self.assertTrue(np.isnan(stats['doe_intervention_rate']))

    def test_checkpoint_selection_rejects_incomplete_and_unknown(self):
        from utility.evaluation_reporting import eligible_checkpoint_rows
        row = dict(intended_episodes=2, completed_episodes=2, failed_episodes=0,
                   incomplete_episodes=0, final_network_feasibility_rate=1., pf_failed_solves=0,
                   worst_voltage_violation=5e-7)
        bad = dict(row, completed_episodes=1, incomplete_episodes=1)
        selected = eligible_checkpoint_rows(pd.DataFrame([row, bad]))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.index[0], 0)
        self.assertTrue(eligible_checkpoint_rows(pd.DataFrame([{'mean_feasible_rate': 1}])).empty)

    def test_periodic_validation_reset_failure_accounting(self):
        from utility.periodic_validation import evaluate_trainer
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'validation'/'step_0'
            output.mkdir(parents=True)
            env = Mock(episode_length=96, episode_start_index=0, num_buses=33)
            env.reset.return_value = ({'x': [np.nan]}, dict(episode_failed=True, observation_valid=False, pf_failed_solve_count=1))
            trainer = SimpleNamespace(eval_env=env, config={}, checkpoint_dir=tmp,
                global_step=0, eval_episodes=2, max_eval_steps=96, agent=Mock())
            # F16: the Mock env has no time series, so supply its fixed periodic schedule.
            schedule = dict(schedule_id='S-PER-v1', sha256='mock', episodes=[
                dict(episode_id='val_a', start_index=0, seed=1), dict(episode_id='val_b', start_index=96, seed=2)])
            with patch('utility.periodic_validation.RunManifest'), \
                    patch('utility.periodic_validation.periodic_schedule', return_value=schedule):
                result = evaluate_trainer(trainer)
            trainer.agent.select_action.assert_not_called()
            self.assertEqual(result['failed_episodes'], 2)
            self.assertEqual(result['completed_episodes'], 0)
            self.assertTrue((output/'episode_summary.csv').is_file())

    def test_mpc_csv_dispatch_interface_without_optimizer(self):
        for path in EVALUATORS[8:]:
            env = self.env if 'ieee69' not in path else DESSEnv(synthetic_config(IEEE69_CONFIG), seed=123)
            scope = evaluator(path)
            _, reset = env.reset(seed=123)
            points = SimpleNamespace(timestamps=pd.DatetimeIndex([reset['date_time']]))
            scope['solve_mpc_action'] = lambda **kw: dict(action=np.zeros(env.num_agents), status='test_controller', solve_time_sec=.01)
            original = env.step
            def once(action):
                obs, reward, term, trunc, info = original(action)
                return obs, reward, False, True, info
            with patch.object(env, 'step', side_effect=once):
                rows = scope['evaluate_one_episode'](controller='mpc', env=env, scenario_data=None,
                    point_data=points, episode_id=0, start_index=env.episode_start_index,
                    seed=123, horizon_t=1, n_scenarios=1)
            ep, agg = scope['summarize_results'](pd.DataFrame(rows))
            self.assertIn('backup_pf_evaluations', rows[0])
            self.assertIsNone(rows[0]['smpc_network_feasible'])
            self.assertEqual(ep['attempted_evaluation_steps'].iloc[0], 1)
            self.assertEqual(agg['incomplete_episodes'].iloc[0], 1)

    def test_real_doe_backup_major_penalty_and_csv(self):
        for config in (IEEE33_CONFIG, IEEE69_CONFIG):
            env = DESSEnv(synthetic_config(config), seed=321)
            fields = exercise_interventions(env)
            with tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp)/'interventions.csv'
                pd.DataFrame([fields]).to_csv(file, index=False)
                row = pd.read_csv(file).iloc[0]
                self.assertTrue(row['doe_intervened'])
                self.assertTrue(row['major_backup_penalty_applied'])

    def test_unseen_topology_actual_effective_ratings(self):
        from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_topology_config('TP2', base_config=synthetic_config(IEEE69_CONFIG), output_root=tmp)
            env = DESSEnv(cfg, seed=123)
            validate_primary_env(env, 'synthetic')
            self.assertEqual(len(env.grid.in_service_line_df), 68)
            amps = env.grid.get_line_current_limits_pu()*1000/(np.sqrt(3)*12.66)
            np.testing.assert_allclose(amps, 255)

    def test_distinct_action_stage_labels(self):
        info = dict(marl_raw_action=np.array([-.9]), marl_prepared_action=np.array([-.9]),
            physical_limited_power_kw=np.array([-60.]), doe_conditioned_action=np.array([-.3]),
            doe_power_kw=np.array([-22.5]), requested_action=np.array([-.3]),
            accepted_action=np.array([.2]), final_battery_power_kw=np.array([15.]))
        fields = evaluation_fields(info)
        self.assertEqual(json.loads(fields['policy_proposal_action_normalized']), [-.9])
        self.assertEqual(json.loads(fields['doe_action_normalized']), [-.3])
        self.assertEqual(json.loads(fields['backup_selected_action_normalized']), [.2])
        self.assertEqual(json.loads(fields['executed_battery_power_kw']), [15.])

    def test_actual_gnn_mlp_csv_writers_and_failure_after_dispatch(self):
        env = self.env
        for path in EVALUATORS[:8]:
            scope = evaluator(path)
            agent = SimpleNamespace(select_action=lambda obs, noise_std=0: -np.ones(env.num_agents))
            args = dict(env=env, agent=agent, policy='maddpg', episode_id=0,
                        start_index=env.episode_start_indices[0], seed=123)
            if 'run_tp_' in path: args['topology_case'] = 'TP1'
            # Force next-observation PF failure only after a real committed step.
            original_step = env.step
            def one_step(action):
                result = original_step(action)
                obs, reward, term, trunc, info = result
                info = dict(info, episode_failed=True, observation_valid=False, pf_failure_stage='next_observation')
                return obs, reward, False, True, info
            with patch.object(env, 'step', side_effect=one_step):
                rows = scope['evaluate_one_episode'](**args)
            df = pd.DataFrame(rows)
            ep, agg = scope['summarize'](df)
            with tempfile.TemporaryDirectory() as tmp:
                for name, frame in [('steps', df), ('episodes', ep), ('aggregate', agg)]:
                    frame.to_csv(Path(tmp)/(name+'.csv'), index=False)
                loaded = pd.read_csv(Path(tmp)/'steps.csv')
                self.assertIn('policy_proposal_action_normalized', loaded)
                self.assertTrue(loaded['doe_enabled'].iloc[0])
                self.assertEqual(loaded['backup_mode'].iloc[0], 'directional_incremental')
                self.assertEqual(ep['successful_evaluation_steps'].iloc[0], 1)
                self.assertEqual(ep['completed_episodes'].iloc[0], 0)
                self.assertEqual(agg['failed_episodes'].iloc[0], 1)

    def test_complete_short_episode_keeps_valid_costs(self):
        env = self.env
        scope = evaluator(EVALUATORS[0])
        agent = SimpleNamespace(select_action=lambda obs, noise_std=0: np.zeros(env.num_agents))
        # Explicit two-step smoke horizon; not represented as a 96-step day.
        with patch.object(env, 'episode_length', 2):
            rows = scope['evaluate_one_episode'](env, agent, 'maddpg', 0, env.episode_start_indices[0], 123)
        ep, agg = scope['summarize'](pd.DataFrame(rows))
        self.assertEqual(len(rows), 2)
        self.assertEqual(ep['completed_episodes'].iloc[0], 1)
        self.assertEqual(ep['successful_evaluation_steps'].iloc[0], 2)
        self.assertEqual(agg['failed_episodes'].iloc[0], 0)
        self.assertAlmostEqual(ep['total_energy_cost'].iloc[0], sum(r['energy_cost'] for r in rows))


class NineBatteryAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DESSEnv(synthetic_config(IEEE69_CONFIG), seed=123)

    def setUp(self):
        self.env.reset(seed=123)

    def scenario(self, action, target):
        env = self.env
        p, q = env.grid.get_base_load_kw().astype(float), env.grid.get_base_load_kvar().astype(float)
        lo, hi = 0., 2.
        for _ in range(18):
            mid = (lo+hi)/2
            first = env._evaluate_candidate_action(action, p*mid, q*mid, np.zeros(env.num_buses))
            if first['pf_result']['voltage_pu'].min() < target: hi = mid
            else: lo = mid
        return p*hi, q*hi, np.zeros(env.num_buses)

    def run_search(self, profiles, action):
        env = self.env
        first = env._evaluate_candidate_action(action, *profiles)
        start = first['dess_power_kw'][env.dess_buses]
        before = env._snapshot_battery_states()
        best, diag = search(first, start, [b.get_available_charge_kw() for b in env.batteries],
            [b.get_available_discharge_kw() for b in env.batteries],
            lambda power: env._evaluate_candidate_action(env._power_to_action(power), *profiles),
            env._measure_violation, env.backup_cfg)
        self.assertEqual(before, env._snapshot_battery_states())
        self.assertLessEqual(diag['backup_pf_evaluations'], 20)
        return first, best, diag

    def test_nine_heavy_charging_correctable(self):
        env = self.env
        self.assertEqual(env.num_agents, 9)
        a = -np.ones(9)
        profiles = self.scenario(a, .949)
        first, best, diag = self.run_search(profiles, a)
        self.assertLess(first['pf_result']['voltage_pu'].min(), .95-1e-6)
        self.assertTrue(np.all(first['dess_power_kw'][env.dess_buses] < 0))
        self.assertTrue(best['feasible'])
        self.assertEqual(diag['backup_final_score'], 0)

    def test_empty_soc_no_invented_discharge(self):
        env = self.env
        for battery in env.batteries: battery.soc = battery.config.soc_min
        profiles = self.scenario(np.zeros(9), .94)
        first, best, diag = self.run_search(profiles, -np.ones(9))
        self.assertFalse(best['feasible'])
        self.assertTrue(diag['backup_best_effort'])
        self.assertTrue(np.all(best['dess_power_kw'][env.dess_buses] <= 1e-9))

    def test_feasible_unchanged(self):
        env = self.env
        profiles = (env.grid.get_base_load_kw()*.2, env.grid.get_base_load_kvar()*.2, np.zeros(env.num_buses))
        first, best, diag = self.run_search(profiles, np.zeros(9))
        self.assertTrue(first['feasible'])
        np.testing.assert_array_equal(first['accepted_action'], best['accepted_action'])
        self.assertEqual(diag['backup_pf_evaluations'], 1)

    def test_reverse_flow_discharge_worsens_current(self):
        env = self.env
        p, q = env.grid.get_base_load_kw()*.25, env.grid.get_base_load_kvar()*.25
        pv = np.zeros(env.num_buses)
        pv[env.dess_buses] = (p.sum()+400)/env.num_agents
        with patch.object(env, 'grid_export_limit_kw', 100000.):
            zero = env._evaluate_candidate_action(np.zeros(9), p, q, pv)
            discharge = env._evaluate_candidate_action(np.full(9, .2), p, q, pv)
            self.assertGreater(discharge['pf_result']['line_current_pu'][0], zero['pf_result']['line_current_pu'][0])
            limits = env.grid.line_current_limits_pu.copy()
            limits[0] = zero['pf_result']['line_current_pu'][0]*.98
            with patch.object(env.grid, 'line_current_limits_pu', limits):
                first, best, diag = self.run_search((p,q,pv), np.zeros(9))
                self.assertGreater(diag['backup_initial_score'], 0)
                self.assertTrue(best['feasible'])
                self.assertEqual(diag['backup_direction'], 'charge')


if __name__ == '__main__': unittest.main()
