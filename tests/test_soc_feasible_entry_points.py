"""SOC-feasible mapping through the REAL maintained entry points (training.train.main with its DEFAULT protocol,
evaluation.evaluate select/test/zero, evaluation.summarize, evaluation.reporting.evaluate, evaluation.oracle) for every
controller (MLP A0-A3, GCN, GAT, TAGConv) on BOTH feeders (IEEE33 and IEEE69).

Kept light: 205 training steps (200 warm-up, 5 updates) per controller, and every evaluation stage runs ONE day
(run_episodes truncated; the 61-day selection rule is stubbed because one day cannot satisfy it, but the arguments the
maintained pipeline passes to scripts/select_checkpoint are checked). The final-test and zero stages are exercised on
VALIDATION days (episode_plan and the selection record are substituted), so no December test data is used.
A spy on environments.marl_env.soc_feasible_action proves the mapping is executed at every environment step.
Run: python -m unittest tests.test_soc_feasible_entry_points -v      (~30-50 min on a laptop CPU)
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from environments.battery_feasibility import soc_feasible_action as REAL_MAPPING
from tests.support import CONTROLLER_SPECS, series

STEPS = 205
TOPOLOGIES = dict(ieee33=['TP1', 'TP2'], ieee69=['TP1', 'TP10'])
SCALES = [1.0, 1.3]


def mapping_spy():
    return patch('environments.marl_env.soc_feasible_action', side_effect=REAL_MAPPING)


def lists(series_):
    return np.array([json.loads(v) for v in series_], dtype=float)


def key_of(family, variant):
    from configs.experiments import controller_key
    return controller_key(family, variant)


@unittest.skipUnless(series('ieee33') and series('ieee69'), 'no processed time series')
class EntryPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from configs.experiments import RunLayout, feeder_config
        from configs.ieee33_config import IEEE33_CONFIG
        from configs.ieee69_config import IEEE69_CONFIG
        from environments.marl_env import make_marl_env
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / 'soc_feasible'
        cls.layout = RunLayout(cls.root)
        cls.patchers = [patch.dict(IEEE33_CONFIG['paths'], processed_time_series=Path(series('ieee33'))),
                        patch.dict(IEEE69_CONFIG['paths'], processed_time_series=Path(series('ieee69')))]
        for p in cls.patchers:
            p.start()
        cls.val_starts = {f: list(make_marl_env(feeder_config(f, experiment='soc_feasible'), 'val', 42, 0.0)
                                  .episode_start_indices[:31]) for f in ('ieee33', 'ieee69')}

    @classmethod
    def tearDownClass(cls):
        for p in cls.patchers:
            p.stop()
        cls.tmp.cleanup()

    # ------------------------------------------------------------------ helpers
    def spec(self, feeder, key):
        from evaluation.evaluate import EvaluationSpec
        return EvaluationSpec(feeder, key, 0, 'soc_feasible', self.layout, cpu=True)

    def train(self, feeder, family, variant):
        from training.train import main
        argv = ['--feeder', feeder, '--controller', family, '--seed', '0', '--output-root', str(self.root),
                '--time-series', series(feeder), '--total-steps', str(STEPS), '--warmup-steps', '200', '--buffer-size',
                '400', '--batch-size', '16', '--eval-every', '1000000000', '--save-every', str(STEPS),
                '--diagnostics-every', '1', '--cpu']            # no --protocol: the maintained default must be soc_feasible
        if variant:
            argv += ['--variant', variant]
        with mapping_spy() as spy:
            out = main(argv)
        key = key_of(family, variant)
        self.assertEqual(out, self.layout.run_dir(feeder, key, 0) / 'training')
        self.assertEqual(out.relative_to(self.root).parts[:3 if variant else 2],
                         (feeder, family, variant) if variant else (feeder, family))
        self.assertGreaterEqual(spy.call_count, STEPS)                     # mapping executed at every training step
        manifest = json.loads((out / 'run_manifest.json').read_text())
        self.assertNotIn('b3', manifest)
        record = manifest['run_record']
        self.assertEqual((record['experiment'], record['protocol'], record['action_mapping'],
                          record['feasibility_supervision'], record['lambda_feasibility'], record['feasibility_formulation'],
                          record['controller'], record['checkpoint_schema']),
                         ('soc_feasible', 'soc_feasible', 'soc_feasible', False, 0.0, 'none', key,
                          'graph-marl-dess-checkpoint/1'))
        self.assertIn('source_inventory', record)
        for env in manifest['environments']:
            self.assertEqual(env['reward'], manifest['environments'][0]['reward'])
        import torch
        checkpoint = out / 'checkpoints' / f'maddpg_step_{STEPS}.pt'
        ck = torch.load(checkpoint, weights_only=False)
        self.assertEqual((ck['schema'], ck['controller'], ck['action_mapping'], ck['feasibility_supervision'],
                          ck['lambda_feasibility'], ck['feasibility_formulation']),
                         ('graph-marl-dess-checkpoint/1', key, 'soc_feasible', False, 0.0, 'none'))
        updates = pd.read_csv(out / 'training_updates.csv')
        self.assertTrue((updates.feasibility_formulation == 'none').all())
        return checkpoint

    def one_day(self, spy_counts):
        import evaluation.evaluate as evaluate
        real = evaluate.run_episodes

        def run(feeder, env, agent, policy, tp, plan):
            if agent is not None:
                self.assertEqual(env.action_mapping, 'soc_feasible')
            with mapping_spy() as spy:
                frame = real(feeder, env, agent, policy, tp, list(plan)[:1])
            spy_counts.append(spy.call_count)
            return frame
        return run

    def check_step_metrics(self, folder, topologies):
        steps = pd.read_csv(folder / 'step_metrics.csv')
        self.assertEqual(set(steps.marl_action_mapping), {'soc_feasible'})
        self.assertEqual(set(steps.topology_case), set(topologies))
        self.assertLessEqual(float(lists(steps.marl_soc_limiter_correction_kw).max()), 1e-6)
        np.testing.assert_allclose(lists(steps.policy_requested_power_kw), lists(steps.marl_power_soc_mapped_kw))
        episodes = pd.read_csv(folder / 'episode_summary.csv')
        self.assertEqual(set(episodes.action_mapping), {'soc_feasible'})
        for column in ('b3_soc_near_min_fraction', 'b3_soc_near_max_fraction', 'b3_max_soc_boundary_streak',
                       'b3_charge_throughput_kwh', 'b3_discharge_throughput_kwh', 'adjusted_energy_cost'):
            self.assertIn(column, episodes.columns)

    def select(self, feeder, key):
        import evaluation.evaluate as evaluate
        counts = []
        with patch.object(evaluate, 'run_episodes', self.one_day(counts)), \
                patch('scripts.select_checkpoint.main', return_value='stubbed: one day cannot meet the 61-day rule') as rule:
            evaluate.select(self.spec(feeder, key))
        self.assertTrue(counts and min(counts) >= 96)                      # one full day, mapping at every step
        argv = rule.call_args.args[0]
        sweep = self.layout.run_dir(feeder, key, 0) / 'selection_sweep'
        self.assertEqual(argv, ['--feeder', feeder, '--model', key, '--run', 'seed_0', '--sweep_dir', str(sweep),
                                '--output_root', str(self.layout.selections_dir(feeder))])
        manifest = json.loads((sweep / f'step_{STEPS}' / 'run_manifest.json').read_text())
        self.assertEqual((manifest['run_record']['action_mapping'], manifest['run_record']['feasibility_supervision'],
                          manifest['run_record']['controller']), ('soc_feasible', False, key))
        self.check_step_metrics(sweep / f'step_{STEPS}', ['TP1'])

    def evaluation_patches(self, feeder, checkpoint=None):
        import evaluation.evaluate as evaluate
        from utility.evaluation_protocol import authorize as real_authorize

        def authorize(**kw):                                              # learned final test without a 61-day selection
            protocol = real_authorize(mode='test', final=True, run_purpose='final_test', learned=False,
                                      topologies=kw['topologies'])
            if checkpoint is not None:
                protocol['selection'] = dict(checkpoint_path=str(checkpoint), selection_sha256='integration-test')
            return protocol

        def validation_days(env, episodes, seed, protocol):               # never December test data here
            return self.val_starts[feeder], [seed + i for i in range(31)], None
        return [patch.object(evaluate, 'existing_selections', return_value=[Path('integration-test-selection')]),
                patch.object(evaluate, 'authorize', side_effect=authorize),
                patch.object(evaluate, 'episode_plan', side_effect=validation_days),
                patch.dict(evaluate.TOPOLOGIES, {feeder: TOPOLOGIES[feeder]})]

    def final_test(self, feeder, key, checkpoint):
        import evaluation.evaluate as evaluate
        from experiments.load_stress.stress_env import scale_label
        counts = []
        patches = self.evaluation_patches(feeder, checkpoint) + [patch.object(evaluate, 'run_episodes', self.one_day(counts))]
        for p in patches:
            p.start()
        try:
            evaluate.test(self.spec(feeder, key), SCALES)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(len(counts), len(TOPOLOGIES[feeder]) * len(SCALES))
        self.assertTrue(min(counts) >= 96)
        for scale in SCALES:
            folder = self.layout.run_dir(feeder, key, 0) / 'test' / scale_label(scale)
            manifest = json.loads((folder / 'run_manifest.json').read_text())
            self.assertEqual((manifest['run_record']['action_mapping'], manifest['run_record']['controller']),
                             ('soc_feasible', key))
            self.check_step_metrics(folder, TOPOLOGIES[feeder])

    def zero_reference(self, feeder):
        import evaluation.evaluate as evaluate
        counts = []
        patches = self.evaluation_patches(feeder) + [patch.object(evaluate, 'run_episodes', self.one_day(counts))]
        for p in patches:
            p.start()
        try:
            outs = evaluate.zero(feeder, 'soc_feasible', self.layout, [1.0])
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(outs, [self.layout.zero_dir(feeder) / 'load_x100'])
        manifest = json.loads((outs[0] / 'run_manifest.json').read_text())
        self.assertEqual(set(manifest['run_record']['zero_dispatch_checks']), set(TOPOLOGIES[feeder]))

    def feeder_pipeline(self, feeder):
        for family, variant in CONTROLLER_SPECS:
            key = key_of(family, variant)
            with self.subTest(feeder=feeder, controller=key):
                checkpoint = self.train(feeder, family, variant)
                self.select(feeder, key)
                self.final_test(feeder, key, checkpoint)
        self.zero_reference(feeder)
        # Final report: the action_mapping label is kept per controller group and never averaged.
        from evaluation.summarize import summarize
        summary_dir = summarize(feeder, self.layout)
        table = pd.read_csv(summary_dir / 'by_controller.csv')
        learned = table[table.controller != 'zero_action_no_backup']
        self.assertEqual(set(learned.action_mapping), {'soc_feasible'})
        self.assertEqual(set(learned.controller), {key_of(f, v) for f, v in CONTROLLER_SPECS})
        self.assertEqual(len(learned), len(CONTROLLER_SPECS) * len(SCALES) * len(TOPOLOGIES[feeder]))
        for column in ('adjusted_energy_cost_mean', 'b3_soc_near_max_fraction_mean', 'b3_charge_throughput_kwh_mean'):
            self.assertTrue(pd.api.types.is_numeric_dtype(table[column]), column)
        self.assertFalse(any(c.startswith('action_mapping_') for c in table.columns))   # not aggregated

    # ------------------------------------------------------------------ tests
    def test_ieee33_all_controllers_train_select_test_zero_summary(self):
        self.feeder_pipeline('ieee33')

    def test_ieee69_all_controllers_train_select_test_zero_summary(self):
        self.feeder_pipeline('ieee69')

    def test_summary_refuses_mixed_action_mappings(self):
        from configs.experiments import RunLayout
        from evaluation.summarize import summarize
        root = Path(self.tmp.name) / 'mixed'
        for key, mapping in (('mlp_a0', 'soc_feasible'), ('gcn', 'rated')):
            folder = RunLayout(root).run_dir('ieee33', key, 0) / 'test' / 'load_x100'
            folder.mkdir(parents=True)
            (folder / 'run_manifest.json').write_text(json.dumps(dict(status='finished')))
            pd.DataFrame(dict(controller=[key], load_scaling_factor=[1.0], topology_case=['TP1'], run_id=[0],
                              total_energy_cost=[1.0], action_mapping=[mapping])).to_csv(folder / 'episode_summary.csv',
                                                                                      index=False)
        with self.assertRaisesRegex(SystemExit, 'Mixed action_mapping'):
            summarize('ieee33', RunLayout(root))

    def test_strict_reporting_evaluator_and_rated_checkpoint_rejection(self):
        import evaluation.evaluate as evaluate
        import evaluation.reporting.evaluate as reporting
        from types import SimpleNamespace
        train_dir = self.layout.run_dir('ieee33', 'mlp_a0', 0) / 'training'
        if not (train_dir / 'run_manifest.json').is_file():
            self.train('ieee33', 'mlp', 'A0')
        manifest = train_dir / 'run_manifest.json'
        training = json.loads(manifest.read_text())
        checkpoint = train_dir / 'checkpoints' / f'maddpg_step_{STEPS}.pt'
        args = SimpleNamespace(experiment_class='soc_feasible', controller='mlp_a0', feeder='ieee33', run=0,
                               training_manifest=manifest, time_series=series('ieee33'), cpu=True)
        counts = []
        out = Path(self.tmp.name) / 'strict' / 'select_step'
        with patch.object(evaluate, 'run_episodes', self.one_day(counts)):
            reporting.evaluate_checkpoint(args, training, checkpoint, out, 'select')
        written = json.loads((out / 'run_manifest.json').read_text())
        self.assertEqual(written['scientific_identity']['experiment_class'], 'soc_feasible')
        self.assertTrue(counts and counts[0] >= 96)
        # A rated (old-convention) checkpoint is refused by the same evaluator in the soc_feasible environment.
        import torch
        from configs.experiments import feeder_config
        from environments.marl_env import make_marl_env
        rated = torch.load(checkpoint, weights_only=False)
        rated.update(action_mapping='rated', feasibility_supervision=True, feasibility_formulation='pre_tanh_squared_hinge')
        rated_path = Path(self.tmp.name) / 'rated.pt'
        torch.save(rated, rated_path)
        env = make_marl_env(feeder_config('ieee33', series('ieee33'), 'soc_feasible'), 'val', 42, 0.0)
        with self.assertRaisesRegex(ValueError, 'action_mapping'):
            evaluate.load_model('mlp_a0', rated_path, env, 'cpu')

    def test_oracle_dispatch_identical_under_both_mappings(self):
        """Same optimisation; only the kW -> action conversion follows the env mapping; executed dispatch must match."""
        from configs.experiments import feeder_config
        from environments.marl_env import make_marl_env
        from evaluation.evaluate import POLICY, run_episodes
        from evaluation.oracle.network import LinDistFlowNetwork
        from evaluation.oracle.run import OracleController
        from evaluation.oracle.solver import OracleSettings
        frames, records = {}, {}
        for experiment in ('b4', 'soc_feasible'):
            env = make_marl_env(feeder_config('ieee33', series('ieee33'), experiment), 'val', 42, 0.0)
            start = int(env.episode_start_indices[3])
            controller = OracleController(env, LinDistFlowNetwork(env), OracleSettings(), 'TP1', 1.0)
            frames[experiment] = run_episodes('ieee33', env, controller, POLICY, 'TP1', [(start, 42)])
            controller.finish()
            records[experiment] = controller.records[0]
        b4, soc = frames['b4'], frames['soc_feasible']
        np.testing.assert_allclose(lists(soc.executed_battery_power_kw), lists(b4.executed_battery_power_kw), atol=1e-3)
        self.assertAlmostEqual(float(soc.energy_cost.sum()), float(b4.energy_cost.sum()), places=3)
        self.assertEqual(records['soc_feasible']['predicted_objective'], records['b4']['predicted_objective'])
        self.assertLessEqual(float(lists(soc.marl_soc_limiter_correction_kw).max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
