"""Historical B3/B4 commands keep their exact names, layout, defaults and schemas after the Phase 2 consolidation.

    python -m unittest tests.test_historical_compatibility -v

Shims resolve to the maintained implementation; historical entry points write the historical layout
(<root>/<feeder>/b3_<controller>/run_<r>), labels (b3_*), manifest block ('b3' with b3_sources), checkpoint schema
('b3-checkpoint/2') and selection keys (run_<r>); the maintained loader accepts those checkpoints and vice versa.
One 205-step training run (IEEE33, mlp_a0) and one-day selection on validation days; no December test data.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.support import series

STEPS = 205


class ShimIdentity(unittest.TestCase):
    def test_shims_resolve_to_maintained_objects(self):
        import environments.feasibility_env as old_env, environments.marl_env as env
        import models.feasibility_agents as old_models, models.controllers as models
        import training.feasibility_trainer as old_trainer, training.controller_trainer as trainer
        import utility.experiment_setup as old_setup, configs.experiments as setup
        import experiments.b3.zero as old_zero, evaluation.zero as zero
        import experiments.b3_reporting.evaluate as old_rep, evaluation.reporting.evaluate as rep
        import experiments.b3_reporting.metrics as old_met, evaluation.reporting.metrics as met
        import experiments.b4.oracle.solver as old_solver, evaluation.oracle.solver as solver
        self.assertIs(old_env.B3DESSEnv, env.MarlDESSEnv)
        self.assertIs(old_env.make_b3_env, env.make_marl_env)
        self.assertIs(old_models.FeasibilityAgentMixin, models.ControllerMixin)
        self.assertEqual(old_models.CHECKPOINT_SCHEMA, 'b3-checkpoint/2')
        self.assertEqual(models.CHECKPOINT_SCHEMA, 'graph-marl-dess-checkpoint/1')
        self.assertIs(old_trainer.trainer_class, trainer.trainer_class)
        self.assertIs(old_setup.feeder_config, setup.feeder_config)
        for a, b in ((old_zero, zero), (old_rep, rep), (old_met, met), (old_solver, solver)):
            self.assertIs(a, b)

    def test_historical_defaults_unchanged(self):
        from experiments.b3.train import parse_args
        from models.feasibility_agents import CONTROLLERS, build_model
        from utility.experiment_setup import output_root_for
        args = parse_args(['--feeder', 'ieee33', '--controller', 'mlp_a0', '--run', '0'])
        self.assertEqual(args.experiment, 'b3')                               # historical default protocol
        self.assertEqual(Path(output_root_for('b3', None)), Path('results/b3'))
        self.assertEqual(Path(output_root_for('b4', None)), Path('results/b4'))
        self.assertEqual(CONTROLLERS['mlp_a3'], 'b3_mlp_a3')
        model = build_model('mlp_a0', 3.0, agent_obs_dim=10, num_agents=5, action_dim_per_agent=1, device='cpu',
                            share_actor=False)
        self.assertEqual((model.controller, model.action_mapping, model.feasibility_supervision),
                         ('b3_mlp_a0', 'rated', True))
        from experiments.b4.oracle import run as oracle
        with self.assertRaises(SystemExit), patch('sys.stdout'):
            oracle.main(['--help'])


@unittest.skipUnless(series('ieee33'), 'no processed time series')
class HistoricalPipeline(unittest.TestCase):
    def test_historical_train_select_layout_and_cross_loading(self):
        import torch
        from configs.ieee33_config import IEEE33_CONFIG
        import evaluation.evaluate as maintained
        from experiments.b3 import evaluate as historical
        from experiments.b3.train import main as train
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(IEEE33_CONFIG['paths'], processed_time_series=Path(series('ieee33'))):
            root = Path(tmp) / 'b3_root'
            out = train(['--feeder', 'ieee33', '--controller', 'mlp_a0', '--run', '0', '--experiment', 'soc_feasible',
                         '--output-root', str(root), '--time-series', series('ieee33'), '--total-steps', str(STEPS),
                         '--warmup-steps', '200', '--buffer-size', '400', '--batch-size', '16', '--eval-every',
                         '1000000000', '--save-every', str(STEPS), '--cpu'])
            self.assertEqual(out, root / 'ieee33' / 'b3_mlp_a0' / 'run_0' / 'training')
            manifest = json.loads((out / 'run_manifest.json').read_text())
            self.assertNotIn('run_record', manifest)
            self.assertEqual((manifest['b3']['controller'], manifest['b3']['checkpoint_schema']),
                             ('b3_mlp_a0', 'b3-checkpoint/2'))
            self.assertIn('b3_sources', manifest['b3'])
            checkpoint = out / 'checkpoints' / f'maddpg_step_{STEPS}.pt'
            ck = torch.load(checkpoint, weights_only=False)
            self.assertEqual((ck['schema'], ck['controller'], ck['action_mapping']),
                             ('b3-checkpoint/2', 'b3_mlp_a0', 'soc_feasible'))

            real = maintained.run_episodes
            one_day = lambda feeder, env, agent, policy, tp, plan: real(feeder, env, agent, policy, tp, list(plan)[:1])
            with patch.object(maintained, 'run_episodes', one_day), \
                    patch('scripts.select_checkpoint.main', return_value=None) as rule:
                historical.main(['select', '--feeder', 'ieee33', '--controller', 'mlp_a0', '--run', '0',
                                 '--experiment', 'soc_feasible', '--output-root', str(root), '--cpu'])
            argv = rule.call_args.args[0]
            self.assertEqual(argv[argv.index('--model') + 1], 'b3_mlp_a0')
            self.assertEqual(argv[argv.index('--run') + 1], 'run_0')
            self.assertEqual(Path(argv[argv.index('--output_root') + 1]), root / 'ieee33' / 'selections')
            sweep = json.loads((out.parent / 'selection_sweep' / f'step_{STEPS}' / 'run_manifest.json').read_text())
            self.assertEqual((sweep['b3']['controller'], sweep['b3']['action_mapping']), ('b3_mlp_a0', 'soc_feasible'))
            self.assertIn('b3_sources', sweep['b3'])

            # The maintained loader accepts the historical checkpoint (same controller, either label).
            from configs.experiments import feeder_config
            from environments.marl_env import make_marl_env
            env = make_marl_env(feeder_config('ieee33', series('ieee33'), 'soc_feasible'), 'val', 42, 0.0)
            model = maintained.load_model('mlp_a0', checkpoint, env, 'cpu')
            self.assertEqual((model.checkpoint_schema, model.controller), ('b3-checkpoint/2', 'mlp_a0'))
            with self.assertRaisesRegex(ValueError, 'controller'):
                maintained.load_model('mlp_a1', checkpoint, env, 'cpu')


if __name__ == '__main__':
    unittest.main()
