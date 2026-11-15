"""F9: identical, explicit, reproducible exploration for GCN, GAT, TAGConv and MLP trainers.

Real trainers on the real IEEE33 environment (tiny networks). No training campaign.
"""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.ieee33_config import IEEE33_CONFIG
from environments.primary_marl import make_primary_env

ARCHS = ("gcn", "gat", "tagconv", "mlp")
SMALL = dict(batch_size=4, buffer_size=64, warmup_steps=16, total_steps=16, verbose=False, store_buffer_on_device=False,
             gnn_hidden_dim=8, gnn_embedding_dim=8, gnn_num_layers=2, actor_hidden_dims=(8,), critic_hidden_dims=(8,))


def trainer(arch, env, tmp, **extra):
    conf = dict(SMALL, checkpoint_dir=tmp, **extra)
    if arch == "mlp":
        from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
        return MLPMADDPGTrainer(env, None, conf, device="cpu")
    from training.trainer import GraphMADDPGTrainer
    return GraphMADDPGTrainer(env, None, dict(conf, gnn_type=arch), device="cpu")


class Exploration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=0)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_identical_defaults_across_architectures(self):
        configs = {}
        for arch in ARCHS:
            t = trainer(arch, self.env, self.tmp.name)
            cfg = t.exploration_config()
            self.assertEqual(cfg["controller"], arch)
            configs[arch] = {k: v for k, v in cfg.items() if k != "controller"}
            self.assertEqual((t.noise_std, t.warmup_action_low, t.warmup_action_high), (0.05, -1.0, 1.0), arch)
        for arch in ARCHS[1:]:
            self.assertEqual(configs[arch], configs["gcn"], arch)
        self.assertEqual(configs["gcn"]["evaluation_noise_std"], 0.0)

    def test_warmup_covers_action_space_and_is_reproducible(self):
        for arch in ARCHS:
            t = trainer(arch, self.env, self.tmp.name)
            np.random.seed(7)
            a = np.stack([t._random_action() for _ in range(4000)])
            self.assertEqual(a.shape[1], self.env.num_agents)
            self.assertTrue(a.min() >= -1.0 and a.max() <= 1.0, arch)
            self.assertTrue(a.min() < -0.95 and a.max() > 0.95, f"{arch}: warmup must span [-1, 1] (GNN was +-0.75)")
            np.random.seed(7)
            b = np.stack([t._random_action() for _ in range(4000)])
            np.testing.assert_array_equal(a, b)

    def test_configured_values_are_respected_and_validated(self):
        for arch in ("gcn", "mlp"):
            t = trainer(arch, self.env, self.tmp.name, noise_std=0.02, warmup_action_low=-0.5, warmup_action_high=0.5)
            self.assertEqual(t.exploration_config()["noise_std"], 0.02)
            np.random.seed(0)
            a = np.stack([t._random_action() for _ in range(500)])
            self.assertTrue(a.min() >= -0.5 and a.max() <= 0.5)
            for bad in (dict(warmup_action_low=-1.5), dict(warmup_action_low=0.5, warmup_action_high=0.5),
                        dict(noise_std=-0.1), dict(noise_std=float("nan"))):
                with self.assertRaises(ValueError, msg=str(bad)):
                    trainer(arch, self.env, self.tmp.name, **bad)

    def test_noise_reproducible_clipped_and_off_for_evaluation(self):
        obs, _ = self.env.reset(seed=1)
        for arch in ARCHS:
            t = trainer(arch, self.env, self.tmp.name)
            outs = []
            for _ in range(2):
                torch.manual_seed(3)
                outs.append(t.agent.select_action(obs, noise_std=t.noise_std))
            np.testing.assert_array_equal(outs[0], outs[1])          # same seed -> same noise
            clean = t.agent.select_action(obs, noise_std=0.0)
            np.testing.assert_array_equal(clean, t.agent.select_action(obs, noise_std=0.0))  # deterministic
            self.assertFalse(np.array_equal(outs[0], clean))
            loud = t.agent.select_action(obs, noise_std=50.0)
            self.assertTrue(np.all(np.abs(loud) <= 1.0))               # clipping preserved
        # Periodic validation and evaluators call the actor with noise_std=0.
        from utility.periodic_validation import evaluate_trainer
        val = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=2)
        t = trainer("mlp", self.env, self.tmp.name)
        t.eval_env, t.eval_episodes, t.max_eval_steps, t.global_step = val, 1, 2, 1
        seen = []
        real = t.agent.select_action
        t.agent.select_action = lambda o, noise_std=0.0: seen.append(noise_std) or real(o, noise_std=noise_std)
        with redirect_stdout(io.StringIO()):
            evaluate_trainer(t)
        self.assertEqual(set(seen), {0.0})
        for rel in ("scripts/run_eval_ieee33.py", "scripts/run_eval_ieee69.py",
                    "experiments/no_gnn_maddpg/run_eval_mlp_ieee33.py", "experiments/topology_generalization/run_tp_eval.py"):
            src = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("select_action(obs, noise_std=0.0)", src.replace("\r", ""), rel)

    def test_manifest_records_exploration(self):
        from utility.run_manifest import RunManifest
        with tempfile.TemporaryDirectory() as tmp:
            t = trainer("gat", self.env, tmp)
            run = RunManifest(Path(tmp) / "run", "gat", 42, "train")
            run.record_exploration(dict(t.exploration_config(), seed=42))
            man = json.loads(run.path.read_text())
        self.assertEqual(man["exploration"]["warmup_action_low"], -1.0)
        self.assertEqual(man["exploration"]["noise_std"], 0.05)
        self.assertEqual(man["exploration"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
