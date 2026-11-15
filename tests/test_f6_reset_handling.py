"""F6: training-reset handling -- fail fast by default, opt-in bounded deterministic retries.

Uses the real trainer methods (AST-extracted, as test_replay_action_semantics does) and a real
MLP trainer on the real IEEE33 environment. Invalid reset observations never reach replay.
"""
import copy
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_replay_action_semantics import make, TRAINERS
from training.transition_semantics import learning_valid

FAILED = ({"x": np.array([np.nan])}, dict(episode_failed=True, observation_valid=False,
                                         pf_failure_reason="injected reset failure", pf_failed_solve_count=1))


def scripted(results):
    seq = list(results)
    return lambda: seq.pop(0)


class ResetRetries(unittest.TestCase):
    def test_default_fails_fast(self):
        for path in TRAINERS:
            t, rows = make(path)
            t.env.reset = scripted([FAILED, ({"x": np.ones(2)}, {})])
            with self.assertRaisesRegex(RuntimeError, r"Reset PF failed.*injected reset failure.*1 attempt"):
                t._reset_env(t.env)
            self.assertEqual((t.reset_attempts, len(t.reset_failures), t.failed_episode_count), (1, 1, 1))
            self.assertEqual(t.reset_failures[0]["reason"], "injected reset failure")
            self.assertEqual(rows, [])

    def test_configured_recovery(self):
        for path in TRAINERS:
            t, rows = make(path)
            t.max_reset_retries = 2
            good = ({"x": np.ones(2)}, {"observation_valid": True})
            t.env.reset = scripted([FAILED, good])
            obs, info = t._reset_env(t.env)
            self.assertIs(obs, good[0])
            self.assertEqual((t.reset_attempts, len(t.reset_failures)), (2, 1))
            self.assertEqual(t.failed_episode_count, 1)
            self.assertEqual(t.pf_failed_solve_count, 1)
            self.assertEqual(rows, [])

    def test_retry_exhaustion(self):
        for path in TRAINERS:
            t, rows = make(path)
            t.max_reset_retries = 2
            t.env.reset = scripted([FAILED, FAILED, FAILED, ({"x": np.ones(2)}, {})])
            with self.assertRaisesRegex(RuntimeError, r"3 attempt\(s\), max_reset_retries=2"):
                t._reset_env(t.env)
            self.assertEqual([f["attempt"] for f in t.reset_failures], [1, 2, 3])
            self.assertEqual(rows, [])


def mlp_trainer(env, retries, tmp="."):
    from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
    return MLPMADDPGTrainer(env, None, dict(batch_size=4, buffer_size=64, warmup_steps=20, total_steps=20,
                                            max_reset_retries=retries, checkpoint_dir=tmp, verbose=False,
                                            actor_hidden_dims=(8,), critic_hidden_dims=(8,),
                                            store_buffer_on_device=False), device="cpu")


class RealTrainer(unittest.TestCase):
    def test_config_validation_and_replay_safety(self):
        import tempfile
        from configs.ieee33_config import IEEE33_CONFIG
        from environments.primary_marl import make_primary_env
        with tempfile.TemporaryDirectory() as tmp:
            env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=3)
            for bad in (-1, 11):
                with self.assertRaisesRegex(ValueError, "max_reset_retries"):
                    mlp_trainer(env, bad, tmp)
            starts = []
            for _ in range(2):                                    # identical runs -> identical recovery day
                env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=3)
                t = mlp_trainer(env, 1, tmp)
                real, calls = env.reset, {"n": 0}

                def reset(*a, **k):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        real(*a, **k)        # advances the env RNG exactly as a real failed reset would
                        return FAILED
                    return real(*a, **k)
                env.reset = reset
                with redirect_stdout(io.StringIO()):
                    t.warmup()
                starts.append(t.reset_failures[0]["start_index"])
                self.assertEqual(t.reset_summary()["reset_attempts"], 2)
                buf = t.replay_buffer
                n = len(buf)
                self.assertEqual(n, 20)
                for name in ("agent_obs", "next_agent_obs"):
                    self.assertTrue(np.isfinite(np.asarray(getattr(buf, name)[:n].cpu())).all(), name)
            self.assertEqual(starts[0], starts[1])

    def test_evaluation_never_uses_training_retries(self):
        import tempfile
        from configs.ieee33_config import IEEE33_CONFIG
        from environments.primary_marl import make_primary_env
        with tempfile.TemporaryDirectory() as tmp:
            env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=3)
            val = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=4)
            t = mlp_trainer(env, 3, tmp)
            t.eval_env, t.eval_episodes, t.max_eval_steps = val, 2, 2
            val.reset = lambda *a, **k: FAILED
            with redirect_stdout(io.StringIO()):
                result = t.evaluate()
            self.assertEqual(result["failed_episodes"], 2)          # recorded, not retried
            self.assertEqual(t.reset_attempts, 0)


if __name__ == "__main__":
    unittest.main()
