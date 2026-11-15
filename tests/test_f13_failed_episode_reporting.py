"""F13: failed/incomplete episodes must not contaminate complete-episode reward/economic averages.

Uses the real IEEE33 evaluator functions (loaded unchanged by test_experiment_integration.evaluator)
on a real AC environment. Short horizons are explicit (episode_length patched), not 96-step days.
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from configs.ieee33_config import IEEE33_CONFIG
from environments.primary_marl import make_primary_env
from training.transition_semantics import learning_valid
from utility.evaluation_reporting import eligible_checkpoint_rows
from test_experiment_integration import evaluator, EVALUATORS

ECON = ("total_reward_mean", "total_reward_team", "total_energy_cost", "total_grid_import_mwh",
        "total_curtailment_mwh", "total_throughput_mwh")


class FailingPF:
    """run_power_flow wrapper that raises after `ok_calls` successful solves."""
    def __init__(self, real, ok_calls):
        self.real, self.ok_calls, self.n = real, ok_calls, 0

    def __call__(self, **kw):
        self.n += 1
        if self.n > self.ok_calls:
            raise RuntimeError("injected power flow failure (F13 test)")
        return self.real(**kw)


class FailedEpisodeReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=123)
        cls.scope = evaluator(EVALUATORS[0])   # scripts/run_eval_ieee33.py
        cls.agent = SimpleNamespace(select_action=lambda obs, noise_std=0: np.zeros(cls.env.num_agents))

    def episodes(self):
        env, sc, start = self.env, self.scope, self.env.episode_start_indices[0]
        rows = []
        with patch.object(env, "episode_length", 2):          # ep0: complete, failure-free
            valid = sc["evaluate_one_episode"](env, self.agent, "maddpg", 0, start, 123)
        rows += valid
        real = env.power_flow.run_power_flow
        # ep1: reset PF + step-1 candidate PF + step-1 next-observation PF succeed (3 solves);
        # step 2's solves fail -> one good step, then a failed dispatch with the -10000 penalty.
        with patch.object(env, "episode_length", 3), \
                patch.object(env.power_flow, "run_power_flow", FailingPF(real, 3)):
            partial = sc["evaluate_one_episode"](env, self.agent, "maddpg", 1, start, 124)
        rows += partial
        failed_reset = dict(episode_failed=True, observation_valid=False, pf_failure_stage="reset",
                            pf_failed_solve_count=1)
        with patch.dict(sc, reset_at_episode_start=Mock(return_value=({"x": np.array([np.nan])}, failed_reset))):
            rows += sc["evaluate_one_episode"](env, self.agent, "maddpg", 2, start, 125)   # ep2: reset failure
            rows += sc["evaluate_one_episode"](env, self.agent, "zero", 0, start, 126)     # all-invalid group
        return valid, partial, pd.DataFrame(rows)

    def test_reporting(self):
        valid, partial, step_df = self.episodes()
        self.assertTrue(any(r["episode_failed"] for r in partial), "PF failure must have been injected")
        self.assertEqual(sum(r["record_type"] == "dispatch_attempt" for r in partial), 2)
        # Step-level diagnostics of the failed episode are preserved (including the penalty step).
        self.assertTrue((step_df[(step_df.episode_id == 1)]["reward_mean"] <= -1000).any())
        ep, agg = self.scope["summarize"](step_df)
        ep = ep.set_index(["policy", "episode_id"])
        v = ep.loc[("maddpg", 0)]
        self.assertTrue(bool(v["economic_totals_valid"]))
        self.assertAlmostEqual(v["total_reward_mean"], sum(r["reward_mean"] for r in valid), places=6)
        self.assertAlmostEqual(v["total_energy_cost"], sum(r["energy_cost"] for r in valid), places=9)
        for key in (("maddpg", 1), ("maddpg", 2), ("zero", 0)):
            self.assertFalse(bool(ep.loc[key, "economic_totals_valid"]))
            for col in ECON:
                self.assertTrue(np.isnan(ep.loc[key, col]), f"{key} {col} must be NaN")
        agg = agg.set_index("policy")
        m = agg.loc["maddpg"]
        # Headline means use completed, failure-free episodes only; denominators are explicit.
        self.assertAlmostEqual(m["mean_total_reward_mean"], v["total_reward_mean"], places=6)
        self.assertAlmostEqual(m["mean_energy_cost"], v["total_energy_cost"], places=9)
        self.assertEqual(m["economic_valid_episodes"], 1)
        self.assertEqual(m["economic_excluded_episodes"], 2)
        self.assertEqual((m["intended_episodes"], m["completed_episodes"], m["failed_episodes"]), (3, 1, 2))
        z = agg.loc["zero"]
        self.assertTrue(np.isnan(z["mean_total_reward_mean"]) and np.isnan(z["mean_energy_cost"]))
        self.assertEqual(z["economic_valid_episodes"], 0)
        # Checkpoint eligibility still rejects any evaluation containing failures.
        self.assertTrue(eligible_checkpoint_rows(agg.reset_index()).empty)

    def test_training_reward_and_replay_unchanged(self):
        env = self.env
        env.reset(seed=3)
        with patch.object(env.power_flow, "run_power_flow", side_effect=RuntimeError("injected")):
            obs, r, te, tr, info = env.step(np.zeros(env.num_agents, np.float32))
        self.assertTrue(np.all(np.asarray(r) == env.reward_cfg["nonconvergence_penalty"]))
        self.assertFalse(learning_valid({"x": np.zeros(1)}, np.zeros(env.num_agents), r, obs, info))

    def test_periodic_validation_reward_excludes_invalid_episodes(self):
        from utility.periodic_validation import evaluate_trainer
        env = self.env
        calls = {"n": 0}
        real_reset = env.reset

        def reset(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:   # second validation episode: reset PF failure
                return {"x": np.array([np.nan])}, dict(episode_failed=True, observation_valid=False,
                                                       pf_failed_solve_count=1)
            return real_reset(seed=10)
        with tempfile.TemporaryDirectory() as tmp, patch.object(env, "episode_length", 2), \
                patch.object(env, "reset", side_effect=reset), \
                patch("utility.periodic_validation.RunManifest"):
            (Path(tmp) / "validation" / "step_0").mkdir(parents=True)
            trainer = SimpleNamespace(eval_env=env, config={}, checkpoint_dir=tmp, global_step=0,
                                      eval_episodes=2, max_eval_steps=2, agent=self.agent,
                                      _step_env=lambda e, a: (lambda o: (o[0], o[1], bool(o[2] or o[3]), o[2], o[3], o[4]))(e.step(a)))
            result = evaluate_trainer(trainer)
            ep_df = pd.read_csv(Path(tmp) / "validation" / "step_0" / "episode_summary.csv")
        self.assertEqual(result["reward_valid_episodes"], 1)
        self.assertEqual(result["failed_episodes"], 1)
        self.assertTrue(np.isnan(ep_df.loc[1, "reward"]))
        self.assertAlmostEqual(result["mean_reward"], ep_df.loc[0, "reward"], places=9)


if __name__ == "__main__":
    unittest.main()
