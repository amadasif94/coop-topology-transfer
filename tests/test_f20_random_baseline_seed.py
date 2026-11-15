"""F20: the random baseline (env.action_space.sample) is reproducible per evaluation seed."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from configs.ieee33_config import IEEE33_CONFIG
from environments.primary_marl import make_primary_env
from utility.evaluation_reporting import reset_for_evaluation
from test_experiment_integration import evaluator, EVALUATORS


class RandomBaselineSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=123)

    def draws(self, seed, n=20):
        reset_for_evaluation(self.env, self.env.episode_start_indices[0], seed)
        return np.array([self.env.action_space.sample() for _ in range(n)])

    def test_same_seed_same_actions(self):
        a, b, c = self.draws(5), self.draws(5), self.draws(6)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        # Distribution preserved: still the env's Box action space.
        self.assertTrue(np.all(a >= self.env.action_space.low) and np.all(a <= self.env.action_space.high))
        self.assertEqual(a.shape[1], self.env.num_agents)

    def test_real_evaluator_random_policy_reproducible(self):
        env, scope = self.env, evaluator(EVALUATORS[0])
        start = env.episode_start_indices[0]
        runs = []
        for _ in range(2):
            with patch.object(env, "episode_length", 3):
                rows = scope["evaluate_one_episode"](env, None, "random", 0, start, 77)
            runs.append(pd.DataFrame(rows))
        cols = [c for c in runs[0] if c.startswith(("requested_action_agent_", "dess_power_kw_agent_"))] + ["reward_mean"]
        pd.testing.assert_frame_equal(runs[0][cols], runs[1][cols])

    def test_none_seed_and_missing_action_space_are_safe(self):
        reset_for_evaluation(self.env, self.env.episode_start_indices[0], None)

        class NoSpace:
            episode_start_indices = [0]

            def reset(self, seed=None):
                return {}, {}
        reset_for_evaluation(NoSpace(), 0, 3)


if __name__ == "__main__":
    unittest.main()
