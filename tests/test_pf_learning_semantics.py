"""Dependency-light tests of actual trainer methods and actual target expressions.

No claim of torch optimizer execution: test_learning_optimizer_smoke.py covers it.
"""
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_replay_action_semantics import make, TRAINERS, ROOT
from training.transition_semantics import learning_valid


class FailureLearning(unittest.TestCase):
    def test_failed_transition_excluded_and_reset_both_trainers(self):
        for path in TRAINERS:
            for stage in ('pre_action', 'candidate_search', 'next_observation', 'reset'):
                t, rows = make(path)
                t.env.step = lambda action: ({'x': np.zeros(2)}, np.ones(3), False, True,
                    dict(episode_failed=True, observation_valid=False, transition_valid=False,
                         pf_failure_stage=stage, step_pf_failed_solves=2))
                obs, _, done, info = t.collect_step({'x': np.ones(2)})
                self.assertTrue(done)
                self.assertEqual(rows, [])
                self.assertIs(obs, t.env.reset_obs)
                self.assertEqual(t.failed_episode_count, 1)
                self.assertEqual(t.pf_failed_solve_count, 2)
                self.assertFalse(info['replay_transition_stored'])

    def test_reset_failure_never_reaches_policy(self):
        for path in TRAINERS:
            t, rows = make(path)
            t.env.reset = lambda: ({'x': np.zeros(2)}, dict(episode_failed=True,
                 observation_valid=False, pf_failure_reason='injected reset failure'))
            with self.assertRaisesRegex(RuntimeError, 'Reset PF failed'):
                t._reset_env(t.env)
            self.assertEqual(rows, [])
            self.assertEqual(t.failed_episode_count, 1)

    def test_valid_time_limit_bootstraps_and_true_terminal_does_not(self):
        for path in TRAINERS:
            for terminal in (False, True):
                t, rows = make(path)
                next_obs = {'x': np.array([2.0])}
                t.env.step = lambda action: (next_obs, np.ones(3), terminal, not terminal, {})
                _, _, done, _ = t.collect_step({'x': np.array([1.0])})
                self.assertTrue(done)
                self.assertEqual(rows[0]['done'], terminal)
                self.assertIs(rows[0]['next_obs'], next_obs)

    def test_recovered_invalid_pre_action_is_excluded(self):
        for path in TRAINERS:
            t, rows = make(path)
            t.env.step = lambda action: ({'x': np.ones(2)}, np.ones(3), False, False,
                                        dict(transition_valid=False, episode_failed=False))
            _, _, done, _ = t.collect_step({'x': np.zeros(2)})
            self.assertFalse(done)
            self.assertEqual(rows, [])
            self.assertEqual(t.invalid_transition_count, 1)

    def test_nonfinite_and_split_boundary_excluded(self):
        for bad in (np.nan, np.inf):
            self.assertFalse(learning_valid({'x': [bad]}, [0], [1], {'x': [1]}, {}))
            self.assertFalse(learning_valid({'x': [1]}, [0], [1], {'x': [bad]}, {}))
        self.assertFalse(learning_valid({}, [0], [1], {}, {'transition_valid': False}))

    def test_actual_gnn_and_mlp_bellman_expressions(self):
        # Compile the expression from each production update(), not a copied formula.
        for rel in ('models/maddpg.py', 'experiments/no_gnn_maddpg/maddpg_mlp.py'):
            tree = ast.parse((ROOT/rel).read_text())
            assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'q_target_i' for t in n.targets))
            code = compile(ast.Expression(assignment.value), rel, 'eval')
            scope = dict(self=SimpleNamespace(gamma=.9), reward_i=np.array([[2.], [2.]]),
                         dones=np.array([[0.], [1.]]), target_q_i=np.array([[10.], [10.]]))
            np.testing.assert_allclose(eval(code, scope), [[11.], [2.]])

    def test_repeated_invalid_rollouts_cannot_hang_warmup(self):
        for path in TRAINERS:
            t, rows = make(path)
            t.max_consecutive_invalid_transitions = 2
            t.env.step = lambda action: ({'x': [1.]}, np.ones(3), False, True,
                                         dict(episode_failed=True, transition_valid=False))
            t.collect_step({'x': [1.]})
            with self.assertRaisesRegex(RuntimeError, 'Too many consecutive invalid'):
                t.collect_step({'x': [1.]})
            self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
