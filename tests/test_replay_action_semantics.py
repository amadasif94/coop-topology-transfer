"""Run actual trainer collection methods without optional model dependencies.

AST extraction avoids importing missing GNN modules/PyTorch. Only environment,
agent and replay sink are test doubles; trainer methods are compiled unchanged.
This is a collection smoke test, not a neural-network optimization test.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import unittest
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.transition_semantics import learning_valid, finite_tree

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = ('training/trainer.py', 'experiments/no_gnn_maddpg/trainer_mlp.py')


def load_collection(path):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
               n.name in ('collect_step', '_step_env', '_reset_env', '_random_action')]
    node = ast.ClassDef(name='Collector', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {'np': np, 'Dict': Dict, 'learning_valid': learning_valid, 'finite_tree': finite_tree}
    exec(compile(module, str(ROOT / path), 'exec'), scope)
    return scope['Collector']


class NoActionDiagnostics(dict):
    def get(self, *args, **kwargs):
        if args[0] in ('accepted_action', 'requested_action'):
            raise AssertionError('Training must not use action fallback')
        return super().get(*args, **kwargs)

    def __getitem__(self, key):
        raise AssertionError('No action diagnostics are required by collect_step')


class Agent:
    def __init__(self, dtype, column):
        self.clean = np.array([.9, -.9, .1], dtype=dtype)
        self.noise = np.array([.5, -.5, .5], dtype=dtype)
        self.column = column

    def select_action(self, obs, noise_std):
        self.used_noise_std = noise_std
        self.output = np.clip(self.clean + noise_std*self.noise, -1, 1)
        if self.column:
            self.output = self.output[:, None]
        return self.output


class Env:
    def __init__(self, missing=False, legacy=False):
        self.missing, self.legacy = missing, legacy
        self.count = 0
        self.reset_obs = {'reset': True}

    def reset(self):
        return self.reset_obs, {}

    def step(self, action):
        self.count += 1
        self.received = action.copy()
        self.passed_array = action
        # Simulate in-place filtering; actor output and saved replay must survive.
        action[:] = 0
        self.reward = np.array([-3, -2, -1], dtype=np.float32)
        self.next_obs = {'executed_step': self.count, 'executed_power': action.copy()}
        self.info = NoActionDiagnostics() if self.missing else {
            'accepted_action': action.copy(), 'requested_action': action.copy()}
        done = self.count % 3 == 0
        if self.legacy:
            return self.next_obs, self.reward, done, self.info
        return self.next_obs, self.reward, False, done, self.info


def make(path, dtype=np.float32, column=False, missing=False, legacy=False):
    trainer = load_collection(path)()
    trainer.agent = Agent(dtype, column)
    trainer.env = Env(missing, legacy)
    trainer.noise_std = .5
    trainer.num_agents = 3
    trainer.global_step = trainer.episode_count = 0
    records = []
    # Intentionally retain references: copying by a sink cannot hide alias bugs.
    trainer.replay_buffer = SimpleNamespace(add=lambda **kw: records.append(kw))
    return trainer, records


class ReplaySemantics(unittest.TestCase):
    def test_noisy_clipped_proposal_shapes_and_mutation(self):
        for path in TRAINERS:
            for dtype in (np.float32, np.float64):
                for column in (False, True):
                    with self.subTest(path=path, dtype=dtype, column=column):
                        t, rows = make(path, dtype, column)
                        obs = {'observation': 'before'}
                        returned, reward, done, info = t.collect_step(obs)
                        expected = np.array([1, -1, .35], dtype=dtype)
                        if column: expected = expected[:, None]
                        r = rows[0]
                        np.testing.assert_allclose(r['action'], expected)
                        np.testing.assert_array_equal(r['action'], t.env.received)
                        np.testing.assert_array_equal(t.agent.output, t.env.received)
                        self.assertEqual(r['action'].dtype, dtype)
                        self.assertEqual(r['action'].shape, expected.shape)
                        self.assertEqual(t.agent.used_noise_std, t.noise_std)
                        self.assertIs(r['obs'], obs)
                        self.assertIs(r['reward'], reward)
                        self.assertIs(r['next_obs'], returned)
                        self.assertFalse(done)
                        self.assertFalse(np.shares_memory(r['action'], t.agent.output))
                        self.assertFalse(np.shares_memory(r['action'], t.env.passed_array))
                        self.assertFalse(np.array_equal(r['action'], info['accepted_action']))

    def test_missing_diagnostics_no_substitution(self):
        for path in TRAINERS:
            t, rows = make(path, missing=True)
            t.collect_step({})
            np.testing.assert_array_equal(rows[0]['action'], t.env.received)

    def test_warmup_proposal_preserved(self):
        for path in TRAINERS:
            t, rows = make(path)
            original = np.array([-.6, .2, .7], dtype=np.float32)
            t._random_action = lambda: original
            t.collect_step({}, random_action=True)
            np.testing.assert_array_equal(rows[0]['action'], original)
            np.testing.assert_allclose(original, [-.6, .2, .7])

    def test_deterministic_collection_smoke_and_terminal_observation(self):
        results = []
        for path in TRAINERS:
            for legacy in (False, True):
                t, rows = make(path, legacy=legacy)
                obs = {}
                for i in range(6):
                    obs, reward, done, _ = t.collect_step(obs)
                    self.assertIs(rows[-1]['next_obs'], t.env.next_obs)
                    self.assertIs(rows[-1]['reward'], t.env.reward)
                    self.assertEqual(rows[-1]['done'], done if legacy else False)
                    if done: self.assertIs(obs, t.env.reset_obs)
                self.assertEqual(t.global_step, 6)
                self.assertEqual(t.episode_count, 2)
                results.append(np.stack([r['action'] for r in rows]))
        for result in results[1:]: np.testing.assert_array_equal(result, results[0])

    def test_all_known_collection_insertions_use_saved_proposal(self):
        for path in TRAINERS:
            tree = ast.parse((ROOT / path).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'add'
                     and ast.unparse(n.func.value) == 'self.replay_buffer']
            self.assertEqual(len(calls), 1)
            action = next(k.value for k in calls[0].keywords if k.arg == 'action')
            self.assertEqual(ast.unparse(action), 'proposed_action')


if __name__ == '__main__':
    unittest.main()
