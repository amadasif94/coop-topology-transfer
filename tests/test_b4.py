"""Experiment b4 (utility.experiment_setup.apply_experiment): new battery ratings, training-only random initial SoC,
two-sided terminal stored-energy valuation replacing the quadratic terminal penalty. b3 must stay unchanged.

Real-environment checks use the processed time series (data/time_series on RIT, or the local audit_tmp/replica copy)
and are skipped without one. Run: python -m unittest tests.test_b4 -v
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from environments.feasibility_env import terminal_energy_value
from experiments.no_shared_kpi.protocol import inventory_adjusted_cost
from tests.test_b3 import series
from utility.experiment_setup import feeder_config, load_coefficients, output_root_for

RATINGS = dict(ieee33=(5, 700.0, 140.0, 0.3945), ieee69=(9, 850.0, 170.0, 0.3984))


class Configuration(unittest.TestCase):
    def test_b3_config_is_unchanged(self):
        from configs.ieee33_config import IEEE33_CONFIG
        from configs.ieee69_config import IEEE69_CONFIG
        for feeder, original in (('ieee33', IEEE33_CONFIG), ('ieee69', IEEE69_CONFIG)):
            self.assertEqual(json.dumps(feeder_config(feeder), sort_keys=True, default=str),
                             json.dumps(original, sort_keys=True, default=str))
            self.assertNotIn('experiment_protocol', feeder_config(feeder))
        self.assertEqual(IEEE33_CONFIG['dess']['capacity_kwh'], 500.0)       # module constants never modified
        self.assertEqual(IEEE69_CONFIG['dess']['max_charge_kw'], 250.0)

    def test_b4_battery_ratings(self):
        for feeder, (n, cap, power, penetration) in RATINGS.items():
            dess = feeder_config(feeder, experiment='b4')['dess']
            self.assertEqual((dess['capacity_kwh'], dess['max_charge_kw'], dess['max_discharge_kw']), (cap, power, power))
            self.assertEqual((dess['soc_min'], dess['soc_max'], dess['soc_init']), (0.10, 0.90, 0.50))
            self.assertEqual((dess['charge_efficiency'], dess['discharge_efficiency']), (0.95, 0.95))
            self.assertEqual(len(dess['dess_buses']), n)
            p = feeder_config(feeder, experiment='b4')['experiment_protocol']
            self.assertAlmostEqual(p['battery']['c_rate'], 0.20)
            self.assertAlmostEqual(p['battery']['power_penetration'], penetration, places=4)
            self.assertEqual(p['battery']['fleet_capacity_kwh'], n * cap)
            self.assertEqual(p['initial_soc'], dict(train_uniform=[0.30, 0.70], validation_and_test=0.50))
            self.assertFalse(p['terminal']['quadratic_shortfall_penalty'])

    def test_coefficients_and_output_roots(self):
        self.assertEqual(load_coefficients('ieee33', experiment='b4')[:2], (3.0, 0.0))
        self.assertEqual(load_coefficients('ieee69')[:2], (3.0, 186.923))
        with self.assertRaises(ValueError):
            load_coefficients('ieee33', None, 51.923, 'b4')
        self.assertEqual(Path(output_root_for('b3')), Path('results/b3'))
        self.assertEqual(Path(output_root_for('b4')), Path('results/b4'))
        with self.assertRaises(ValueError):
            output_root_for('b4', 'results/b3')
        with self.assertRaises(ValueError):
            feeder_config('ieee33', experiment='b5')


class TerminalValuation(unittest.TestCase):
    def test_signs_scaling_and_units(self):
        v = terminal_energy_value(40.0, [0.5, 0.5, 0.5], [0.4, 0.6, 0.5], [700.0, 850.0, 700.0])
        np.testing.assert_allclose(v, [-40 * 70 / 1000, 40 * 85 / 1000, 0.0])      # charge, credit, zero ($)
        np.testing.assert_allclose(terminal_energy_value(40.0, [0.3], [0.5], [850.0]),
                                   2 * terminal_energy_value(40.0, [0.3], [0.4], [850.0]))
        np.testing.assert_allclose(terminal_energy_value(80.0, [0.5], [0.4], [700.0]),
                                   2 * terminal_energy_value(40.0, [0.5], [0.4], [700.0]))

    def test_matches_adjusted_net_daily_cost(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            s0, sT, cap = rng.uniform(.1, .9, 9), rng.uniform(.1, .9, 9), np.full(9, 850.0)
            price, raw = rng.uniform(10, 90), rng.uniform(100, 900)
            adjusted = inventory_adjusted_cost(raw, float(s0 @ cap), float(sT @ cap), price)
            self.assertAlmostEqual(raw - terminal_energy_value(price, s0, sT, cap).sum(), adjusted, places=9)


class Discount(unittest.TestCase):
    def test_protocol_gamma_and_refusal(self):
        from utility.experiment_setup import B4_GAMMA, TRAIN_SETTINGS
        self.assertEqual(B4_GAMMA, 1.0)
        self.assertEqual(TRAIN_SETTINGS['gamma'], 0.99)                                   # B3 unchanged
        self.assertEqual(feeder_config('ieee69', experiment='b4')['experiment_protocol']['discount'],
                         dict(gamma=1.0, completed_day_terminal=True))
        from experiments.b3.train import main
        with self.assertRaises(SystemExit):                                               # before any output exists
            main(['--feeder', 'ieee33', '--controller', 'mlp_a0', '--run', '0', '--experiment', 'b4', '--gamma', '0.95'])

    def test_critic_targets_stop_at_terminal_with_gamma_one(self):
        """All four learners: y = r exactly where done = 1; bootstrapped (depends on next obs) where done = 0."""
        from unittest.mock import patch
        import torch
        from models.feasibility_agents import build_model
        from tests.test_b3 import batch
        n, B = 5, 16
        done = torch.zeros(B, 1)
        done[::2] = 1.0
        for controller in ('mlp_a0', 'mlp_a1', 'mlp_a2', 'mlp_a3'):
            torch.manual_seed(0)
            model = build_model(controller, 3.0, agent_obs_dim=10, num_agents=n, action_dim_per_agent=1, actor_lr=5e-5,
                                critic_lr=5e-4, gamma=1.0, grad_clip_norm=None, device='cpu', share_actor=False)
            self.assertEqual(model.gamma, 1.0)
            targets = []
            for shift in (0.0, 0.5):
                b = batch(n, B, seed=3)
                b['dones'] = done.clone()
                b['next_agent_obs'] = b['next_agent_obs'] + shift
                captured = []
                real = torch.nn.functional.mse_loss

                def spy(inp, target, *a, **k):
                    captured.append(target.detach().clone())
                    return real(inp, target, *a, **k)
                with patch('torch.nn.functional.mse_loss', side_effect=spy):
                    deepcopy(model).update(b)
                y = torch.cat(captured[:n], dim=1)                                            # one target per critic
                torch.testing.assert_close(y[::2], b['rewards'][::2])                       # terminal: no bootstrap
                targets.append(y)
            self.assertTrue((targets[0][1::2] != targets[1][1::2]).all(), controller)     # non-terminal: bootstraps
            torch.testing.assert_close(targets[0][::2], targets[1][::2])


@unittest.skipUnless(series('ieee33') and series('ieee69'), 'no processed time series')
class RealEnvironment(unittest.TestCase):
    def env(self, feeder, mode='train', seed=7, a3=False, experiment='b4'):
        from environments.feasibility_env import make_b3_env
        from environments.local_reward_env import make_a3_env
        cfg = feeder_config(feeder, series(feeder), experiment)
        lam = 0.0 if experiment == 'b4' else load_coefficients(feeder)[1]
        return (make_a3_env if a3 else make_b3_env)(cfg, mode, seed, lam)

    def test_physical_ratings_in_env(self):
        from environments.battery_feasibility import normalized_bounds
        for feeder, (n, cap, power, _) in RATINGS.items():
            env = self.env(feeder)
            env.reset(seed=1)
            self.assertEqual(env.num_agents, n)
            for b in env.batteries:
                self.assertEqual((b.config.capacity_kwh, b.config.max_charge_kw, b.config.max_discharge_kw), (cap, power, power))
                self.assertEqual((b.config.soc_min, b.config.soc_max), (0.10, 0.90))
                self.assertAlmostEqual(b.get_available_discharge_kw(), power)       # mid-range SoC: rating binds
            lo, hi = normalized_bounds(env)
            np.testing.assert_allclose(lo, -1.0)
            np.testing.assert_allclose(hi, 1.0)
        with self.assertRaises(ValueError):
            from environments.feasibility_env import make_b3_env
            make_b3_env(feeder_config('ieee33', series('ieee33'), 'b4'), 'train', 1, 51.923)   # quadratic refused

    def test_training_initial_soc(self):
        from environments.dess_env import DESSEnv
        env, twin = self.env('ieee33'), self.env('ieee33')
        draws = []
        for k in range(6):
            env.reset(seed=100 + k)
            soc = np.array([b.get_soc() for b in env.batteries])
            self.assertTrue(((soc >= 0.30) & (soc <= 0.70)).all())
            self.assertGreater(soc.max() - soc.min(), 1e-6)                          # independent per battery
            np.testing.assert_array_equal(env.b3_start_soc, soc)                     # kept for terminal accounting
            twin.reset(seed=100 + k)
            np.testing.assert_array_equal(np.array([b.get_soc() for b in twin.batteries]), soc)   # reproducible
            draws.append(soc)
        self.assertGreater(np.ptp(np.array(draws)[:, 0]), 0.05)                       # varies between episodes
        # Unseeded resets continue the seeded stream reproducibly.
        a, b = self.env('ieee33', seed=3), self.env('ieee33', seed=3)
        for _ in range(3):
            a.reset(); b.reset()
            np.testing.assert_array_equal([x.get_soc() for x in a.batteries], [x.get_soc() for x in b.batteries])
        # Observation rebuilt with the drawn SoC; every other feature identical to the original reset.
        obs, _ = env.reset(seed=5)
        drawn = np.array([b.get_soc() for b in env.batteries])
        base, _ = DESSEnv.reset(env, seed=5)                                         # original reset: soc_init 0.50
        self.assertTrue(all(b.get_soc() == 0.5 for b in env.batteries))
        changed = np.where(~np.isclose(obs['agent_obs'], base['agent_obs']).all(axis=0))[0]
        self.assertEqual(changed.tolist(), [2])
        np.testing.assert_allclose(obs['agent_obs'][:, 2], [env.normalizer.normalize_soc(s) for s in drawn])
        x_changed = np.where(~np.isclose(obs['x'], base['x']).all(axis=0))[0]
        self.assertLessEqual(len(x_changed), 1)
        # The step starts from the drawn SoC; the day sequence is the same as without the draw.
        env.reset(seed=9)
        drawn = np.array([b.get_soc() for b in env.batteries])
        _, _, _, _, info = env.step(np.zeros(env.num_agents, dtype=np.float32))
        np.testing.assert_allclose(info['soc_before'], drawn)
        fresh, plain = self.env('ieee33', seed=21), self.env('ieee33', seed=21, experiment='b3')
        for k in range(4):                                     # SoC draws use their own stream, not the day sampler
            fresh.reset(); plain.reset()
            self.assertEqual(fresh.episode_start_index, plain.episode_start_index)

    def test_validation_starts_at_half(self):
        for feeder in RATINGS:
            env = self.env(feeder, mode='val', seed=1042)
            for k in range(3):
                env.reset(seed=None if k else 1)
                self.assertTrue(all(b.get_soc() == 0.5 for b in env.batteries))
                np.testing.assert_array_equal(env.b3_start_soc, 0.5)

    def test_unseen_topology_test_mode_starts_at_half(self):
        """Reset only (no rollout, no result): TP2 keeps the b4 batteries/protocol; test episodes start at 0.50."""
        from environments.feasibility_env import make_b3_env
        from environments.local_reward_env import make_a3_env
        from utility.evaluation_protocol import authorize, protocol_config
        from utility.experiment_setup import topology_config
        protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=False, topologies=['TP1', 'TP2'])
        with tempfile.TemporaryDirectory() as d:
            cfg = protocol_config(topology_config('ieee33', 'TP2', Path(d), series('ieee33'), 'b4'), protocol)
            self.assertEqual(cfg['dess']['capacity_kwh'], 700.0)
            self.assertEqual(cfg['experiment_protocol']['name'], 'b4')
            for factory in (make_b3_env, make_a3_env):
                for scale in (1.0, 1.3):
                    env = factory(deepcopy(cfg), 'test', 42, 0.0, scale)
                    for k in range(3):
                        env.reset(seed=None if k else 42)
                        self.assertTrue(all(b.get_soc() == 0.5 for b in env.batteries))

    def run_day(self, env, seed, rng):
        env.reset(seed=seed)
        rows = []
        for t in range(env.episode_length):
            _, reward, _, _, info = env.step(rng.uniform(-1, 1, env.num_agents).astype(np.float32))
            self.assertFalse(info.get('episode_failed', False))
            rows.append((np.asarray(reward, float), info))
        return rows

    def test_team_terminal_valuation_on_last_transition(self):
        for feeder in RATINGS:
            env = self.env(feeder)
            rows = self.run_day(env, 11, np.random.default_rng(0))
            w = env.reward_cfg['w_grid'] * env.reward_cfg['phi_cost']
            start = rows[0][1]['soc_before']
            prices = np.array([r[1]['kpis']['price'] for r in rows])
            for _, info in rows[:-1]:
                self.assertEqual(info['marl_terminal_penalty'], 0.0)
                self.assertFalse(info['marl_terminal_completed'])
                self.assertFalse(info['terminated'] or info['truncated'])
            reward, last = rows[-1]
            self.assertTrue(last['marl_terminal_completed'])
            self.assertTrue(last['terminated'] and last['truncated'])                        # finite horizon (b4)
            cap = np.array([b.config.capacity_kwh for b in env.batteries])
            self.assertAlmostEqual(last['marl_day_mean_price'], prices.mean(), places=9)
            # info soc_before/soc_after are float32 exports of the float64 battery SoC used by the env (~1e-7 relative).
            value = prices.mean() * (last['soc_after'] * cap - start * cap) / 1000
            np.testing.assert_allclose(last['marl_terminal_energy_value_usd'], value, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(last['reward_term_terminal_soc'], w * value.sum(), rtol=1e-5, atol=1e-4)
            exact = terminal_energy_value(prices.mean(), env.b3_start_soc, [b.get_soc() for b in env.batteries], cap)
            np.testing.assert_allclose(last['marl_terminal_energy_value_usd'], exact, rtol=1e-12)
            self.assertEqual(env.lambda_terminal, 0.0)                               # no quadratic term remains
            # Reward exactly = original step reward + team valuation (rewards are float32).
            np.testing.assert_allclose(reward, np.asarray(last['reward_terms_total'], float), rtol=1e-5, atol=1e-4)
            # Economic identity with the adjusted net daily cost.
            raw = sum(r[1]['kpis']['price'] * max(0.0, r[1]['kpis']['grid_import_kw']) * 0.25 / 1000 for r in rows)
            grid = sum(float(r[1]['reward_term_grid_energy_cost'][0]) for r in rows)
            adjusted = inventory_adjusted_cost(raw, float(env.b3_start_soc @ cap),
                                               float(np.array([b.get_soc() for b in env.batteries]) @ cap), prices.mean())
            self.assertAlmostEqual(grid + float(last['reward_term_terminal_soc'][0]), -w * adjusted, places=6)
            # The gamma = 1 return of these components IS that identity; a gamma = 0.99 return would not be.
            costs = [float(r[1]['reward_term_grid_energy_cost'][0]) for r in rows]
            costs[-1] += float(last['reward_term_terminal_soc'][0])
            undiscounted = sum(1.0 ** t * c for t, c in enumerate(costs))
            discounted = sum(0.99 ** t * c for t, c in enumerate(costs))
            self.assertAlmostEqual(undiscounted, -w * adjusted, places=6)
            self.assertGreater(abs(discounted + w * adjusted), 1e-3)

    def test_a3_individual_terminal_valuation(self):
        env = self.env('ieee33', a3=True)
        rows = self.run_day(env, 12, np.random.default_rng(1))
        w = env.reward_cfg['w_grid'] * env.reward_cfg['phi_cost']
        last = rows[-1][1]
        own = w * np.asarray(last['marl_terminal_energy_value_usd'])
        np.testing.assert_allclose(last['reward_term_a3_terminal_soc'], own, rtol=1e-9)
        self.assertGreater(np.ptp(own), 0)                                          # individual, not shared
        np.testing.assert_allclose(own.sum(), last['reference_a0_reward_term_terminal_soc'][0], rtol=1e-9)
        self.assertIsNone(last['marl_terminal_coefficients'])
        for _, info in rows[:-1]:
            np.testing.assert_array_equal(info['reward_term_a3_terminal_soc'], 0.0)

    def test_incomplete_day_gets_no_terminal_value_and_is_not_terminal(self):
        """A failed dispatch (injected at the last step, or mid-day) is never a completed day: no terminal valuation,
        terminated stays False, and the trainer does not store the failed transition."""
        from unittest.mock import patch
        from environments.dess_env import DESSEnv
        from training.transition_semantics import learning_valid
        real = DESSEnv.step
        for fail_at in (95, 40):
            env = self.env('ieee33')
            env.reset(seed=13)

            def failing(self_, action, fail_at=fail_at):
                index = int(self_.current_step)
                out = list(real(self_, action))
                if index == fail_at:
                    out[4] = dict(out[4], episode_failed=True, transition_valid=False, terminated=False, truncated=True)
                    out[2], out[3] = False, True
                return tuple(out)
            with patch.object(DESSEnv, 'step', failing):
                for t in range(fail_at + 1):
                    obs, reward, terminated, truncated, info = env.step(np.zeros(env.num_agents, dtype=np.float32))
            self.assertTrue(info['episode_failed'])
            self.assertFalse(info['marl_terminal_completed'])
            self.assertFalse(terminated)
            self.assertEqual(info['marl_terminal_penalty'], 0.0)
            self.assertTrue(np.isnan(info['marl_terminal_energy_value_usd']).all())
            self.assertFalse(learning_valid(obs, np.zeros(env.num_agents), reward, obs, info))

    def test_replay_marks_day_end_terminal(self):
        from training.feasibility_trainer import b3_trainer_class
        for experiment, expected in (('b4', [95, 191]), ('b3', [])):
            env = self.env('ieee33', experiment=experiment)
            with tempfile.TemporaryDirectory() as d:
                cfg = dict(total_steps=200, warmup_steps=200, buffer_size=500, batch_size=32, eval_every=10 ** 9,
                           save_every=10 ** 9, checkpoint_dir=str(Path(d) / 'checkpoints'), share_actor=False,
                           action_dim_per_agent=1, gamma=1.0 if experiment == 'b4' else 0.99)
                trainer = b3_trainer_class('mlp_a0', 3.0)(env, None, cfg, 'cpu')
                trainer.warmup()
                dones = np.asarray(trainer.replay_buffer.dones[:len(trainer.replay_buffer)]).reshape(-1)
                self.assertEqual(len(dones), 200)
                self.assertEqual(np.flatnonzero(dones).tolist(), expected, experiment)
                self.assertEqual(trainer.agent.gamma, cfg['gamma'])

    def test_b3_terminal_unchanged(self):
        env = self.env('ieee33', experiment='b3')
        self.assertIsNone(env.protocol)
        rows = self.run_day(env, 11, np.random.default_rng(0))
        last = rows[-1][1]
        self.assertTrue(all(s == 0.5 for s in rows[0][1]['soc_before']))
        self.assertFalse(last['terminated'])                                  # B3: time limit, bootstrapped
        self.assertTrue(last['truncated'])
        self.assertAlmostEqual(last['marl_terminal_penalty'],
                               -51.923 * float(np.mean(last['marl_terminal_soc_shortfall'] ** 2)), places=9)
        self.assertNotIn('marl_terminal_energy_value_usd', last)

    def test_training_cli_records_protocol_and_gradient_balance(self):
        from experiments.b3.train import main
        with tempfile.TemporaryDirectory() as d:
            out = main(['--feeder', 'ieee33', '--controller', 'mlp_a0', '--run', '0', '--experiment', 'b4',
                        '--output-root', str(Path(d) / 'b4'), '--time-series', series('ieee33'), '--total-steps', '230',
                        '--warmup-steps', '200', '--buffer-size', '500', '--batch-size', '32', '--eval-every', '1000000000',
                        '--save-every', '1000000000', '--diagnostics-every', '10', '--cpu'])
            m = json.loads((out / 'run_manifest.json').read_text())
            self.assertEqual((m['b3']['experiment'], m['b3']['lambda_terminal'], m['b3']['lambda_feasibility']), ('b4', 0.0, 3.0))
            self.assertEqual(m['effective_training_config']['gamma'], 1.0)
            self.assertEqual(m['b3']['settings_overridden']['gamma'], 1.0)            # recorded as a deviation
            import torch
            ck = torch.load(out / 'checkpoints' / 'maddpg_final.pt', map_location='cpu', weights_only=False)
            self.assertEqual(ck['model_kwargs']['gamma'], 1.0)
            self.assertEqual(m['b3']['experiment_protocol']['name'], 'b4')
            for e in m['environments']:
                self.assertEqual({b['capacity_kwh'] for b in e['battery_constraints']}, {700.0})
            rows = (out / 'training_gradient_balance.csv').read_text().splitlines()
            self.assertGreaterEqual(len(rows), 3)
            import csv
            last = list(csv.DictReader(rows))[-1]
            self.assertEqual(float(last['lambda_feasibility']), 3.0)
            self.assertTrue(np.isfinite([float(last['q_param_grad_norm']), float(last['feasibility_param_grad_norm'])]).all())
            self.assertEqual(len(json.loads(last['gradient_balance_per_agent'])), 5)


if __name__ == '__main__':
    unittest.main()
