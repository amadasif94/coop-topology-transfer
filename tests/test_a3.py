"""A3 (fully independent DTDE) correctness checks: E3 local reward, per-battery terminal SoC, information isolation of
observations / critics / rewards, B3 pre-tanh supervision, checkpoints, and small replica smoke runs.

    python -m unittest tests.test_a3 -v

Real-environment checks are skipped without a processed time series (RIT data or the local audit_tmp/replica copy).
"""
from copy import deepcopy
import csv
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from environments.battery_feasibility import terminal_penalty
from environments.local_reward_env import (OVERRIDE_TOL_KW, local_reward_terms, own_bus_cost_change,
                                           terminal_coefficients)
from models.feasibility_agents import CHECKPOINT_SCHEMA, LOCAL_REWARD_CONTROLLERS, build_model
from training.feasibility_supervision import FORMULATION, actor_networks, pretanh_bounds, pretanh_feasibility_loss
from tests.test_b3 import FakeEnv, batch, make, mixed_bounds, preactivations, saturate, series

PRICE, DT = 40.0, 0.25
REWARD = dict(w_grid=0.65, phi_cost=2.0, phi_voltage_deviation=2.0, infeasible_action_penalty=-20.0)


def first_in(module):
    return next(p for p in module.parameters() if p.dim() == 2).shape[1]


class E3Tariff(unittest.TestCase):
    def test_own_bus_import_export_cases(self):
        unit = PRICE * DT / 1000
        cases = [  # (own-bus net load kW, battery power kW, expected cost change in units of price*dt/1000)
            (100.0, 40.0, -40.0),     # importing bus, discharge offsets own import: saving
            (100.0, 150.0, -100.0),   # saving capped at the own-bus import; the 50 kW exported is uncompensated
            (-50.0, 40.0, 0.0),       # exporting bus: discharge has no value (export uncompensated, as A0's tariff)
            (-50.0, -30.0, 0.0),      # charging from own surplus is free
            (-50.0, -80.0, 30.0),     # charging beyond own surplus imports and costs
            (20.0, -10.0, 10.0),      # charging at an importing bus costs the full energy
            (20.0, 0.0, 0.0)]         # idle battery: no economic signal
        for n, p, expected in cases:
            self.assertAlmostEqual(float(own_bus_cost_change(PRICE, n, p, DT)), expected * unit, places=12, msg=(n, p))

    def test_reward_terms_use_only_own_quantities(self):
        # The function signature admits no feeder import, feeder KPI, other-agent or shared-reward input ...
        self.assertEqual(list(inspect.signature(local_reward_terms).parameters),
                         ['cfg', 'price', 'dt_hours', 'v_ref', 'net_load_kw', 'power_kw', 'physical_kw', 'voltage_pu',
                          'local_term', 'shortfall', 'kappa', 'completed', 'terminal_value_usd'])
        # ... and every term is elementwise: agent i's reward equals the reward computed from agent i's data alone.
        rng = np.random.default_rng(0)
        args = dict(net_load_kw=rng.normal(0, 80, 5), power_kw=rng.uniform(-75, 75, 5), voltage_pu=rng.uniform(.95, 1.05, 5),
                    local_term=rng.normal(0, .01, 5), shortfall=rng.uniform(0, 1, 5), kappa=np.full(5, 10.3846))
        args['physical_kw'] = args['power_kw'] + np.array([0, 5, 0, 0, 1e-9])        # backup moved agent 1 only
        joint = local_reward_terms(REWARD, PRICE, DT, 1.0, completed=True, **args)
        for i in range(5):
            alone = local_reward_terms(REWARD, PRICE, DT, 1.0, completed=True, **{k: v[i:i + 1] for k, v in args.items()})
            for name in joint:
                self.assertAlmostEqual(float(joint[name][i]), float(alone[name][0]), places=12, msg=(name, i))
        np.testing.assert_array_equal(joint['backup'], [0, -20.0, 0, 0, 0])            # own override only (> tol)
        self.assertLess(1e-9, OVERRIDE_TOL_KW)
        np.testing.assert_allclose(joint['voltage'], -0.65 * 2.0 * np.abs(args['voltage_pu'] - 1.0))
        np.testing.assert_allclose(joint['terminal_soc'], -10.3846 * args['shortfall'] ** 2)
        self.assertTrue((local_reward_terms(REWARD, PRICE, DT, 1.0, completed=False, **args)['terminal_soc'] == 0).all())
        # b4: the terminal term is the battery's OWN stored-energy valuation, still elementwise.
        value = rng.normal(0, 3, 5)
        joint = local_reward_terms(REWARD, PRICE, DT, 1.0, completed=True, terminal_value_usd=value, **args)
        np.testing.assert_allclose(joint['terminal_soc'], 0.65 * 2.0 * value)
        for i in range(5):
            alone = local_reward_terms(REWARD, PRICE, DT, 1.0, completed=True, terminal_value_usd=value[i:i + 1],
                                       **{k: v[i:i + 1] for k, v in args.items()})
            self.assertAlmostEqual(float(joint['terminal_soc'][i]), float(alone['terminal_soc'][0]), places=12)


class PerBatteryTerminal(unittest.TestCase):
    def test_coefficients_match_the_b3_team_term(self):
        from configs.ieee33_config import IEEE33_CONFIG
        from configs.ieee69_config import IEEE69_CONFIG
        for config, n, lambda_t, kappa in ((IEEE33_CONFIG, 5, 51.923, 10.3846), (IEEE69_CONFIG, 9, 186.923, 20.769222)):
            d = config['dess']
            env = FakeEnv([0.5] * n, d['capacity_kwh'], d['max_discharge_kw'])
            k = terminal_coefficients(env.batteries, lambda_t)
            np.testing.assert_allclose(k, kappa, rtol=1e-6)
            r = config['reward']                   # kappa_i = w_grid * phi_cost * mean train price * own replacement MWh
            own_mwh = (d['soc_init'] - d['soc_min']) * d['capacity_kwh'] / d['charge_efficiency'] / 1000
            self.assertAlmostEqual(r['w_grid'] * r['phi_cost'] * 37.943768 * own_mwh, kappa, places=3)
            shortfall = np.random.default_rng(n).uniform(0, 1, n)
            # sum of the A3 battery penalties == the B3 team penalty; each battery carries exactly its own component.
            self.assertAlmostEqual(float(-(k * shortfall ** 2).sum()), terminal_penalty(lambda_t, shortfall), places=9)

    def test_heterogeneous_allocation_follows_replacement_energy(self):
        env = FakeEnv([0.5, 0.5], 500.0, 75.0)
        env.batteries[1].config.capacity_kwh = 1500.0
        np.testing.assert_allclose(terminal_coefficients(env.batteries, 40.0), [10.0, 30.0])


class Architecture(unittest.TestCase):
    def test_local_actor_and_local_critic(self):
        model = make('mlp_a3', 3.0)
        self.assertEqual(model.controller, 'b3_mlp_a3')
        self.assertIn('mlp_a3', LOCAL_REWARD_CONTROLLERS)
        for i in range(5):
            self.assertEqual(first_in(model.actor.actors[i]), 7)
            self.assertEqual(first_in(model.critics[i]), 8)               # own 7 local features + own action
        self.assertEqual(len(model.actor_optimizers), 5)
        self.assertFalse(hasattr(model, 'actor_optimizer'))
        with self.assertRaisesRegex(ValueError, 'ONLY'):
            model.critics[0](torch.rand(4, 10), torch.rand(4, 1))           # the shared-KPI features cannot enter

    def test_policy_ignores_feeder_kpi_features(self):
        model = make('mlp_a3', 3.0)
        obs = torch.rand(6, 5, 10)
        changed = obs.clone()
        changed[..., 7:] = torch.rand(6, 5, 3) * 100
        a = model.select_action(dict(agent_obs=obs))
        np.testing.assert_array_equal(a, model.select_action(dict(agent_obs=changed)))


def perturbed(data, agents=None, features=None, seed=9):
    g = torch.Generator().manual_seed(seed)
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
    for key in ('agent_obs', 'next_agent_obs'):
        idx = (slice(None), agents if agents is not None else slice(None), features if features is not None else slice(None))
        out[key][idx] = torch.rand(out[key][idx].shape, generator=g) * 5
    if agents is not None:
        out['actions'][:, agents] = torch.rand(out['actions'][:, agents].shape, generator=g) * 2 - 1
        out['rewards'][:, agents] = torch.randn(out['rewards'][:, agents].shape, generator=g) * 50
    return out


class UpdateIsolation(unittest.TestCase):
    def updated(self, data, clip=None):
        model = make('mlp_a3', 3.0)
        model.grad_clip_norm = clip
        model.update(data)
        return model

    def assert_same(self, ma, mb, agents):
        for i in agents:
            for name, module_a, module_b in (('critic', ma.critics[i], mb.critics[i]), ('actor', ma.actor.actors[i], mb.actor.actors[i]),
                                             ('target critic', ma.target_critics[i], mb.target_critics[i])):
                for pa, pb in zip(module_a.parameters(), module_b.parameters()):
                    torch.testing.assert_close(pa, pb, atol=0, rtol=0, msg=f'{name} {i}')

    def test_shared_kpi_features_never_influence_the_update(self):
        data = batch(5, lower=mixed_bounds(5)[0], upper=mixed_bounds(5)[1])
        for clip in (None, 1.0):
            self.assert_same(self.updated(data, clip), self.updated(perturbed(data, features=slice(7, 10)), clip), range(5))

    def test_other_agents_never_influence_agent_zero(self):
        data = batch(5, lower=mixed_bounds(5)[0], upper=mixed_bounds(5)[1])
        a, b = self.updated(data), self.updated(perturbed(data, agents=slice(1, 5)))
        self.assert_same(a, b, [0])                      # obs, next obs, actions, rewards of agents 1-4 changed
        with self.assertRaises(AssertionError):
            self.assert_same(a, b, [1])                  # (the perturbation is real)

    def test_critic_trains_on_own_reward_column(self):
        data = batch(5)
        data['rewards'][:, 0] = 1000.0
        model = make('mlp_a3', 0.0)
        logs = model.update(data)
        self.assertGreater(logs['critic_loss_agent_0'], 100 * max(logs[f'critic_loss_agent_{i}'] for i in range(1, 5)))
        bad = dict(data, rewards=data['rewards'][:, :1])
        with self.assertRaisesRegex(ValueError, 'one individual reward per agent'):
            make('mlp_a3', 0.0).update(bad)


class PreTanhSupervision(unittest.TestCase):
    def test_combined_objective_equivalence(self):
        n, coef = 5, 3.0
        for magnitude in (10.0, 0.5):                                      # strongly saturated / moderate
            model, probe = make('mlp_a3', coef, n), make('mlp_a3', coef, n)
            saturate(model, n, magnitude)
            saturate(probe, n, magnitude)
            lower, upper = mixed_bounds(n)
            data = batch(n, lower=lower, upper=upper)
            logs = model.update(data)
            zs = {}
            hooks = [net.register_forward_hook(lambda m, a, out, i=i: zs.__setitem__(i, out))
                     for i, net in enumerate(actor_networks(probe.actor, n))]
            try:
                local = data['agent_obs'][..., :7]
                q_part = torch.stack([-model.critics[i](local[:, i], probe.actor.actors[i](local[:, i])).mean()
                                      for i in range(n)]).mean()
            finally:
                for h in hooks:
                    h.remove()
            z = torch.cat([zs[i].reshape(16, -1) for i in range(n)], 1)
            feas = pretanh_feasibility_loss(z, *pretanh_bounds(lower, upper))
            expected = torch.autograd.grad(q_part + coef * feas, list(probe.actor.parameters()), allow_unused=True)
            for actual, e in zip(model.actor.parameters(), expected):
                torch.testing.assert_close(actual.grad, e, atol=1e-6, rtol=1e-4)
            self.assertEqual(logs['feasibility_formulation'], FORMULATION)
            self.assertAlmostEqual(logs['feasibility_loss'], float(feas), places=4)

    def test_hooks_capture_genuine_preactivation(self):
        model = make('mlp_a3', 1.0)
        z, a = preactivations(model, batch(5))
        self.assertTrue(torch.equal(torch.tanh(z), a))

    def test_gradient_balance_diagnostics(self):
        for controller in ('mlp_a3', 'mlp_a0'):
            model = make(controller, 3.0)
            model.capture = {}                                           # detailed diagnostics on this update
            saturate(model, 5, 2.0)
            logs = model.update(batch(5, lower=mixed_bounds(5)[0], upper=mixed_bounds(5)[1]))
            q, f = logs['q_param_grad_norm_per_agent'], logs['feasibility_param_grad_norm_per_agent']
            self.assertEqual((len(q), len(f), len(logs['gradient_balance_per_agent'])), (5, 5, 5))
            self.assertAlmostEqual(float(np.sqrt(np.sum(np.square(q)))), logs['q_param_grad_norm'], places=6)
            self.assertAlmostEqual(float(np.sqrt(np.sum(np.square(f)))), logs['feasibility_param_grad_norm'], places=6)
            self.assertAlmostEqual(logs['gradient_balance'], logs['q_param_grad_norm'] / logs['feasibility_param_grad_norm'],
                                   places=6)


class Checkpoints(unittest.TestCase):
    def test_round_trip_and_controller_identity(self):
        model = make('mlp_a3', 3.0, n=9)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'a3.pt'
            model.save(path)
            fresh = make('mlp_a3', 0.5, n=9, seed=4)
            fresh.load(path)
            self.assertEqual((fresh.checkpoint_schema, fresh.lambda_feasibility), (CHECKPOINT_SCHEMA, 3.0))
            obs = torch.rand(3, 9, 10)
            torch.testing.assert_close(fresh._actor_forward(fresh.actor, dict(agent_obs=obs)),
                                       model._actor_forward(model.actor, dict(agent_obs=obs)))
            with self.assertRaisesRegex(ValueError, 'checkpoint controller'):
                make('mlp_a2', 3.0, n=9).load(path)


@unittest.skipUnless(series('ieee33') and series('ieee69'), 'no processed time series')
class RealEnvironment(unittest.TestCase):
    def envs(self, feeder, lambda_t):
        from environments.feasibility_env import make_b3_env
        from environments.local_reward_env import make_a3_env
        from utility.experiment_setup import feeder_config
        cfg = feeder_config(feeder, series(feeder))
        return make_a3_env(deepcopy(cfg), 'val', 0, lambda_t), make_b3_env(deepcopy(cfg), 'val', 0, lambda_t)

    def test_same_physics_and_local_reward_both_feeders(self):
        from utility.evaluation_reporting import reset_for_evaluation
        for feeder, lambda_t in (('ieee33', 51.923), ('ieee69', 186.923)):
            a3, b3 = self.envs(feeder, lambda_t)
            start = int(a3.episode_start_indices[0])
            reset_for_evaluation(a3, start, 0)
            reset_for_evaluation(b3, start, 0)
            rng = np.random.default_rng(1)
            buses = list(a3.dess_buses)
            overridden = 0
            for step in range(96):
                load, _, pv, price, _ = a3._get_profiles(a3.current_index)
                action = rng.uniform(-1, 1, a3.num_agents).astype(np.float32)
                oa, ra, _, ta, ia = a3.step(action)
                ob, rb, _, tb, ib = b3.step(action)
                for key in ('agent_obs', 'x'):
                    np.testing.assert_array_equal(oa[key], ob[key])                        # identical physics/observations
                for key in ('final_battery_power_kw', 'physical_limited_power_kw', 'soc_after', 'final_voltage_pu'):
                    np.testing.assert_array_equal(ia[key], ib[key])
                np.testing.assert_allclose(ia['marl_a0_team_reward'], rb)                  # team reward kept for reference
                np.testing.assert_allclose(ra, ia['reward_terms_total'], rtol=1e-6, atol=1e-5)
                p = np.asarray(ia['final_battery_power_kw'])
                expected = -0.65 * 2.0 * price * 0.25 / 1000 * (np.maximum(0, load[buses] - pv[buses] - p)
                                                                 - np.maximum(0, load[buses] - pv[buses]))
                np.testing.assert_allclose(ia['reward_term_a3_economic'], expected, rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(ia['reward_term_a3_voltage'],
                                           -0.65 * 2.0 * np.abs(np.asarray(ib['final_voltage_pu'])[buses] - 1.0))
                np.testing.assert_allclose(ia['reward_term_a3_local'], ib['reward_term_local'])
                moved = np.abs(p - np.asarray(ia['physical_limited_power_kw'])) > OVERRIDE_TOL_KW
                overridden += int(moved.sum())
                np.testing.assert_array_equal(ia['reward_term_a3_backup'], np.where(moved, -20.0, 0.0))
                self.assertFalse(any(k.startswith('reward_term_') and not k.startswith('reward_term_a3_') for k in ia))
                if step < 95:
                    self.assertTrue((ia['reward_term_a3_terminal_soc'] == 0).all())
            self.assertTrue(ta and ia['marl_terminal_completed'])
            np.testing.assert_allclose(ia['reward_term_a3_terminal_soc'],
                                       -a3.terminal_coefficients * ia['marl_terminal_soc_shortfall'] ** 2)
            np.testing.assert_allclose(a3.terminal_coefficients, lambda_t / a3.num_agents)
            print(f'{feeder}: battery-steps moved by backup under uniform random actions: {overridden}')

    def test_trainer_smoke_and_evaluation_tables(self):
        from environments.local_reward_env import make_a3_env
        from experiments.b3.evaluate import run_episodes, write_tables
        from training.feasibility_trainer import BALANCE_FIELDS, b3_trainer_class
        from utility.experiment_setup import feeder_config
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            env = make_a3_env(feeder_config('ieee33', series('ieee33')), 'train', 42, lambda_terminal=51.923)
            cfg = dict(total_steps=230, warmup_steps=200, buffer_size=500, batch_size=32, eval_every=10 ** 9,
                       save_every=10 ** 9, checkpoint_dir=str(Path(d) / 'checkpoints'), share_actor=False,
                       action_dim_per_agent=1, diagnostics_every=10, gradient_balance_log=True)
            trainer = b3_trainer_class('mlp_a3', 3.0)(env, None, cfg, 'cpu')
            obs = trainer.warmup()
            while trainer.global_step < 230:
                obs, *_ = trainer.collect_step(obs)
                trainer.agent.update(trainer.replay_buffer.sample(32))
            rewards = trainer.replay_buffer.rewards[:len(trainer.replay_buffer)]
            self.assertGreater(float((rewards[:, 0] - rewards[:, 1]).abs().max()), 0)   # individual rewards stored
            with (Path(d) / 'training_gradient_balance.csv').open() as f:
                balance = list(csv.DictReader(f))
            self.assertEqual(list(balance[0]), list(BALANCE_FIELDS))
            self.assertEqual(len(balance), 3)
            self.assertEqual(len(json.loads(balance[-1]['gradient_balance_per_agent'])), 5)
            with (Path(d) / 'training_days.csv').open() as f:
                days = list(csv.DictReader(f))
            self.assertTrue(all(r['completed_day'] == 'True' for r in days[:2]))
            # Frozen-policy evaluation exactly as for A0/A1/A2: the B3 environment (system-level B3 reward used for
            # selection), through the unchanged B3 tables (one validation day).
            from environments.feasibility_env import make_b3_env
            val = make_b3_env(feeder_config('ieee33', series('ieee33')), 'val', 42, lambda_terminal=51.923)
            frame = run_episodes('ieee33', val, trainer.agent, 'maddpg', 'TP1', [(int(val.episode_start_indices[0]), 42)])
            out = Path(d) / 'eval'
            out.mkdir()
            from unittest.mock import patch
            from configs.ieee33_config import IEEE33_CONFIG
            # runner.energy_rows reads day prices from the configured series path (RIT data); point it at the local copy.
            with patch.dict(IEEE33_CONFIG['paths'], processed_time_series=Path(series('ieee33'))):
                ep = write_tables('ieee33', out, [frame], {'TP1': val}, 'b3_mlp_a3', dict(run_id=0))
            steps = pd.read_csv(out / 'step_metrics.csv')
            for column in ('grid_import_kw', 'energy_cost', 'final_network_feasible', 'backup_activated',
                           'policy_requested_power_kw'):
                self.assertIn(column, steps.columns)
            self.assertFalse(any(c.startswith('reward_term_a3_') for c in steps.columns))   # B3 reward, not A3's
            for column in ('total_energy_cost', 'adjusted_energy_cost', 'raw_energy_cost', 'terminal_stored_energy_kwh'):
                self.assertTrue(np.isfinite(ep[column]).all(), column)
            self.assertEqual(len(steps), 96)

    def test_selection_and_test_use_the_common_b3_environment(self):
        """Selection metric for A3 = the same system-level B3 reward as A0/A1/A2 (no controller-specific env)."""
        import experiments.b3.evaluate as legacy
        import experiments.b3_reporting.evaluate as reporting
        for module in (legacy, reporting):
            self.assertNotIn('make_a3_env', inspect.getsource(module))
        self.assertIn('mlp_a3', reporting.CONTROLLERS)


if __name__ == '__main__':
    unittest.main()
