"""SOC-feasible action mapping (protocol soc_feasible): mapping, action semantics, replay, supervision off,
checkpoint convention. Maintained modules only (environments.marl_env, configs.experiments, models.controllers,
training.controller_trainer, evaluation.evaluate). Real-environment tests use data/time_series (RIT) or the local audit_tmp/replica copy and only
validation days. Run: python -m unittest tests.test_soc_feasible_mapping -v
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from environments.battery import Battery, BatteryConfig
from environments.battery_feasibility import soc_feasible_action
from tests.support import HAS_PYG, batch, series

DT = 0.25
CONTROLLERS = ('mlp_a0', 'mlp_a1', 'mlp_a2', 'mlp_a3') + (('gcn', 'gat', 'tagconv') if HAS_PYG else ())


def battery(soc, capacity=850.0, charge_kw=100.0, discharge_kw=180.0, eta_ch=0.95, eta_dis=0.92):
    b = Battery(BatteryConfig(capacity, charge_kw, discharge_kw, charge_efficiency=eta_ch, discharge_efficiency=eta_dis,
                              soc_min=0.10, soc_max=0.90, soc_init=0.50), DT)
    b.soc = soc
    return b


def executed(b, a_prime):
    """Battery.step on a copy: (executed kW, requested kW = a' * rating, post-step SoC)."""
    out = deepcopy(b).step(float(a_prime))
    return out['power_kw'], out['requested_power_kw'], out['soc']


def affine(a, b):
    """Expected affine SOC-feasible power (kW) from the battery's own available powers."""
    ch, dis = b.get_available_charge_kw(), b.get_available_discharge_kw()
    return -ch + (a + 1) / 2 * (ch + dis)


class Mapping(unittest.TestCase):
    """Affine SOC-feasible mapping P = -P_ch_avail + (a+1)/2 (P_ch_avail + P_dis_avail) (version affine_v2)."""

    def check(self, b, cases):
        for a, expected in cases:
            a_prime = soc_feasible_action([a], [b])[0]
            p, requested, _ = executed(b, a_prime)
            self.assertAlmostEqual(p, expected, delta=1e-4, msg=(b.soc, a))
            self.assertLessEqual(abs(requested - p), 1e-9)                    # limiter has nothing to remove

    def test_version(self):
        from environments.battery_feasibility import SOC_FEASIBLE_MAPPING_VERSION
        self.assertEqual(SOC_FEASIBLE_MAPPING_VERSION, 'affine_v2')

    def test_minimum_soc_10_percent(self):
        b = battery(0.10)                                              # charge 100 kW available, discharge 0
        self.assertEqual((b.get_available_charge_kw(), b.get_available_discharge_kw()), (100.0, 0.0))
        self.check(b, ((-1.0, -100.0), (0.0, -50.0), (1.0, 0.0)))     # max charge / midpoint / zero discharge
        self.assertEqual(soc_feasible_action([1.0], [b])[0], 0.0)

    def test_maximum_soc_90_percent(self):
        b = battery(0.90)                                              # charge 0, discharge 180 kW available
        self.assertEqual((b.get_available_charge_kw(), b.get_available_discharge_kw()), (0.0, 180.0))
        self.check(b, ((-1.0, 0.0), (0.0, 90.0), (1.0, 180.0)))       # zero charge / midpoint / max discharge
        self.assertEqual(soc_feasible_action([-1.0], [b])[0], 0.0)

    def test_intermediate_soc_symmetric_limits(self):
        b = battery(0.50, charge_kw=150.0, discharge_kw=150.0)
        self.check(b, ((-1.0, -150.0), (0.0, 0.0), (1.0, 150.0), (0.4, 60.0), (-0.4, -60.0)))
        self.assertEqual(soc_feasible_action([0.0], [b])[0], 0.0)      # exactly 0 kW at a = 0 when symmetric

    def test_asymmetric_limits_and_normalized_conversion(self):
        """Charge 100 kW / discharge 180 kW: a = 0 is the midpoint (+40 kW); a' uses the rating of P's direction."""
        b = battery(0.5)
        self.assertEqual(executed(b, 0.5)[1], 0.5 * 180.0)             # Battery.step: a > 0 -> a * max_discharge_kw
        self.assertEqual(executed(b, -0.5)[1], -0.5 * 100.0)           # a < 0 -> a * max_charge_kw
        self.check(b, ((-1.0, -100.0), (0.0, 40.0), (1.0, 180.0), (-0.5, -30.0), (0.5, 110.0)))
        for soc in (0.11, 0.12, 0.5, 0.88, 0.895):
            b = battery(soc)
            for a in (-0.8, -0.2, 0.0, 0.3, 0.8):
                p = affine(a, b)
                a_prime = float(soc_feasible_action([a], [b])[0])
                expected = p / (180.0 if p > 0 else 100.0) if p != 0 else 0.0
                self.assertAlmostEqual(a_prime, expected, places=6, msg=(soc, a))
                self.assertAlmostEqual(executed(b, a_prime)[0], p, delta=1e-4)
        self.assertAlmostEqual(battery(0.12).get_available_discharge_kw(), 0.02 * 850 * 0.92 / DT)   # energy-limited

    def test_monotone_and_continuous(self):
        for soc in (0.10, 0.3, 0.9):
            b = battery(soc)
            grid = np.linspace(-1, 1, 201)
            powers = [executed(b, soc_feasible_action([a], [b])[0])[0] for a in grid]
            self.assertTrue(np.all(np.diff(powers) >= -1e-4), soc)                     # no dead zone, no reversal
            self.assertLess(np.max(np.abs(np.diff(powers))), (b.get_available_charge_kw()
                                                             + b.get_available_discharge_kw()) / 100 + 1e-3)

    def test_efficiency_and_15_minute_step(self):
        b = battery(0.12)                                              # energy-limited discharge
        p, _, soc = executed(b, soc_feasible_action([1.0], [b])[0])
        self.assertAlmostEqual(soc, 0.10, places=6)                    # drains exactly to SoC_min
        self.assertAlmostEqual(p, 0.02 * 850 * 0.92 / DT, delta=1e-3)  # eta_dis and dt = 0.25 h
        b = battery(0.88)
        p, _, soc = executed(b, soc_feasible_action([-1.0], [b])[0])
        self.assertAlmostEqual(soc, 0.90, places=6)
        self.assertAlmostEqual(-p, 0.02 * 850 / 0.95 / DT, delta=1e-3)  # eta_ch

    def test_new_array_float32_and_bounded(self):
        bs = [battery(0.2), battery(0.85), battery(0.5)]
        a = np.array([0.9, -0.9, 0.0], dtype=np.float32)
        keep = a.copy()
        a_prime = soc_feasible_action(a, bs)
        np.testing.assert_array_equal(a, keep)
        self.assertIsNot(a_prime, a)
        self.assertEqual(a_prime.dtype, np.float32)
        self.assertTrue((np.abs(a_prime) <= 1.0).all())
        with self.assertRaises(ValueError):
            soc_feasible_action([np.nan, 0.0, 0.0], bs)

    def test_randomized_float32_safety_no_soc_clipping(self):
        rng = np.random.default_rng(0)
        worst = 0.0
        for k in range(3000):
            b = battery(rng.uniform(0.10, 0.90), capacity=rng.uniform(200, 1500), charge_kw=rng.uniform(40, 300),
                        discharge_kw=rng.uniform(40, 300), eta_ch=rng.uniform(0.85, 1.0), eta_dis=rng.uniform(0.85, 1.0))
            if k % 10 == 0:
                b.soc = float(rng.choice([0.10, 0.90]))               # exactly at a limit
            a = float(rng.choice([-1.0, 0.0, 1.0])) if k % 7 == 0 else float(rng.uniform(-1, 1))
            a_prime = float(soc_feasible_action([a], [b])[0])
            available = b.get_available_discharge_kw() if a_prime >= 0 else b.get_available_charge_kw()
            rated = b.config.max_discharge_kw if a_prime >= 0 else b.config.max_charge_kw
            p, requested, _ = executed(b, a_prime)
            self.assertLessEqual(abs(a_prime) * rated, available + 1e-12)   # float32 request never exceeds P_avail
            self.assertEqual(requested, p)                                   # exactly no limiter correction
            worst = max(worst, abs(p - affine(a, b)))
        self.assertLess(worst, 1e-3)

    def test_physical_limiter_still_active_without_mapping(self):
        b = battery(0.11)
        p, requested, _ = executed(b, 1.0)                             # rated request, energy-limited battery
        self.assertEqual(requested, 180.0)
        self.assertLess(p, requested - 100.0)


@unittest.skipUnless(series('ieee33') and series('ieee69'), 'no processed time series')
class Environment(unittest.TestCase):
    def env(self, feeder, experiment='soc_feasible', a3=False):
        from configs.experiments import feeder_config
        from environments.local_reward_env import make_a3_env
        from environments.marl_env import make_marl_env
        env = (make_a3_env if a3 else make_marl_env)(feeder_config(feeder, series(feeder), experiment), 'val', 42, 0.0)
        env.reset(seed=1)
        return env

    @staticmethod
    def state(env):
        socs = [0.895, 0.105] + [0.5] * (env.num_agents - 2)
        for b, s in zip(env.batteries, socs):
            b.soc = s
        return np.array([-1.0, 1.0, 0.6, -0.3] + [0.0] * (env.num_agents - 4), dtype=np.float32)

    def test_executed_mapping_and_action_quantities(self):
        for feeder, n in (('ieee33', 5), ('ieee69', 9)):
            for a3 in (False, True):
                env = self.env(feeder, a3=a3)
                self.assertEqual((env.num_agents, env.action_mapping), (n, 'soc_feasible'))
                a = self.state(env)
                expected_prime = soc_feasible_action(a, env.batteries)
                ch = np.array([b.get_available_charge_kw() for b in env.batteries])
                dis = np.array([b.get_available_discharge_kw() for b in env.batteries])
                expected_kw = -ch + (a + 1) / 2 * (ch + dis)                             # affine mapping
                sent = a.copy()
                _, _, _, _, info = env.step(sent)
                self.assertEqual(info['marl_action_mapping_version'], 'affine_v2')
                np.testing.assert_allclose(info['marl_available_charge_kw'], ch)
                np.testing.assert_allclose(info['marl_available_discharge_kw'], dis)
                self.assertEqual(env.action_mapping_version, 'affine_v2')
                np.testing.assert_array_equal(sent, a)                                  # caller's array untouched
                np.testing.assert_allclose(info['marl_action_policy_normalized'], a)
                np.testing.assert_allclose(info['marl_action_execution_normalized'], expected_prime)
                np.testing.assert_allclose(info['marl_power_soc_mapped_kw'], expected_kw, atol=1e-4)
                np.testing.assert_allclose(info['marl_requested_power_kw'], info['marl_power_soc_mapped_kw'])
                self.assertLessEqual(float(np.max(info['marl_soc_limiter_correction_kw'])), 1e-6)
                self.assertLessEqual(float(np.max(info['physical_clipped_kw'])), 1e-6)
                np.testing.assert_allclose(info['marl_backup_correction_kw'],
                                           np.abs(info['physical_limited_power_kw'] - info['final_battery_power_kw']))
                np.testing.assert_array_equal(info['marl_feasible_action_lower'], -1.0)
                np.testing.assert_array_equal(info['marl_feasible_action_upper'], 1.0)
                self.assertEqual(info['marl_action_mapping'], 'soc_feasible')
                self.assertFalse(np.allclose(expected_prime[:2], a[:2]))                   # a != a' at the bounds

    def test_rated_mapping_is_not_used_and_still_works_for_b4(self):
        for feeder in ('ieee33', 'ieee69'):
            env = self.env(feeder, experiment='b4')
            self.assertEqual(env.action_mapping, 'rated')
            a = self.state(env)
            _, _, _, _, info = env.step(a)
            ratings = np.array([b.config.max_discharge_kw if x >= 0 else b.config.max_charge_kw
                                for x, b in zip(a, env.batteries)])
            np.testing.assert_allclose(info['marl_requested_power_kw'], a * ratings, atol=1e-4)  # old a * P_rated
            self.assertGreater(float(info['marl_soc_limiter_correction_kw'][0]), 1.0)          # limiter corrects it
            np.testing.assert_allclose(info['marl_action_execution_normalized'], a)

    def test_zero_action_reference_executes_exactly_zero_kw(self):
        """The zero-action reference never uses the affine mapping (where a = 0 is the midpoint): 0 kW at every SoC."""
        from configs.experiments import feeder_config
        from evaluation.zero import make_zero_action_env
        for feeder in ('ieee33', 'ieee69'):
            env = make_zero_action_env(feeder, feeder_config(feeder, series(feeder), 'soc_feasible'), 'val', 42)
            env.reset(seed=1)
            for b, s in zip(env.batteries, [0.10, 0.90] + [0.5] * (env.num_agents - 2)):
                b.soc = s
            _, _, _, _, info = env.step(np.zeros(env.num_agents, dtype=np.float32))
            for key in ('physical_limited_power_kw', 'final_battery_power_kw'):
                np.testing.assert_array_equal(np.asarray(info[key], dtype=float), 0.0)
        # By contrast the soc_feasible MARL environment maps a = 0 to the midpoint (charging at s_min, discharging at s_max).
        marl = self.env('ieee33')
        a = self.state(marl)
        a[:] = 0.0
        _, _, _, _, info = marl.step(a)
        p = np.asarray(info['marl_power_soc_mapped_kw'], dtype=float)
        self.assertLess(p[1], -1.0)                       # SoC 0.105 (near min): midpoint is a charge
        self.assertGreater(p[0], 1.0)                     # SoC 0.895 (near max): midpoint is a discharge

    def test_episode_diagnostics_under_affine_mapping(self):
        """One validation day through the evaluator: SoC-boundary, low-SoC and mapping diagnostics are produced, and a
        policy that always outputs a = +1 never produces a kW discharge request at s_min (no dead zone, no violation)."""
        from evaluation.evaluate import run_episodes
        from utility.feasibility_metrics import episode_diagnostics
        env = self.env('ieee33')
        for b in env.batteries:
            b.config.soc_init = 0.105                                         # start the day near s_min
        agent = SimpleNamespace(select_action=lambda obs, noise_std=0.0: np.full(env.num_agents, 1.0, np.float32))
        start = int(env.episode_start_indices[0])
        frame = run_episodes('ieee33', env, agent, 'maddpg', 'TP1', [(start, 42)])
        self.assertEqual(len(frame), 96)
        self.assertEqual(set(frame.marl_action_mapping_version), {'affine_v2'})
        for column in ('soc_before', 'soc_after', 'marl_action_policy_normalized', 'marl_action_execution_normalized',
                       'marl_power_soc_mapped_kw', 'marl_power_executed_kw', 'marl_available_charge_kw',
                       'marl_available_discharge_kw', 'marl_backup_correction_kw', 'final_network_feasible'):
            self.assertIn(column, frame.columns)
        d = episode_diagnostics(frame, env).iloc[0]
        self.assertEqual((d['action_mapping'], d['action_mapping_version']), ('soc_feasible', 'affine_v2'))
        for column in ['b3_soc_near_min_fraction', 'b3_soc_near_max_fraction', 'b3_max_soc_boundary_streak',
                       'b3_near_min_step_fraction', 'b3_near_min_positive_raw_action_fraction',
                       'b3_near_min_discharge_request_fraction', 'b3_backup_activation_rate'] + \
                [f'b3_max_soc_boundary_streak_agent_{i}' for i in range(env.num_agents)] + \
                [f'b3_near_min_mean_raw_action_agent_{i}' for i in range(env.num_agents)]:
            self.assertIn(column, d.index)
        self.assertEqual(d['b3_max_soc_boundary_streak'],
                         max(d[f'b3_max_soc_boundary_streak_agent_{i}'] for i in range(env.num_agents)))
        if d['b3_near_min_step_fraction'] > 0:
            self.assertEqual(d['b3_near_min_positive_raw_action_fraction'], 1.0)     # actor says a = +1 ...
            self.assertEqual(d['b3_near_min_discharge_request_fraction'], 0.0)      # ... which is not a kW discharge

    def test_network_backup_still_operates(self):
        env = self.env('ieee33')
        env.v_min = 0.9999                                     # tightened limit: every dispatch violates it
        a = self.state(env)
        _, _, _, _, info = env.step(a)
        self.assertTrue(info['backup_activated'])
        self.assertIn('marl_backup_correction_kw', info)
        self.assertLessEqual(float(np.max(info['marl_soc_limiter_correction_kw'])), 1e-6)

    def test_replay_stores_policy_action_not_execution_action(self):
        from training.controller_trainer import trainer_class
        env = self.env('ieee33')
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(total_steps=10, warmup_steps=5, buffer_size=50, batch_size=4, eval_every=10 ** 9, save_every=10 ** 9,
                       checkpoint_dir=str(Path(d) / 'checkpoints'), share_actor=False, action_dim_per_agent=1)
            trainer = trainer_class('mlp_a0', 0.0, 'soc_feasible', False)(env, None, cfg, 'cpu')
            obs, _ = env.reset(seed=3)
            a = self.state(env)
            trainer.agent.select_action = lambda obs, noise_std=0.0: a.copy()
            size = trainer.replay_buffer.size
            _, _, _, info = trainer.collect_step(obs)
            stored = trainer.replay_buffer.actions[size].cpu().numpy().reshape(-1)
            np.testing.assert_allclose(stored, a)                                   # replay keeps a
            self.assertFalse(np.allclose(info['marl_action_execution_normalized'], a))  # execution used a'
            np.testing.assert_allclose(info['marl_action_policy_normalized'], a)
            with self.assertRaises(ValueError):                                     # mismatched trainer/env mapping
                trainer_class('mlp_a0', 3.0, 'rated', True)(env, None, cfg, 'cpu')


def model_kwargs(controller, n=5):
    kw = dict(agent_obs_dim=10, num_agents=n, action_dim_per_agent=1, actor_lr=5e-5, critic_lr=5e-4, grad_clip_norm=1.0,
              device='cpu', share_actor=False)
    if controller in ('gcn', 'gat', 'tagconv'):
        kw.update(node_feature_dim=4, dess_buses=[1, 3, 5, 7, 9][:n])
    return kw


class SupervisionOff(unittest.TestCase):
    def test_update_is_exactly_the_original_for_every_controller(self):
        from models.controllers import ControllerMixin, build_model
        narrow = (torch.zeros(16, 5), torch.zeros(16, 5))     # zero-width bounds: every z != 0 would be penalised
        for controller in CONTROLLERS:
            with self.subTest(controller=controller):
                torch.manual_seed(0)
                model = build_model(controller, 0.0, action_mapping='soc_feasible', feasibility_supervision=False,
                                    **model_kwargs(controller))
                self.assertEqual((model.feasibility_formulation, model.lambda_feasibility), ('none', 0.0))
                reference = deepcopy(model)
                b = batch(5, lower=narrow[0], upper=narrow[1])
                torch.manual_seed(1)
                logs = model.update(b)
                torch.manual_seed(1)
                super(ControllerMixin, reference).update(b)               # bare original update
                for name in ('actor', 'critics'):
                    for (k, p), (_, q) in zip(getattr(model, name).state_dict().items(),
                                              getattr(reference, name).state_dict().items()):
                        self.assertTrue(torch.equal(p, q), f'{controller} {name}.{k}')
                self.assertEqual(logs['feasibility_formulation'], 'none')
                with self.assertRaises(ValueError):
                    build_model(controller, 3.0, feasibility_supervision=False, **model_kwargs(controller))

    def test_supervision_on_still_changes_the_update(self):
        from models.controllers import ControllerMixin, build_model
        torch.manual_seed(0)
        model = build_model('mlp_a0', 3.0, **model_kwargs('mlp_a0'))
        reference = deepcopy(model)
        b = batch(5, lower=torch.zeros(16, 5), upper=torch.zeros(16, 5))
        torch.manual_seed(1)
        model.update(b)
        torch.manual_seed(1)
        super(ControllerMixin, reference).update(b)
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(model.actor.parameters(), reference.actor.parameters())))


class Checkpoints(unittest.TestCase):
    def test_mapping_convention_enforced_on_load(self):
        from models.controllers import build_model
        with tempfile.TemporaryDirectory() as d:
            new = build_model('mlp_a0', 0.0, action_mapping='soc_feasible', feasibility_supervision=False,
                              **model_kwargs('mlp_a0'))
            old = build_model('mlp_a0', 3.0, **model_kwargs('mlp_a0'))
            new.save(Path(d) / 'new.pt')
            old.save(Path(d) / 'old.pt')
            legacy = torch.load(Path(d) / 'old.pt', weights_only=False)
            for key in ('action_mapping', 'feasibility_supervision'):           # a checkpoint from before the option
                legacy.pop(key)
            torch.save(legacy, Path(d) / 'legacy.pt')
            zero_preserving = torch.load(Path(d) / 'new.pt', weights_only=False)    # a soc_feasible checkpoint from
            zero_preserving.pop('action_mapping_version')                         # before mapping versions existed
            torch.save(zero_preserving, Path(d) / 'zero_preserving.pt')
            soc_env = SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='affine_v2')
            rated_env = SimpleNamespace(action_mapping='rated', action_mapping_version=None)
            old_soc_env = SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='zero_preserving_v1')
            for path, env, pattern in (('old.pt', soc_env, 'action_mapping'), ('legacy.pt', soc_env, 'action_mapping'),
                                       ('new.pt', rated_env, 'action_mapping'),
                                       ('zero_preserving.pt', soc_env, 'action_mapping_version'),
                                       ('new.pt', old_soc_env, 'action_mapping_version')):
                with self.subTest(path=path, env=env), self.assertRaisesRegex(ValueError, pattern):
                    build_model('mlp_a0', 3.0, **model_kwargs('mlp_a0')).load(Path(d) / path, env)
            # Matching mapping and version pass both checks (this fake env then fails only the provenance check).
            with self.assertRaisesRegex(ValueError, 'provenance'):
                build_model('mlp_a0', 0.0, **model_kwargs('mlp_a0')).load(Path(d) / 'new.pt', soc_env)
            with self.assertRaisesRegex(ValueError, 'provenance'):          # rated checkpoints carry no version
                build_model('mlp_a0', 3.0, **model_kwargs('mlp_a0')).load(Path(d) / 'old.pt', rated_env)
            record = torch.load(Path(d) / 'new.pt', weights_only=False)
            self.assertEqual((record['action_mapping'], record['action_mapping_version'], record['feasibility_supervision'],
                              record['feasibility_formulation']), ('soc_feasible', 'affine_v2', False, 'none'))

    def test_old_results_root_is_never_mixed_or_modified(self):
        """A root holding zero-preserving soc_feasible runs (maintained or historical layout) is refused for new
        training, and select/test of such a run stop before writing anything; nothing in the old tree changes."""
        import json
        from configs.experiments import RunLayout, check_root_mapping_version
        from evaluation.evaluate import EvaluationSpec, select, test
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / 'soc_feasible'
            for layout in (RunLayout(old), RunLayout(old, legacy=True)):
                run = layout.run_dir('ieee33', 'mlp_a3', 0) / 'training'
                run.mkdir(parents=True)
                (run / 'run_manifest.json').write_text(json.dumps({
                    'status': 'finished', layout.block: dict(experiment='soc_feasible', action_mapping='soc_feasible')}))
            before = sorted((p.relative_to(old), p.stat().st_mtime_ns) for p in old.rglob('*'))
            with self.assertRaisesRegex(ValueError, 'action_mapping_version'):
                check_root_mapping_version(old, 'ieee33', 'soc_feasible')
            fresh = Path(d) / 'soc_feasible_affine_v2'
            self.assertEqual(check_root_mapping_version(fresh, 'ieee33', 'soc_feasible'), 0)
            for legacy in (False, True):
                spec = EvaluationSpec('ieee33', 'mlp_a3', 0, 'soc_feasible', RunLayout(old, legacy=legacy), cpu=True)
                with self.assertRaisesRegex(SystemExit, 'zero_preserving_v1'):
                    select(spec)
                with self.assertRaisesRegex(SystemExit, 'zero_preserving_v1'):
                    test(spec, [1.0])
            self.assertEqual(sorted((p.relative_to(old), p.stat().st_mtime_ns) for p in old.rglob('*')), before)
            from training.train import main as train
            with self.assertRaisesRegex(SystemExit, 'action_mapping_version'):
                train(['--feeder', 'ieee33', '--controller', 'mlp', '--variant', 'A3', '--seed', '1',
                       '--output-root', str(old), '--cpu'])
            self.assertEqual(sorted((p.relative_to(old), p.stat().st_mtime_ns) for p in old.rglob('*')), before)

    def test_verified_mapping_across_manifest_env_checkpoint(self):
        from evaluation.evaluate import verified_mapping
        new = dict(action_mapping='soc_feasible', action_mapping_version='affine_v2')
        for training in (dict(run_record=new), dict(b3=new)):                         # maintained / historical manifest
            self._verified(verified_mapping, training)
        # A soc_feasible run without a version (zero-preserving, before 2026-11-07) is refused in an affine environment.
        env = SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='affine_v2')
        agent = SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='affine_v2', feasibility_supervision=False)
        with self.assertRaisesRegex(SystemExit, 'action_mapping_version'):
            verified_mapping(dict(run_record=dict(action_mapping='soc_feasible')), env, agent)
        old_agent = SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='zero_preserving_v1')
        with self.assertRaisesRegex(SystemExit, 'action_mapping_version'):
            verified_mapping(dict(run_record=new), env, old_agent)

    def _verified(self, verified_mapping, training):
        env = SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='affine_v2')
        ok = verified_mapping(training, env, SimpleNamespace(action_mapping='soc_feasible', action_mapping_version='affine_v2',
                                                             feasibility_supervision=False))
        self.assertEqual(ok, dict(action_mapping='soc_feasible', action_mapping_version='affine_v2',
                                  feasibility_supervision=False))
        with self.assertRaises(SystemExit):
            verified_mapping(training, env, SimpleNamespace(action_mapping='rated'))
        with self.assertRaises(SystemExit):
            verified_mapping({}, env, SimpleNamespace(action_mapping='soc_feasible',
                                                      action_mapping_version='affine_v2'))   # old manifest = rated


if __name__ == '__main__':
    unittest.main()
