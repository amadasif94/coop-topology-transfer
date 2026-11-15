"""Small DOE unit/integration tests; no trained policy or training involved.

Integration fixtures use existing local profiles and an IN-MEMORY global 255 A
limit, so they do not certify the updated 40% production datasets.
Set DOE_BASELINE_SOURCE to an unmodified dess_env.py to check old behavior.
"""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.voltage_doe import apply_voltage_doe
from environments.battery import Battery, BatteryConfig
from environments.dess_env import DESSEnv
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG


def rule(p, v, **kwargs):
    n = len(p)
    args = dict(w=.015, delta_c=.010)
    args.update(kwargs)
    return apply_voltage_doe(p, v, [75]*n, [75]*n, [75]*n, [75]*n, **args)


class Rules(unittest.TestCase):
    def test_pass_through(self):
        np.testing.assert_allclose(rule([-60, 60], [1, 1])[0], [-60, 60])

    def test_prevention(self):
        p, d = rule([-75, 75, 50, -50], [.955, 1.045, .955, 1.045])
        np.testing.assert_allclose(p, [-25, 25, 50, -50])
        np.testing.assert_array_equal(d['doe_preventive_clipped'], [1, 1, 0, 0])

    def test_correction_absolute_and_reversal(self):
        p, d = rule([-30, 30, 60, -60], [.945, 1.055, .945, 1.055])
        np.testing.assert_allclose(p, [37.5, -37.5, 60, -60])
        np.testing.assert_array_equal(d['doe_corrective_intervened'], [1, 1, 0, 0])

    def test_zero_width_and_boundaries(self):
        np.testing.assert_allclose(rule([-75, 75], [.95, 1.05], w=0)[0], [-75, 75])
        np.testing.assert_allclose(rule([-75, 75], [.95, 1.05])[0], [0, 0])

    def test_soc_shortfall_and_efficiency(self):
        b = Battery(BatteryConfig(500, 75, 75, soc_min=.1, soc_max=.9, soc_init=.101), .25)
        ad = b.get_available_discharge_kw()
        self.assertAlmostEqual(ad, 1.9)
        out, diag = apply_voltage_doe([-20], [.94], [75], [75],
            [b.get_available_charge_kw()], [ad], w=.015, delta_c=.01)
        self.assertAlmostEqual(out[0], 1.9)
        self.assertAlmostEqual(diag['doe_corrective_shortfall_kw'][0], 73.1)
        b.soc = .899
        ac = b.get_available_charge_kw()
        self.assertAlmostEqual(ac, 2/.95)
        out, diag = apply_voltage_doe([20], [1.06], [75], [75], [ac], [75], w=.015, delta_c=.01)
        self.assertAlmostEqual(out[0], -ac)
        for soc, voltage in [(.1, .94), (.9, 1.06)]:
            b.soc = soc
            out, _ = apply_voltage_doe([0], [voltage], [75], [75],
                [b.get_available_charge_kw()], [b.get_available_discharge_kw()], w=.015, delta_c=.01)
            self.assertEqual(out[0], 0)

    def test_invalid(self):
        for kwargs in [dict(pf_converged=False), dict(w=-1), dict(delta_c=0)]:
            with self.assertRaises(ValueError):
                rule([0], [1], **kwargs)
        with self.assertRaises(ValueError):
            rule([0], [float('nan')])

    def test_asymmetric_ratings_and_saturation(self):
        out, diag = apply_voltage_doe([20, -20], [1.2, .8],
            [50, 50], [80, 80], [50, 50], [80, 80], w=.015, delta_c=.01)
        np.testing.assert_allclose(out, [-50, 80])
        np.testing.assert_allclose(diag['doe_corrective_required_kw'], [50, 80])


def fixture_config(config):
    c = copy.deepcopy(config)
    net = c['network']
    net['line_current_limit_mode'] = 'global_pu'
    net['global_line_current_limit_pu'] = 255 / (net['s_base_kva'] / (np.sqrt(3)*net['v_base_kv']))
    # These original DOE regression tests explicitly exercise
    # legacy backup and strict PF-failure behavior.
    # Production configurations remain unchanged.
    c["backup"]["backup_mode"] = "legacy_scaling"
    c["doe"]["doe_enabled"] = False
    c["doe"]["pf_failure_mode"] = "strict"
    return c


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [DESSEnv(fixture_config(c), seed=123) for c in (IEEE33_CONFIG, IEEE69_CONFIG)]

    def test_both_feeder_short_rollouts(self):
        for env in self.envs:
            env.doe_enabled = True
            base = env.config['network']['s_base_kva'] / (np.sqrt(3)*env.config['network']['v_base_kv'])
            np.testing.assert_allclose(env.grid.get_line_current_limits_pu()*base, 255)
            obs, _ = env.reset(seed=123)
            for t in range(8):
                pre = obs['agent_obs'][:, 3].copy()
                a = np.full(env.num_agents, -.8 if t % 2 else .8)
                obs, reward, _, _, info = env.step(a)
                np.testing.assert_array_equal(info['pre_action_local_voltage_pu'], pre)
                np.testing.assert_allclose(info['requested_battery_actual_power_kw'], info['doe_power_kw'], atol=1e-8)
                self.assertTrue(np.all(np.isfinite(reward)))
                self.assertEqual(info['final_network_feasible'], env._is_feasible_pf_result({
                    'converged': info['converged'],
                    'voltage_pu': info['final_voltage_pu'],
                    'line_current_pu': info['final_line_current_pu'],
                    'grid_import_kw': info['kpis']['grid_import_kw'],
                    'grid_import_kvar': 0.0,
                    'max_voltage_violation': info['max_voltage_violation'],
                    'max_line_current_violation': info['max_line_current_violation']}))

    def test_invalid_pre_pf_leaves_state(self):
        env = self.envs[0]
        env.doe_enabled = True
        for invalid in ('converged', 'voltage_pu'):
            env.reset(seed=123)
            before = env._snapshot_battery_states()
            if invalid == 'converged':
                env.last_pf_result[invalid] = False
            else:
                env.last_pf_result[invalid][env.dess_buses[0]] = np.nan
            with self.assertRaises(ValueError):
                env.step(np.ones(env.num_agents))
            self.assertEqual(before, env._snapshot_battery_states())
            self.assertEqual(env.current_step, 0)

    def test_backup_receives_doe_and_reports_infeasible_zero(self):
        from unittest.mock import patch
        env = self.envs[0]
        env.doe_enabled = True
        env.reset(seed=123)
        env.last_pf_result['voltage_pu'][env.dess_buses] = .94
        with patch.object(env, '_is_feasible_pf_result', return_value=False), \
             patch.object(env, '_handle_infeasible_action', wraps=env._handle_infeasible_action) as backup:
            _, _, _, _, info = env.step(-np.ones(env.num_agents))
        np.testing.assert_allclose(backup.call_args.kwargs['original_action'], 1)
        self.assertTrue(info['backup_activated'])
        self.assertTrue(info['correction_fallback_to_zero'])
        self.assertFalse(info['zero_fallback_feasible'])
        self.assertFalse(info['final_network_feasible'])
        np.testing.assert_allclose(info['final_battery_power_kw'], 0)

    @unittest.skipUnless(os.environ.get('DOE_BASELINE_SOURCE'),
        'BLOCKED: no authentic pre-DOE dess_env.py is available (expected '
        'misc/before_doe_20261018_213005/environments/dess_env.py). Set DOE_BASELINE_SOURCE '
        'to that unmodified file; a substitute baseline must not be fabricated.')
    def test_disabled_against_original(self):
        spec = importlib.util.spec_from_file_location('original_env', os.environ['DOE_BASELINE_SOURCE'])
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        def equal(a, b):
            if isinstance(a, dict):
                for k in a:
                    if k == "agent_obs":
                        # F5 intentionally removed the final grid_stress feature.
                        self.assertEqual(a[k].shape[:-1], b[k].shape[:-1])
                        self.assertEqual(a[k].shape[-1], b[k].shape[-1] + 1)
                        np.testing.assert_array_equal(
                            a[k][..., :b[k].shape[-1]], b[k]
                        )
                    elif k == "agent_obs_dim":
                        # F5 intentionally removes grid_stress.
                        self.assertEqual(a[k], 11)
                        self.assertEqual(b[k], 10)
                    else:
                        try:
                            equal(a[k], b[k])
                        except AssertionError as exc:
                            raise AssertionError(
                                f"Mismatch under key {k!r}: {exc}"
                            ) from exc
            elif isinstance(a, (list, tuple)):
                for x, y in zip(a, b): equal(x, y)
            else:
                np.testing.assert_equal(a, b)
        for config, env in zip((IEEE33_CONFIG, IEEE69_CONFIG), self.envs):
            old = original.DESSEnv(fixture_config(config), seed=123)
            env.doe_enabled = False
            equal(old.reset(seed=123), env.reset(seed=123))
            rng = np.random.default_rng(42)
            for _ in range(8):
                a = rng.uniform(-1, 1, env.num_agents)
                equal(old.step(a), env.step(a))


if __name__ == '__main__':
    unittest.main()
