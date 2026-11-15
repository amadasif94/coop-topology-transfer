"""Penalty rule: physical limits unpenalized but logged; optional DOE term on the
DOE's own change; every directional backup activation penalized."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.primary_marl import make_primary_env
from utility.evaluation_reporting import diagnostic_summary


class PenaltyRule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [make_primary_env(copy.deepcopy(c), mode="train", seed=123)
                    for c in (IEEE33_CONFIG, IEEE69_CONFIG)]

    def test_defaults(self):
        for env in self.envs:
            self.assertEqual(env.doe_penalty_coef, 0.0)

    def test_physical_clipping_logged_not_penalized(self):
        for env in self.envs:
            env.doe_penalty_coef = 0.0
            env.reset(seed=123)
            env.batteries[0].soc = env.batteries[0].config.soc_min
            a = np.zeros(env.num_agents, np.float32)
            a[0] = 1.0
            _, r, _, _, info = env.step(a)
            self.assertTrue(info["physical_clipped"][0])
            self.assertFalse(info["physical_clipped"][1:].any())
            self.assertAlmostEqual(info["physical_clipped_kw"][0], env.batteries[0].config.max_discharge_kw, places=6)
            self.assertTrue(info["physical_clipping_any"])
            self.assertAlmostEqual(info["physical_clipped_kw_total"], info["physical_clipped_kw"][0], places=6)
            self.assertFalse(info["major_infeasible_action"])
            np.testing.assert_array_equal(info["doe_penalty_amount"], 0.0)
            # Within limits: nothing clipped.
            env.reset(seed=123)
            _, _, _, _, info = env.step(np.full(env.num_agents, 0.2, np.float32))
            self.assertFalse(info["physical_clipping_any"])
            self.assertAlmostEqual(info["physical_clipped_kw_total"], 0.0, delta=1e-5)

    def _doe_step(self, env, coef):
        env.doe_penalty_coef = coef
        env.reset(seed=123)
        # Preventive band: V = 0.955 -> charge limit = Pch * (0.005 / 0.015).
        env.last_pf_result["voltage_pu"][env.dess_buses[0]] = 0.955
        a = np.zeros(env.num_agents, np.float32)
        a[0] = -1.0
        return env.step(a)

    def test_doe_penalty_measured_after_physical_limits(self):
        for env in self.envs:
            try:
                _, r0, _, _, i0 = self._doe_step(env, 0.0)
                _, r2, _, _, i2 = self._doe_step(env, 2.0)
            finally:
                env.doe_penalty_coef = 0.0
            rated = max(env.batteries[0].config.max_charge_kw, env.batteries[0].config.max_discharge_kw)
            expected_change = abs(i2["physical_limited_power_kw"][0] - i2["doe_power_kw"][0])
            self.assertGreater(expected_change, 1.0)
            self.assertAlmostEqual(i2["doe_change_kw"][0], expected_change, places=6)
            self.assertAlmostEqual(i2["doe_change_normalized"][0], expected_change / rated, places=6)
            np.testing.assert_array_equal(i2["doe_change_kw"][1:], 0.0)
            np.testing.assert_array_equal(i0["doe_penalty_amount"], 0.0)
            self.assertEqual(i0["backup_activated"], i2["backup_activated"])
            delta = np.asarray(r2, float) - np.asarray(r0, float)
            np.testing.assert_allclose(delta[0], -2.0 * expected_change / rated, atol=1e-4)
            np.testing.assert_allclose(delta[1:], 0.0, atol=1e-4)

    def test_summary_rates(self):
        df = pd.DataFrame(dict(physical_clipping_any=[True, False, False, False],
                               doe_intervened=[True, True, False, False],
                               backup_activated=[False, False, False, True]))
        out = diagnostic_summary(df)
        self.assertEqual(out["physical_clipping_rate"], 0.25)
        self.assertEqual(out["doe_intervention_rate"], 0.5)
        self.assertEqual(out["backup_activation_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
