"""F10: DOE configuration validation. Production settings must still be accepted unchanged."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from environments.primary_marl import make_primary_env
from environments.voltage_doe import validate_parameters, validate_doe_config, apply_voltage_doe

PROD = dict(doe_enabled=True, doe_w=0.015, doe_delta_c=0.010, pf_failure_mode="logged_fallback",
            doe_penalty_coef=0.0)


class DOEValidation(unittest.TestCase):
    def test_production_accepted(self):
        validate_doe_config(PROD, 0.95, 1.05)
        for cfg in (IEEE33_CONFIG, IEEE69_CONFIG):
            env = make_primary_env(copy.deepcopy(cfg), mode="val", seed=1)
            self.assertEqual((env.doe_enabled, env.doe_w, env.doe_delta_c), (True, 0.015, 0.010))

    def test_preventive_width_bound(self):
        validate_parameters(0.0, 0.01, 0.95, 1.05)
        validate_parameters(0.0499, 0.01, 0.95, 1.05)
        for w in (0.05, 0.08, 1.0):
            with self.assertRaisesRegex(ValueError, "must be <"):
                validate_parameters(w, 0.01, 0.95, 1.05)
        for w, dc in ((-0.001, 0.01), (0.015, 0.0), (0.015, -1.0), (np.nan, 0.01), (0.015, np.inf)):
            with self.assertRaises(ValueError):
                validate_parameters(w, dc, 0.95, 1.05)
        with self.assertRaises(ValueError):
            validate_parameters(0.015, 0.01, 1.05, 0.95)
        # delta_c has no upper bound by design.
        validate_parameters(0.015, 10.0, 0.95, 1.05)

    def test_bad_config_values_rejected(self):
        bad = [dict(PROD, doe_enabled="False"), dict(PROD, doe_enabled=1), dict(PROD, doe_w="0.015"),
               dict(PROD, doe_delta_c=True), dict(PROD, doe_penalty_coef=float("nan")),
               dict(PROD, pf_failure_mode="ignore"), dict(PROD, doe_W=0.015),
               dict(PROD, doe_enabled=False, doe_penalty_coef=0.5), dict(PROD, doe_w=0.06)]
        for cfg in bad:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_doe_config(cfg, 0.95, 1.05)
        validate_doe_config(dict(PROD, doe_enabled=False), 0.95, 1.05)   # DOE off, valid values
        validate_doe_config({}, 0.95, 1.05)                               # library defaults
        validate_doe_config(dict(PROD, doe_enabled=np.bool_(True)), 0.95, 1.05)

    def test_environment_rejects_bad_config(self):
        cfg = copy.deepcopy(IEEE33_CONFIG)
        cfg["doe"]["doe_enabled"] = "False"
        with self.assertRaisesRegex(ValueError, "doe_enabled must be a boolean"):
            DESSEnv(cfg, mode="val", seed=1)
        cfg = copy.deepcopy(IEEE33_CONFIG)
        cfg["doe"]["doe_w"] = 0.05
        cfg["doe"]["doe_enabled"] = False      # validated even when DOE is off
        with self.assertRaises(ValueError):
            DESSEnv(cfg, mode="val", seed=1)

    def test_doe_algorithm_unchanged(self):
        out, diag = apply_voltage_doe([-75, 75, 0], [0.9575, 1.0425, 0.945], [75, 75, 75], [75, 75, 75],
                                      [75, 75, 75], [75, 75, 75], w=0.015, delta_c=0.010)
        np.testing.assert_allclose(out, [-37.5, 37.5, 37.5])


if __name__ == "__main__":
    unittest.main()
