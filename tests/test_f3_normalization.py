"""F3: explicit PV (per bus) and feeder-curtailment normalization references; physics unchanged.

The invariance test compares against the pre-F3 dess_env.py. It reads F3_BASELINE_SOURCE
(default audit_tmp/remediation_stageC/F3/before/environments/dess_env.py) and is skipped if absent.
"""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from environments.primary_marl import make_primary_env, primary_config
from utility.obs_interface import AGENT_OBS_FEATURES

BASELINE = Path(os.environ.get("F3_BASELINE_SOURCE", ROOT / "audit_tmp" / "remediation_stageC" / "F3" / "before"
                               / "environments" / "dess_env.py"))
FEEDERS = (("ieee33", IEEE33_CONFIG, 90.0, 2400.0), ("ieee69", IEEE69_CONFIG, 90.0, 4900.0))
PV, CURT = AGENT_OBS_FEATURES.index("pv_kw"), AGENT_OBS_FEATURES.index("feeder_curtailment_kw")


def same(a, b, key):
    """Exact equality; NaN equals NaN; dicts compared key by key."""
    if isinstance(a, dict):
        assert set(a) == set(b), key
        for k in a:
            same(a[k], b[k], f"{key}.{k}")
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), key)


class References(unittest.TestCase):
    def test_references_are_rounded_up_train_maxima(self):
        for name, cfg, pv_ref, curt_ref in FEEDERS:
            env = make_primary_env(copy.deepcopy(cfg), mode="train", seed=0)
            self.assertEqual(env.agent_obs_dim, 10)
            for n in (env.normalizer, env.graph_builder.normalizer):
                self.assertEqual((n.max_pv_kw, n.max_curtailment_kw), (pv_ref, curt_ref), name)
            pv = np.stack([env._get_profiles(i)[2] for i in env.splits["train_indices"]])
            bus_max, feeder_max = pv.max(axis=0).max(), pv.sum(axis=1).max()
            self.assertTrue((pv.max(axis=0) > 0).all(), "every bus carries PV")
            # train-only maxima, rounded up by less than 10 %
            self.assertTrue(bus_max <= pv_ref < 1.1 * bus_max, (name, bus_max))
            self.assertTrue(feeder_max <= curt_ref < 1.1 * feeder_max, (name, feeder_max))

    def test_missing_reference_is_refused(self):
        for key in ("max_pv_kw", "max_curtailment_kw"):
            cfg = copy.deepcopy(IEEE33_CONFIG)
            cfg["normalization"][key] = None
            with self.assertRaisesRegex(ValueError, key + " must be configured explicitly"):
                DESSEnv(config=cfg, mode="train", seed=0)

    def test_observation_scaling(self):
        for name, cfg, pv_ref, curt_ref in FEEDERS:
            env = make_primary_env(copy.deepcopy(cfg), mode="train", seed=3)
            obs, _ = env.reset(seed=3)
            rng = np.random.default_rng(0)
            for _ in range(60):
                obs, _, te, tr, info = env.step(rng.uniform(-1, 1, env.num_agents).astype(np.float32))
                load, _, pv, _, _ = env._get_profiles(env.current_index)
                np.testing.assert_allclose(obs["agent_obs"][:, PV], pv[env.dess_buses] / pv_ref, rtol=1e-6)
                np.testing.assert_allclose(obs["agent_obs"][:, CURT], env.current_kpis["curtailment_kw"] / curt_ref,
                                           rtol=1e-6, atol=1e-9)
                pv_col = list(env.config["graph"]["node_features"]).index("pv_kw")
                np.testing.assert_allclose(obs["x"][:, pv_col], pv / pv_ref, rtol=1e-6)
                if te or tr:
                    break


@unittest.skipUnless(BASELINE.is_file(), f"pre-F3 baseline not found: {BASELINE}")
class PhysicsUnchanged(unittest.TestCase):
    def test_only_the_curtailment_column_changes(self):
        spec = importlib.util.spec_from_file_location("dess_env_pre_f3", BASELINE)
        old_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_mod)
        for name, cfg, pv_ref, curt_ref in FEEDERS:
            new = DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=4)
            old = old_mod.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=4)
            on, oo = new.reset(seed=4)[0], old.reset(seed=4)[0]
            rng = np.random.default_rng(4)
            curtailed = 0
            for t in range(192):
                a = rng.uniform(-1, 1, new.num_agents).astype(np.float32) if t % 3 else np.full(new.num_agents, -1.0, np.float32)
                on, rn, tn, trn, i_new = new.step(a.copy())
                oo, ro, to, tro, i_old = old.step(a.copy())
                np.testing.assert_array_equal(rn, ro)
                self.assertEqual((tn, trn), (to, tro))
                for k in i_old:
                    if k.startswith("time_") or k == "pf_solves_in_step":
                        continue
                    same(i_new[k], i_old[k], k)
                np.testing.assert_array_equal(on["x"], oo["x"])
                keep = [c for c in range(len(AGENT_OBS_FEATURES)) if c != CURT]
                np.testing.assert_array_equal(on["agent_obs"][:, keep], oo["agent_obs"][:, keep])
                # pre-F3 divided feeder curtailment by the per-bus PV scale; F3 uses the feeder scale
                np.testing.assert_allclose(on["agent_obs"][:, CURT] * curt_ref, oo["agent_obs"][:, CURT] * pv_ref,
                                           rtol=1e-6, atol=1e-6)
                curtailed += bool(i_new["kpis"]["curtailment_kw"] > 0)
                if tn or trn:
                    on, oo = new.reset(seed=50 + t)[0], old.reset(seed=50 + t)[0]
            self.assertGreater(curtailed, 0, f"{name}: no curtailment exercised")


if __name__ == "__main__":
    unittest.main()
