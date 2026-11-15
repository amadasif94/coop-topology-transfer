"""Cross-file compatibility: dess_env <-> directional_backup <-> feasibility <-> reporting.

Guards the failure seen on the cluster (dess_env.py importing backup_attribution
from a directional_backup.py that did not define it). Explicitly imports DESSEnv,
runs the real AC environment on both feeders, and checks attribution diagnostics
end to end (env info -> evaluation_fields -> step_fields -> diagnostic_summary).
"""
import copy
import importlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAUSES = ("none", "action_induced", "preexisting_not_worsened", "preexisting_worsened", "unknown")


class Imports(unittest.TestCase):
    def test_required_names_resolve(self):
        dess = importlib.import_module("environments.dess_env")
        db = importlib.import_module("environments.directional_backup")
        fe = importlib.import_module("utility.feasibility")
        rep = importlib.import_module("utility.evaluation_reporting")
        obs = importlib.import_module("utility.obs_interface")
        for module, names in ((db, ("backup_attribution", "evaluation_fields", "search", "settings",
                                    "violation_metrics", "valid_pf", "legacy_diagnostics")),
                              (fe, ("BACKUP_SCORE_TOLERANCE", "VOLTAGE_TOLERANCE_PU", "CURRENT_RELATIVE_TOLERANCE")),
                              (rep, ("step_fields", "diagnostic_summary", "augment_summaries", "failure_row")),
                              (dess, ("DESSEnv",))):
            for name in names:
                self.assertTrue(hasattr(module, name), f"{module.__name__}.{name} missing")
        self.assertEqual(obs.AGENT_OBS_DIM, 10)
        self.assertEqual(fe.BACKUP_SCORE_TOLERANCE, fe.VOLTAGE_TOLERANCE_PU)


class AttributionOnRealAC(unittest.TestCase):
    def _rollout(self, cfg, steps, policy):
        from environments.primary_marl import make_primary_env
        from environments.directional_backup import evaluation_fields
        from utility.evaluation_reporting import step_fields, finish_episode
        env = make_primary_env(copy.deepcopy(cfg), mode="train", seed=5)
        obs, _ = env.reset(seed=5)
        rng = np.random.default_rng(5)
        rows, infos = [], []
        for t in range(steps):
            a = -np.ones(env.num_agents, np.float32) if policy == "charge" else \
                rng.uniform(-1, 1, env.num_agents).astype(np.float32)
            obs, reward, term, trunc, info = env.step(a)
            self.assertEqual(obs["agent_obs"].shape, (env.num_agents, 10))
            self.assertIn(info["backup_cause"], CAUSES)
            # Uniform rule: every successful-dispatch backup activation is penalized.
            if not info["episode_failed"]:
                self.assertEqual(info["backup_penalty_applied"], info["backup_activated"])
            if info["backup_activated"]:
                self.assertNotEqual(info["backup_cause"], "none")
            else:
                self.assertEqual(info["backup_cause"], "none")
            fields = evaluation_fields(info)
            for key in ("backup_cause", "zero_battery_feasible", "zero_battery_score",
                        "backup_penalty_applied", "backup_activation_count", "backup_penalty_applied_count"):
                self.assertIn(key, fields)
            self.assertEqual(fields["backup_activation_count"], int(info["backup_activated"]))
            rows.append(dict(step=t, reward_mean=float(np.mean(reward)), **step_fields(info, env, term, trunc, obs)))
            infos.append(info)
            if term or trunc:
                break
        return env, finish_episode(rows, env), infos

    def test_both_feeders_attribution_reporting(self):
        from configs.ieee33_config import IEEE33_CONFIG
        from configs.ieee69_config import IEEE69_CONFIG
        from utility.evaluation_reporting import diagnostic_summary
        for name, cfg, policy, need_backup in (("ieee33", IEEE33_CONFIG, "random", False),
                                               ("ieee69", IEEE69_CONFIG, "charge", True)):
            env, rows, infos = self._rollout(cfg, 96, policy)
            frame = pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)))  # CSV round trip
            stats = diagnostic_summary(frame, episode=True)
            n_backup = sum(bool(i["backup_activated"]) for i in infos)
            if need_backup:
                self.assertGreater(n_backup, 0, f"{name}: herd charging should activate the backup")
            self.assertEqual(stats["backup_activation_count"], n_backup)
            self.assertEqual(stats["backup_penalty_applied_count"], n_backup)
            self.assertEqual(stats["backup_cause_measured_steps"], len(rows))
            total = sum(stats["backup_cause_" + c + "_count"] for c in CAUSES)
            self.assertEqual(total, len(rows))
            self.assertAlmostEqual(sum(stats["backup_cause_" + c + "_rate"] for c in CAUSES), 1.0)
            self.assertEqual(stats["backup_cause_none_count"], len(rows) - n_backup)
            for key in ("physical_clipping_rate", "doe_intervention_rate", "backup_activation_rate"):
                self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main()
