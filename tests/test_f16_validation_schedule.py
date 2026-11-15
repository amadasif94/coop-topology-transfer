"""F16: fixed validation episode schedules (S-SEL-v1 all 61 days, S-PER-v1 8 days), TP1 only.

The training-invariance test compares against the pre-F16 periodic_validation.py. It reads
F16_BASELINE_SOURCE (default audit_tmp/remediation_stageC/F16/before/utility/periodic_validation.py)
and is skipped if absent. All outputs go to temporary directories.
"""
import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.primary_marl import make_primary_env
from experiments.topology_generalization.topology_utils import build_topology_config
import utility.periodic_validation as pv
from utility.validation_schedule import (selection_schedule, periodic_schedule, PERIODIC_DATES, SEED_BASE,
                                         SELECTION_SCHEDULE_ID, PERIODIC_SCHEDULE_ID)

BASELINE = Path(os.environ.get("F16_BASELINE_SOURCE", ROOT / "audit_tmp" / "remediation_stageC" / "F16" / "before"
                               / "utility" / "periodic_validation.py"))
FEEDERS = (("ieee33", IEEE33_CONFIG), ("ieee69", IEEE69_CONFIG))


class Schedules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {n: make_primary_env(copy.deepcopy(c), mode="val", seed=0) for n, c in FEEDERS}

    def test_selection_schedule_is_all_61_october_november_days(self):
        for name, env in self.envs.items():
            s = selection_schedule(env)
            eps = s["episodes"]
            self.assertEqual((s["schedule_id"], s["split"], s["topology"], len(eps)), (SELECTION_SCHEDULE_ID, "val", "TP1", 61))
            self.assertEqual([e["start_index"] for e in eps], [26208 + 96 * k for k in range(61)])
            self.assertEqual([e["seed"] for e in eps], [SEED_BASE + k for k in range(61)])
            self.assertEqual(eps[0]["episode_id"], "val_2018-10-01")
            self.assertEqual(eps[-1]["episode_id"], "val_2018-11-30")
            df, col = env.time_series["df"], env.time_series["datetime_col"]
            for e in eps:
                stamps = pd.to_datetime(df.loc[e["start_index"]:e["start_index"] + 95, col])
                self.assertEqual(len(stamps), 96)
                self.assertTrue(set(stamps.dt.month) <= {10, 11}, e)
                self.assertEqual(stamps.dt.date.nunique(), 1)
                self.assertTrue(set(range(e["start_index"], e["start_index"] + 96)) <= set(env.splits["val_indices"]))
            self.assertEqual(selection_schedule(env), s)          # deterministic, same sha
            self.assertEqual(len({e["episode_id"] for e in eps}), 61)

    def test_periodic_schedule_is_frozen_subset_and_reproduces_pv_rule(self):
        for name, env in self.envs.items():
            p, s = periodic_schedule(env), selection_schedule(env)
            self.assertEqual(p["schedule_id"], PERIODIC_SCHEDULE_ID)
            self.assertEqual([e["episode_id"][4:] for e in p["episodes"]], list(PERIODIC_DATES[env.num_buses]))
            by_id = {e["episode_id"]: e for e in s["episodes"]}
            for e in p["episodes"]:
                self.assertEqual(e, by_id[e["episode_id"]])      # same start index and seed
            self.assertEqual(sum(e["episode_id"][9:11] == "10" for e in p["episodes"]), 4)
            # Re-derive the frozen dates from dataset-only daily PV energy (no controller involved).
            df = env.time_series["df"]
            pv_cols = [c for c in df.columns if c.startswith("renewable_active_power_node_")]
            picks = []
            for month in ("10", "11"):
                days = [e for e in s["episodes"] if e["episode_id"][9:11] == month]
                energy = [float(df.loc[e["start_index"]:e["start_index"] + 95, pv_cols].sum(axis=1).sum()) for e in days]
                order = sorted(range(len(days)), key=lambda i: (energy[i], days[i]["start_index"]))
                picks += [days[order[int(np.floor(q * len(days)))]]["episode_id"][4:] for q in (.125, .375, .625, .875)]
            self.assertEqual(sorted(picks), list(PERIODIC_DATES[env.num_buses]))

    def test_refuses_other_splits_and_held_out_topologies(self):
        train = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=0)
        for fn in (selection_schedule, periodic_schedule):
            with self.assertRaisesRegex(ValueError, "mode='val'"):
                fn(train)
        with tempfile.TemporaryDirectory() as tmp:
            tp3 = make_primary_env(build_topology_config("TP3", base_config=copy.deepcopy(IEEE33_CONFIG),
                                                         output_root=tmp), mode="val", seed=0)
            for fn in (selection_schedule, periodic_schedule):
                with self.assertRaisesRegex(ValueError, "held-out topology"):
                    fn(tp3)


def trainer(tmp, env, eval_env, steps=64):
    from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
    torch.manual_seed(0)
    np.random.seed(0)
    return MLPMADDPGTrainer(env, eval_env, dict(batch_size=8, buffer_size=200, warmup_steps=32, update_after=32,
                                               total_steps=steps, eval_every=32, save_every=10 ** 9, eval_episodes=10,
                                               max_eval_steps=3, verbose=False, checkpoint_dir=tmp,
                                               actor_hidden_dims=(16,), critic_hidden_dims=(16,),
                                               store_buffer_on_device=False), device="cpu")


class PeriodicValidation(unittest.TestCase):
    def test_fixed_days_every_evaluation_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=0)
            val = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=1)
            t = trainer(tmp, env, val)
            buf = io.StringIO()
            with redirect_stdout(buf):
                for step in (1, 2):
                    t.global_step = step
                    t.evaluate()
            self.assertIn("8 of its 8 days (eval_episodes=10)", buf.getvalue())
            expected = periodic_schedule(val)
            for step in (1, 2):
                out = Path(tmp) / "validation" / f"step_{step}"
                ep = pd.read_csv(out / "episode_summary.csv")
                self.assertEqual(ep["start_index"].tolist(), [e["start_index"] for e in expected["episodes"]])
                self.assertEqual(ep["schedule_episode_id"].tolist(), [e["episode_id"] for e in expected["episodes"]])
                man = json.loads((out / "run_manifest.json").read_text())
                self.assertEqual(man["validation_schedule"]["sha256"], expected["sha256"])
                self.assertEqual(man["arguments"]["run_purpose"], "periodic_validation")
                self.assertEqual(man["environments"][0]["intended_episodes"], 8)
                agg = pd.read_csv(out / "aggregate_summary.csv")
                self.assertEqual(agg["schedule_id"].iloc[0], PERIODIC_SCHEDULE_ID)
                self.assertEqual(agg["schedule_days_evaluated"].iloc[0], 8)
            # A smaller explicit request evaluates the first N fixed days, never random days.
            t.eval_episodes, t.global_step = 3, 3
            with redirect_stdout(io.StringIO()):
                t.evaluate()
            ep = pd.read_csv(Path(tmp) / "validation" / "step_3" / "episode_summary.csv")
            self.assertEqual(ep["start_index"].tolist(), [e["start_index"] for e in expected["episodes"][:3]])

    @unittest.skipUnless(BASELINE.is_file(), f"pre-F16 baseline not found: {BASELINE}")
    def test_training_unchanged_by_schedule(self):
        """Periodic validation never feeds learning: final weights equal the pre-F16 run's."""
        spec = importlib.util.spec_from_file_location("periodic_validation_pre_f16", BASELINE)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        weights = []
        for evaluate in (pv.evaluate_trainer, old.evaluate_trainer):
            with tempfile.TemporaryDirectory() as tmp:
                env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=0)
                val = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=1)
                t = trainer(tmp, env, val, steps=80)
                with patch.object(pv, "evaluate_trainer", evaluate), redirect_stdout(io.StringIO()):
                    t.train()
                weights.append([p.detach().clone() for m in (t.agent.actor, t.agent.critics) for p in m.parameters()])
        for a, b in zip(*weights):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
