"""F2: split authorization, TP1-only selection, frozen versioned selection records, final-test provenance.

End-to-end tests drive the REAL evaluators (run_eval_mlp_ieee33 / run_eval_ieee33), the REAL selector
(scripts/select_checkpoint.py), real environments and real (untrained) checkpoints. For runtime only,
the S-SEL-v1 schedule is patched to its first 2 real validation days; the production 61-day schedule
is asserted in test_f16_validation_schedule.py. Records are tampered with ONLY to prove that tampered
or held-out-topology records are rejected. Everything is written to temporary directories.
"""
import contextlib
import copy
import hashlib
import importlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from environments.primary_marl import make_primary_env
from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
import utility.validation_schedule as vs
from utility.evaluation_protocol import (ProtocolError, authorize, protocol_config, verify_selection,
                                         NOT_APPLICABLE)
from utility.provenance import checkpoint_provenance

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
    import torch_geometric  # noqa: F401
    HAVE_PYG = True
except ImportError:
    HAVE_PYG = False


class Rules(unittest.TestCase):
    def test_split_and_purpose_rules(self):
        with self.assertRaisesRegex(ProtocolError, "requires explicit final-test authorization"):
            authorize("test", final=False, learned=False)
        with self.assertRaisesRegex(ProtocolError, "requires explicit final-test authorization"):
            authorize("all", final=False, learned=False)
        with self.assertRaisesRegex(ProtocolError, "not allowed"):
            authorize("val", final=True, learned=False)
        with self.assertRaisesRegex(ProtocolError, "requires --final"):
            authorize("val", run_purpose="final_test", learned=False)
        with self.assertRaisesRegex(ProtocolError, "validation split"):
            authorize("train", run_purpose="checkpoint_selection")
        with self.assertRaisesRegex(ProtocolError, "learned controllers only"):
            authorize("val", run_purpose="checkpoint_selection", learned=False)
        with self.assertRaisesRegex(ProtocolError, "requires --selection"):
            authorize(final=True, learned=True)
        with self.assertRaisesRegex(ProtocolError, "only with --final"):
            authorize("val", learned=True, selection="x.json")
        p = authorize(final=True, learned=False, topologies=["TP1", "TP4"])
        self.assertEqual((p["data_split"], p["run_purpose"], p["selection"]), ("test", "final_test", NOT_APPLICABLE))
        self.assertEqual(authorize(learned=False)["data_split"], "val")

    def test_held_out_topologies_only_in_final(self):
        for purpose in ("development", "checkpoint_selection", "baseline_validation"):
            with self.assertRaisesRegex(ProtocolError, "Held-out topologies"):
                authorize("val", run_purpose=purpose, learned=purpose == "checkpoint_selection", topologies=["TP2"])
        with self.assertRaisesRegex(ProtocolError, r"Held-out topologies \['TP7'\]"):
            authorize("val", learned=False, topologies=["TP1", "TP7"])

    def test_env_refuses_december_without_authorization(self):
        for mode in ("test", "all"):
            with self.assertRaisesRegex(ValueError, "final-test authorization"):
                DESSEnv(config=copy.deepcopy(IEEE33_CONFIG), mode=mode, seed=0)
        with self.assertRaisesRegex(ValueError, "final-test authorization"):
            make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="test", seed=0)
        cfg = protocol_config(IEEE33_CONFIG, authorize(final=True, learned=False))
        env = make_primary_env(cfg, mode="test", seed=0)
        self.assertEqual(env.mode, "test")
        self.assertTrue(all(env.time_series["df"].loc[s, env.time_series["datetime_col"]].month == 12
                            for s in env.episode_start_indices))
        self.assertNotIn("evaluation_protocol", protocol_config(IEEE33_CONFIG, authorize(learned=False)))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="train", seed=0)
        cls.ckpts = []
        for i in range(2):
            torch.manual_seed(i)
            agent = MLPMADDPG(agent_obs_dim=env.agent_obs_dim, num_agents=env.num_agents, device="cpu")
            path = cls.tmp / "ckpt" / f"maddpg_step_{5000 * (i + 1)}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(path), provenance=checkpoint_provenance(env))
            cls.ckpts.append(path)
        cls.real_pool = vs.validation_pool
        cls.patches = [patch.object(vs, "validation_pool", lambda e: cls.real_pool(e)[:2]),
                       patch.object(vs, "SELECTION_EPISODES", 2)]
        for p in cls.patches:
            p.start()
        cls.mlp = importlib.import_module("experiments.no_gnn_maddpg.run_eval_mlp_ieee33")
        cls.sweep = cls.tmp / "sweep"
        for ck in cls.ckpts:
            step = ck.stem.split("_")[-1]
            cls.run_eval(cls.mlp, ["--checkpoint", str(ck), "--run_purpose", "checkpoint_selection",
                                   "--episodes", "schedule", "--run_name", f"step_{step}"], cls.sweep)
        cls.selector = importlib.import_module("scripts.select_checkpoint")

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def run_eval(cls, module, argv, csv_dir):
        fig = cls.tmp / "fig" / argv[argv.index("--run_name") + 1]
        args = ["prog"] + argv + ["--csv_dir", str(csv_dir), "--figure_dir", str(fig), "--cpu"]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def select(self, *extra, sweep=None, out=None):
        argv = ["--feeder", "ieee33", "--model", "mlp", "--run", "run_0", "--sweep_dir", str(sweep or self.sweep),
                "--output_root", str(out or self.tmp / "selections"), "--bootstrap", "200"] + list(extra)
        with contextlib.redirect_stdout(io.StringIO()):
            self.selector.main(argv)
        key = Path(out or self.tmp / "selections") / "ieee33" / "mlp" / "run_0"
        return sorted(key.glob("*/selection.json"), key=lambda p: json.loads(p.read_text())["version"])

    def test_1_selection_record_frozen_versioned_and_not_replaceable(self):
        out = self.tmp / "sel_a"
        first = self.select(out=out)
        self.assertEqual(len(first), 1)
        rec = json.loads(first[0].read_text())
        means = {}
        for step_dir in sorted(self.sweep.glob("step_*")):
            man = json.loads((step_dir / "run_manifest.json").read_text())
            self.assertEqual(man["evaluation_protocol"]["run_purpose"], "checkpoint_selection")
            self.assertEqual(man["validation_schedule"]["schedule_id"], "S-SEL-v1")
            ep = pd.read_csv(step_dir / "episode_summary.csv")
            means[int(step_dir.name[5:])] = ep.loc[ep.policy == "mlp_maddpg", "total_reward_mean"].mean()
        best = max(means, key=lambda s: (means[s], -s))
        self.assertEqual(rec["selected"]["step"], best)
        self.assertAlmostEqual(rec["selected"]["metric_value"], means[best], places=9)
        self.assertEqual(rec["selected"]["checkpoint_sha256"], sha(self.ckpts[best // 5000 - 1]))
        self.assertEqual((rec["topology"], rec["data_split"], rec["version"], rec["supersedes"]), ("TP1", "val", 1, None))
        self.assertEqual(len(rec["schedule"]["episodes"]), 2)
        self.assertIn("bootstrap_top1_frequency", rec["reliability"])
        self.assertIsNotNone(rec["configuration_provenance"]["configuration_sha256"])
        frozen = first[0].read_bytes()
        # Re-running selection never replaces the frozen record.
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.select(out=out)
        with self.assertRaisesRegex(ProtocolError, "requires --reason"):
            self.select("--supersede", str(first[0]), out=out)
        with self.assertRaisesRegex(ProtocolError, "latest record"):
            self.select("--supersede", str(self.tmp / "other.json"), "--reason", "x", out=out)
        self.assertEqual(first[0].read_bytes(), frozen)
        second = self.select("--supersede", str(first[0]), "--reason", "re-evaluated sweep", out=out)
        self.assertEqual(len(second), 2)
        rec2 = json.loads(second[1].read_text())
        self.assertEqual((rec2["version"], rec2["supersedes"]["sha256"]), (2, hashlib.sha256(frozen).hexdigest()))
        self.assertNotEqual(rec2["selection_id"], rec["selection_id"])
        self.assertEqual(first[0].read_bytes(), frozen)                     # v1 untouched

    def test_2_tp2_or_tampered_results_rejected(self):
        bad = self.tmp / "sweep_tp2"
        shutil.copytree(self.sweep, bad)
        man_path = sorted(bad.glob("step_*"))[0] / "run_manifest.json"
        man = json.loads(man_path.read_text())
        man["environments"][0]["topology"] = "TP2"
        man_path.write_text(json.dumps(man))
        with self.assertRaisesRegex(ProtocolError, "TP1 only"):
            self.select(sweep=bad, out=self.tmp / "sel_bad")
        man["environments"][0]["topology"] = "TP1"
        man["evaluation_protocol"]["topologies"] = ["TP3"]
        man_path.write_text(json.dumps(man))
        with self.assertRaisesRegex(ProtocolError, "TP1 only"):
            self.select(sweep=bad, out=self.tmp / "sel_bad")
        man["evaluation_protocol"]["topologies"] = ["TP1"]
        man["evaluation_protocol"]["data_split"] = "test"
        man_path.write_text(json.dumps(man))
        with self.assertRaisesRegex(ProtocolError, "not a validation checkpoint-selection run"):
            self.select(sweep=bad, out=self.tmp / "sel_bad")
        # A checkpoint changed after it was evaluated is refused.
        shutil.rmtree(bad)
        shutil.copytree(self.sweep, bad)
        man_path = sorted(bad.glob("step_*"))[0] / "run_manifest.json"
        man = json.loads(man_path.read_text())
        changed = self.tmp / "changed.pt"
        changed.write_bytes(Path(man["checkpoint_file"]["path"]).read_bytes() + b"x")
        man["checkpoint_file"]["path"] = str(changed)
        man_path.write_text(json.dumps(man))
        with self.assertRaisesRegex(ProtocolError, "changed since evaluation"):
            self.select(sweep=bad, out=self.tmp / "sel_bad")
        self.assertFalse((self.tmp / "sel_bad").exists())
        # A selection record claiming a held-out topology is refused by the final-test gate.
        path = self.select(out=self.tmp / "sel_tp")[0]
        rec = json.loads(path.read_text())
        rec["topology"] = "TP2"
        forged = self.tmp / "forged_selection.json"
        forged.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ProtocolError, "TP1 validation only"):
            verify_selection(forged)

    def test_3_final_test_requires_and_records_frozen_selection(self):
        sel = self.select(out=self.tmp / "sel_final")[0]
        rec = json.loads(sel.read_text())
        other = [c for c in self.ckpts if sha(c) != rec["selected"]["checkpoint_sha256"]][0]
        final_dir = self.tmp / "final"
        with self.assertRaisesRegex(ProtocolError, "requires --selection"):
            self.run_eval(self.mlp, ["--checkpoint", str(other), "--final", "--episodes", "1", "--run_name", "f0"], final_dir)
        with self.assertRaisesRegex(ProtocolError, "not the frozen selected checkpoint"):
            self.run_eval(self.mlp, ["--checkpoint", str(other), "--final", "--selection", str(sel), "--episodes", "1",
                                     "--run_name", "f1"], final_dir)
        with self.assertRaisesRegex(ProtocolError, "requires explicit final-test authorization"):
            self.run_eval(self.mlp, ["--checkpoint", str(other), "--mode", "test", "--episodes", "1", "--run_name", "f2"],
                          final_dir)
        self.assertFalse(final_dir.exists())                                  # refused before any output
        self.run_eval(self.mlp, ["--final", "--selection", str(sel), "--episodes", "1", "--include_zero",
                                 "--run_name", "f3"], final_dir)
        man = json.loads((final_dir / "f3" / "run_manifest.json").read_text())
        proto = man["evaluation_protocol"]
        self.assertEqual((proto["data_split"], proto["run_purpose"], proto["final_test"]), ("test", "final_test", True))
        self.assertEqual(proto["selection"]["selection_sha256"], sha(sel))
        self.assertEqual(proto["selection_by_policy"]["mlp_maddpg"]["selection_sha256"], sha(sel))
        self.assertEqual(proto["selection_by_policy"]["zero"], NOT_APPLICABLE)
        self.assertEqual(man["checkpoint_file"]["sha256"], rec["selected"]["checkpoint_sha256"])
        self.assertEqual(man["environments"][0]["mode"], "test")

    @unittest.skipUnless(HAVE_PYG, "torch_geometric not installed")
    def test_4_non_learning_baselines(self):
        gnn = importlib.import_module("scripts.run_eval_ieee33")
        base = self.tmp / "baselines"
        self.run_eval(gnn, ["--baselines_only", "--run_purpose", "baseline_validation", "--episodes", "schedule",
                            "--run_name", "val"], base)
        man = json.loads((base / "val" / "run_manifest.json").read_text())
        self.assertEqual(man["evaluation_protocol"]["selection_by_policy"], dict(zero=NOT_APPLICABLE, random=NOT_APPLICABLE))
        self.assertEqual(man["validation_schedule"]["schedule_id"], "S-SEL-v1")
        with self.assertRaisesRegex(ProtocolError, "requires explicit final-test authorization"):
            self.run_eval(gnn, ["--baselines_only", "--mode", "test", "--episodes", "1", "--run_name", "t0"], base)
        self.run_eval(gnn, ["--baselines_only", "--final", "--episodes", "1", "--run_name", "t1"], base)
        man = json.loads((base / "t1" / "run_manifest.json").read_text())
        self.assertEqual((man["evaluation_protocol"]["data_split"], man["evaluation_protocol"]["selection"]),
                         ("test", NOT_APPLICABLE))
        self.assertIsNone(man["checkpoint"])


if __name__ == "__main__":
    unittest.main()
