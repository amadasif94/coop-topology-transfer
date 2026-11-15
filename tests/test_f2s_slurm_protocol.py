"""F2S: SLURM entry points follow the evaluation protocol (static checks + real argument parsing).

* the 8 checkpoint-selection sweeps evaluate the TP1 validation schedule only (no December, no TP2+);
* the final-test entry points pass --final and, for learned controllers, the frozen selection;
* the new scripts contain no hard-coded absolute filesystem paths;
* every flag a script passes is accepted by the Python evaluator's own argument parser, and the
  resulting arguments pass (or, for finals without a selection, are refused by) the protocol.
"""
import importlib
import re
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utility.evaluation_protocol import ProtocolError, authorize_args

SL = ROOT / "SLURM Script"
SWEEPS = {"eval_conv_gat": "scripts.run_eval_ieee33", "eval_conv_gcn": "scripts.run_eval_ieee33",
          "eval_conv_tagconv": "scripts.run_eval_ieee33", "eval_conv_mlp": "experiments.no_gnn_maddpg.run_eval_mlp_ieee33",
          "eval_gat_ieee69": "scripts.run_eval_ieee69", "eval_gcn_ieee69": "scripts.run_eval_ieee69",
          "eval_tagconv_ieee69": "scripts.run_eval_ieee69", "eval_mlp_ieee69": "experiments.no_gnn_maddpg.run_eval_mlp_ieee69"}
NEW = ("final_test_learned", "final_test_baselines", "validation_baselines")
VALUES = {"${CKPT}": "ck.pt", "${RUN_NAME}": "r", "${MODEL}": "gcn", "${SELECTION}": "sel.json", "${EPISODES}": "all",
          "${TOPOLOGIES}": "TP1", "${HORIZON:-96}": "96", "${SCENARIOS:-100}": "100"}


def python_calls(text):
    """Every 'python -u <script> ...' command with its continuation lines, variables substituted."""
    calls, lines = [], text.split("\n")
    for i, line in enumerate(lines):
        m = re.search(r'python -u "?([\w/.${}]+)"?(.*)', line)
        if not m or "<<" in line:
            continue
        cmd = [m.group(1), m.group(2)]
        j = i
        while lines[j].rstrip().endswith("\\"):
            j += 1
            cmd.append(lines[j])
        flat = " ".join(c.rstrip("\\") for c in cmd)
        flat = re.sub(r'\$\{ARCH\[@\]\+"\$\{ARCH\[@\]\}"\}', "--gnn_type gcn", flat)
        flat = re.sub(r'"\$\{RUN_NAME:-[^}]*\}[^"]*"', "r", flat)
        for k, v in VALUES.items():
            flat = flat.replace(k, v)
        calls.append(shlex.split(flat))
    return calls


def parse(module_name, argv):
    module = importlib.import_module(module_name)
    with patch.object(sys, "argv", ["prog"] + argv):
        return module.parse_args()


class SlurmProtocol(unittest.TestCase):
    def test_sweeps_use_validation_schedule_only(self):
        for name, module in SWEEPS.items():
            text = (SL / f"{name}.slurm").read_text()
            self.assertNotIn("--mode test", text, name)
            self.assertNotIn("--include_zero", text, name)
            self.assertIn('RUN_NAME="selection', text, name)
            calls = python_calls(text)
            self.assertEqual(len(calls), 1, name)
            args = parse(module, calls[0][1:])
            self.assertEqual((args.mode, args.episodes, args.run_purpose, args.final), ("val", "schedule",
                                                                                         "checkpoint_selection", False))
            protocol = authorize_args(args, learned=True)
            self.assertEqual((protocol["data_split"], protocol["topologies"]), ("val", ["TP1"]))

    def test_new_entry_points_have_no_absolute_paths(self):
        for name in NEW:
            text = (SL / f"{name}.slurm").read_text()
            self.assertIsNone(re.search(r"(^|[\s=\"'])/(network|home|usr|opt|scratch)\b", text, re.M), name)
            self.assertIn("GMARL_CONDA_ENV:?", text)
            self.assertNotIn("\r", text)

    def test_final_learned_requires_selection_and_final(self):
        text = (SL / "final_test_learned.slurm").read_text()
        self.assertIn('SELECTION:?', text)
        calls = python_calls(text)
        self.assertEqual(len(calls), 1)
        modules = {"experiments/topology_generalization/run_tp_eval.py": "experiments.topology_generalization.run_tp_eval",
                   "experiments/no_gnn_maddpg/run_tp_eval_mlp.py": "experiments.no_gnn_maddpg.run_tp_eval_mlp"}
        for script, module in modules.items():
            argv = [a for a in calls[0][1:]]
            if "mlp" in script:
                argv = [a for i, a in enumerate(argv) if a != "--gnn_type" and (i == 0 or argv[i - 1] != "--gnn_type")]
            args = parse(module, argv)
            self.assertTrue(args.final)
            self.assertEqual(args.selection, "sel.json")
            with self.assertRaisesRegex(Exception, "sel.json"):   # the evaluator opens and verifies it
                authorize_args(args, learned=True, topologies=args.topologies)

    def test_final_baselines_and_validation_baselines(self):
        calls = python_calls((SL / "final_test_baselines.slurm").read_text())
        self.assertEqual(len(calls), 2)
        zr = parse("experiments.topology_generalization.run_tp_eval", calls[0][1:])
        self.assertTrue(zr.final and zr.baselines_only)
        p = authorize_args(zr, learned=False, topologies=zr.topologies)
        self.assertEqual((p["data_split"], p["selection"]), ("test", "not_applicable"))
        self.assertIn("--final", calls[1])
        val = python_calls((SL / "validation_baselines.slurm").read_text())[0]
        args = parse("scripts.run_eval_ieee33", val[1:])
        self.assertEqual((args.mode, args.run_purpose, args.episodes, args.final), ("val", "baseline_validation",
                                                                                   "schedule", False))
        authorize_args(args, learned=False)

    def test_dev_run_cannot_reach_december_or_held_out_topologies(self):
        args = parse("experiments.topology_generalization.run_tp_eval", ["--checkpoint", "ck.pt", "--topologies", "TP1", "TP2"])
        with self.assertRaisesRegex(ProtocolError, "Held-out topologies"):
            authorize_args(args, learned=True, topologies=args.topologies)
        args = parse("scripts.run_eval_ieee33", ["--checkpoint", "ck.pt", "--mode", "test"])
        with self.assertRaisesRegex(ProtocolError, "final-test authorization"):
            authorize_args(args, learned=True)


if __name__ == "__main__":
    unittest.main()
