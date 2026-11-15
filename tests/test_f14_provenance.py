"""F14: experiment and checkpoint provenance. All artifacts in temporary directories."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.ieee33_config import IEEE33_CONFIG
from environments.primary_marl import make_primary_env
from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
from experiments.topology_generalization.topology_utils import build_topology_config
from utility.provenance import checkpoint_provenance, check_checkpoint_provenance
import utility.run_manifest as rm

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mlp(env):
    torch.manual_seed(0)
    return MLPMADDPG(agent_obs_dim=env.agent_obs_dim, num_agents=env.num_agents, device="cpu",
                     actor_hidden_dims=(16,), critic_hidden_dims=(16,))


class Provenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=1)
        cls.tmp = tempfile.TemporaryDirectory()
        cfg3 = build_topology_config("TP3", base_config=copy.deepcopy(IEEE33_CONFIG), output_root=cls.tmp.name)
        cls.env3 = make_primary_env(cfg3, mode="val", seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_manifest_records_all_inputs_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "c.pt"
            ck.write_bytes(b"weights")
            run = rm.RunManifest(Path(tmp) / "run", "mlp", 3, "val", checkpoint=ck)
            run.add_environment(self.env)
            run.add_environment(self.env3, topology="TP3")
            data = json.loads(run.path.read_text())
        e = data["environments"][0]
        paths = self.env.config["paths"]
        for key in ("line_data", "bus_data", "gridtensor_line_data", "gridtensor_bus_data", "dess_buses",
                    "processed_time_series"):
            self.assertEqual(e["input_files"][key]["sha256"], sha(paths[key]), key)
        self.assertEqual(data["environments"][1]["input_files"]["gridtensor_line_data"]["sha256"],
                         sha(self.env3.config["paths"]["gridtensor_line_data"]))
        self.assertNotEqual(data["environments"][1]["input_files"]["gridtensor_line_data"]["sha256"],
                            e["input_files"]["gridtensor_line_data"]["sha256"])
        self.assertEqual(e["normalization_identity"]["agent_obs"]["max_pv_kw"], self.env.normalizer.max_pv_kw)
        self.assertIn("environments/dess_env.py", e["source_files"]["files"])
        self.assertEqual(e["interface_identity"]["agent_obs_dim"], 10)
        self.assertEqual(data["checkpoint_file"]["sha256"], hashlib.sha256(b"weights").hexdigest())

    def test_contradictory_topology_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = rm.RunManifest(Path(tmp) / "run", "mlp", 3, "val")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                run.add_environment(self.env, topology="TP3")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                run.add_environment(self.env3, topology="TP1")

    def test_checkpoint_roundtrip_and_mismatch_policy(self):
        env = self.env
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "a.pt")
            agent = mlp(env)
            agent.save(path, provenance=checkpoint_provenance(env))
            other = mlp(env)
            other.load(path, env=env)
            self.assertEqual(other.last_load_provenance_status, "verified")
            for p, q in zip(agent.actor.parameters(), other.actor.parameters()):
                self.assertTrue(torch.equal(p, q))
            # Zero-shot on another topology is allowed (config/network differ, interface identical).
            other.load(path, env=self.env3)
            self.assertEqual(other.last_load_provenance_status, "verified")
            # Normalization mismatch is refused.
            with patch.object(env.normalizer, "max_pv_kw", 45.0), self.assertRaisesRegex(ValueError, "max_pv_kw"):
                other.load(path, env=env)
            with patch.object(env.normalizer, "max_curtailment_kw", 1.0), \
                    self.assertRaisesRegex(ValueError, "max_curtailment_kw"):
                other.load(path, env=env)
            # Interface mismatch is refused.
            with patch.object(env, "agent_obs_features", tuple(reversed(env.agent_obs_features))), \
                    self.assertRaisesRegex(ValueError, "agent_obs_features"):
                other.load(path, env=env)
            # Legacy checkpoint (no provenance) predates F3 normalization: refused with an environment
            # (F3; previously accepted with a warning), unchecked without one.
            legacy = str(Path(tmp) / "legacy.pt")
            agent.save(legacy)
            with self.assertRaisesRegex(ValueError, "before the F3 observation normalization"):
                other.load(legacy, env=env)
            other.load(legacy)
            self.assertEqual(other.last_load_provenance_status, "not_checked")
            # A provenance block from before F3 (no normalization schema) is refused too.
            ck = torch.load(path)
            ck["provenance"]["normalization"].pop("schema")
            pre_f3 = str(Path(tmp) / "pre_f3.pt")
            torch.save(ck, pre_f3)
            with self.assertRaisesRegex(ValueError, "normalization.schema"):
                other.load(pre_f3, env=env)
        with self.assertRaisesRegex(ValueError, "no provenance"):
            check_checkpoint_provenance(None, env, "x")

    def test_trainer_checkpoints_carry_provenance(self):
        from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
        with tempfile.TemporaryDirectory() as tmp:
            t = MLPMADDPGTrainer(env=self.env, eval_env=None, device="cpu",
                                 config=dict(checkpoint_dir=tmp, buffer_size=8, store_buffer_on_device=False,
                                             actor_hidden_dims=(16,), critic_hidden_dims=(16,), verbose=False))
            t.save_checkpoint(Path(tmp) / "ck.pt")
            prov = torch.load(Path(tmp) / "ck.pt")["provenance"]
            self.assertEqual(prov["interface"]["agent_obs_features"], list(self.env.agent_obs_features))
            self.assertEqual(prov["input_files"]["processed_time_series"], sha(self.env.config["paths"]["processed_time_series"]))
            t.load_checkpoint(Path(tmp) / "ck.pt")
            self.assertEqual(t.agent.last_load_provenance_status, "verified")


if __name__ == "__main__":
    unittest.main()
