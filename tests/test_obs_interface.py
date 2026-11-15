"""F5: grid_stress removed from actor observations (11 -> 10 features).

The equivalence test compares against the pre-change dess_env.py. Point
F5_BASELINE_SOURCE at that file (default: audit_tmp/pre_f5/environments/dess_env.py);
it is skipped if the file is absent.
"""
import copy
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.primary_marl import make_primary_env, primary_config
import environments.dess_env as dess_module
from utility.obs_interface import AGENT_OBS_DIM, AGENT_OBS_FEATURES, VOLTAGE_COLUMN
sys.path.insert(0, str(ROOT / "tests"))
from f3_obs import assert_agent_obs_equal_except_f3_curtailment

FEEDERS = (("ieee33", IEEE33_CONFIG), ("ieee69", IEEE69_CONFIG))
BASELINE = Path(os.environ.get("F5_BASELINE_SOURCE",
                               ROOT / "audit_tmp" / "pre_f5" / "environments" / "dess_env.py"))


def _no_git_run():
    """RunManifest shells out to git; emulate a missing repo only where git is absent."""
    import subprocess
    real = subprocess.run

    def run(cmd, *a, **k):
        try:
            return real(cmd, *a, **k)
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, 127, "", "git not installed")
    return run


class ObservationInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {name: make_primary_env(copy.deepcopy(cfg), mode="train", seed=123) for name, cfg in FEEDERS}

    def test_ten_features_both_feeders(self):
        self.assertEqual(AGENT_OBS_DIM, 10)
        for env in self.envs.values():
            obs, _ = env.reset(seed=1)
            self.assertEqual(env.agent_obs_dim, 10)
            self.assertEqual(tuple(env.agent_obs_features), AGENT_OBS_FEATURES)
            self.assertEqual(obs["agent_obs"].shape, (env.num_agents, 10))
            self.assertEqual(env.observation_space["agent_obs"].shape, (env.num_agents, 10))
            obs, *_ = env.step(np.zeros(env.num_agents, np.float32))
            self.assertEqual(obs["agent_obs"].shape, (env.num_agents, 10))

    def test_no_line_current_summary_in_actor_obs(self):
        self.assertNotIn("grid_stress", AGENT_OBS_FEATURES)
        for env in self.envs.values():
            obs, _ = env.reset(seed=2)
            load, _, pv, price, _ = env._get_profiles(env.current_index)
            pv_used, _, _ = env._apply_curtailment(load, pv, env.last_dess_power_kw)
            soc = np.array([b.get_soc() for b in env.batteries], np.float32)
            v = env.last_pf_result["voltage_pu"]
            base = env._build_agent_observation(load, pv_used, price, v, soc)
            saved = dict(env.current_kpis)
            try:
                env.current_kpis = dict(saved, line_current_violation=3.0, voltage_violation_max=0.2, grid_stress=3.2)
                stressed = env._build_agent_observation(load, pv_used, price, v, soc)
                # grid_stress still drives the KPI reward.
                env.previous_kpis = dict(saved)
                kpi_stressed = env._calculate_kpi_reward()
                env.current_kpis = dict(saved)
                kpi_base = env._calculate_kpi_reward()
            finally:
                env.current_kpis = saved
            np.testing.assert_array_equal(base, stressed)
            self.assertAlmostEqual(kpi_base - kpi_stressed,
                                   env.reward_cfg.get("beta_grid_stress", 1.0) * 3.2, places=9)

    def test_doe_reads_correct_local_voltage(self):
        for env in self.envs.values():
            obs, _ = env.reset(seed=3)
            seen = []
            real = dess_module.apply_voltage_doe

            def spy(*args, **kwargs):
                seen.append(np.array(args[1], dtype=float))
                return real(*args, **kwargs)
            rng = np.random.default_rng(3)
            with patch.object(dess_module, "apply_voltage_doe", side_effect=spy):
                for _ in range(12):
                    pf_v = np.asarray(env.last_pf_result["voltage_pu"], float)[env.dess_buses]
                    col = obs["agent_obs"][:, VOLTAGE_COLUMN].astype(float)
                    obs, _, _, _, info = env.step(rng.uniform(-1, 1, env.num_agents).astype(np.float32))
                    np.testing.assert_array_equal(seen[-1], pf_v)
                    np.testing.assert_array_equal(info["pre_action_local_voltage_pu"], pf_v)
                    np.testing.assert_allclose(col, pf_v, rtol=0, atol=1e-7)
            self.assertEqual(AGENT_OBS_FEATURES[VOLTAGE_COLUMN], "voltage_pu")

    @unittest.skipUnless(BASELINE.is_file(), f"pre-F5 baseline not found: {BASELINE}")
    def test_equivalence_with_pre_f5_code(self):
        spec = importlib.util.spec_from_file_location("dess_env_pre_f5", BASELINE)
        old_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_module)
        compare = ("final_battery_power_kw", "dess_power_kw", "soc_before", "soc_after", "final_voltage_pu",
                   "final_line_current_pu", "physical_limited_power_kw", "doe_power_kw", "accepted_action",
                   "doe_corrective_shortfall_kw", "doe_preventive_clipped", "doe_corrective_intervened",
                   "backup_activated", "backup_outcome", "backup_pf_evaluations", "major_backup_penalty_applied",
                   "feasible", "kpis", "curtailment_by_bus_kw")
        for name, cfg in FEEDERS:
            new = dess_module.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=7)
            old = old_module.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=7)
            on, _ = new.reset(seed=7)
            oo, _ = old.reset(seed=7)
            rng = np.random.default_rng(11)
            steps = 0
            for t in range(200):
                self.assertEqual(oo["agent_obs"].shape[1], 11)
                # F3: only the feeder-curtailment scale changed (same physical value).
                assert_agent_obs_equal_except_f3_curtailment(on["agent_obs"], oo["agent_obs"][:, :10], new, old)
                for k in ("x", "edge_index", "edge_attr"):
                    np.testing.assert_array_equal(on[k], oo[k])
                np.testing.assert_array_equal(old.current_kpis["grid_stress"], new.current_kpis["grid_stress"])
                a = rng.uniform(-1, 1, new.num_agents).astype(np.float32)
                if t % 7 == 0:
                    a[:] = -1.0  # herd charging to exercise DOE/backup
                on, rn, tn, trn, i_new = new.step(a.copy())
                oo, ro, to, tro, i_old = old.step(a.copy())
                np.testing.assert_array_equal(rn, ro)
                self.assertEqual((tn, trn), (to, tro))
                for k in compare:
                    if isinstance(i_old[k], dict):
                        self.assertEqual(i_new[k], i_old[k], f"{name} step {t} {k}")
                    else:
                        np.testing.assert_array_equal(np.asarray(i_new[k]), np.asarray(i_old[k]), f"{name} step {t} {k}")
                steps += 1
                if tn or trn:
                    on, _ = new.reset(seed=100 + t)
                    oo, _ = old.reset(seed=100 + t)
            self.assertEqual(steps, 200)

    def test_models_train_replay_update_evaluate(self):
        from training.trainer import GraphMADDPGTrainer
        from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
        torch.set_num_threads(1)
        git_patch = (patch("utility.run_manifest.subprocess.run", side_effect=_no_git_run())
                     if shutil.which("git") is None else patch.dict(os.environ, {}))
        with git_patch:
            for name, cfg in FEEDERS:
                train_env = self.envs[name]
                eval_env = make_primary_env(copy.deepcopy(cfg), mode="val", seed=5)
                for model in ("gcn", "gat", "tagconv", "mlp"):
                    with tempfile.TemporaryDirectory() as tmp:
                        conf = dict(total_steps=12, warmup_steps=8, update_after=8, buffer_size=64, batch_size=4,
                                    eval_every=12, save_every=10**9, eval_episodes=1, max_eval_steps=3,
                                    checkpoint_dir=tmp, verbose=False, gnn_type=model, store_buffer_on_device=False,
                                    gnn_hidden_dim=16, gnn_embedding_dim=16, gnn_num_layers=2,
                                    actor_hidden_dims=(32,), critic_hidden_dims=(32,))
                        T = MLPMADDPGTrainer if model == "mlp" else GraphMADDPGTrainer
                        torch.manual_seed(0)
                        np.random.seed(0)
                        trainer = T(env=train_env, eval_env=eval_env, config=conf, device="cpu")
                        self.assertEqual(trainer.replay_buffer.agent_obs.shape[-1], 10)
                        logs = trainer.train()
                        self.assertEqual(len(trainer.replay_buffer), 12)
                        self.assertTrue(np.isfinite(logs[-1]["update"]["critic_loss"]))
                        self.assertEqual(logs[-1]["eval"]["completed_episodes"] + logs[-1]["eval"]["incomplete_episodes"], 1)
                        # Checkpoint round trip, then refusal of an 11-feature checkpoint.
                        ck = Path(tmp) / "maddpg_final.pt"
                        self.assertEqual(torch.load(ck)["agent_obs_dim"], 10)
                        obs, _ = eval_env.reset(seed=9)
                        fresh = T(env=train_env, eval_env=None, config=dict(conf, checkpoint_dir=tmp), device="cpu")
                        fresh.load_checkpoint(ck)
                        np.testing.assert_array_equal(trainer.agent.select_action(obs), fresh.agent.select_action(obs))
                        self._assert_legacy_checkpoint_rejected(T, train_env, conf, tmp)

    def _assert_legacy_checkpoint_rejected(self, T, env, conf, tmp):
        legacy = T(env=env, eval_env=None, config=dict(conf, checkpoint_dir=tmp), device="cpu")
        # Build an 11-feature model of the same class to emulate a pre-F5 checkpoint.
        cls = type(legacy.agent)
        kwargs = dict(agent_obs_dim=11, num_agents=env.num_agents, device="cpu",
                      actor_hidden_dims=conf["actor_hidden_dims"], critic_hidden_dims=conf["critic_hidden_dims"])
        if conf["gnn_type"] != "mlp":
            kwargs.update(node_feature_dim=env.observation_space["x"].shape[-1], dess_buses=env.dess_buses,
                          gnn_type=conf["gnn_type"], gnn_hidden_dim=conf["gnn_hidden_dim"],
                          gnn_embedding_dim=conf["gnn_embedding_dim"], gnn_num_layers=conf["gnn_num_layers"])
        old_agent = cls(**kwargs)
        path = Path(tmp) / "legacy11.pt"
        old_agent.save(str(path))
        with self.assertRaisesRegex(ValueError, r"agent_obs_dim=11.*expects agent_obs_dim=10.*pre-F5"):
            legacy.load_checkpoint(path)
        # Checkpoints saved before agent_obs_dim was recorded: inferred from the first actor layer.
        ckpt = torch.load(path)
        ckpt.pop("agent_obs_dim", None)
        torch.save(ckpt, path)
        with self.assertRaisesRegex(ValueError, r"agent_obs_dim=11.*pre-F5"):
            legacy.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
