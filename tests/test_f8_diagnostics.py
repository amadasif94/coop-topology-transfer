"""F8: diagnostics are instrumentation only (plus the optional evaluation-only proposal PF).

The invariance test compares against the pre-F8 dess_env.py. It reads F8_BASELINE_SOURCE
(default audit_tmp/remediation/F8/before/environments/dess_env.py) and is skipped if absent.
"""
import copy
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.primary_marl import make_primary_env, primary_config
from environments.directional_backup import evaluation_fields
from utility.evaluation_reporting import diagnostic_summary
from test_experiment_integration import evaluator, EVALUATORS
from f3_obs import assert_agent_obs_equal_except_f3_curtailment

BASELINE = Path(os.environ.get("F8_BASELINE_SOURCE",
                               ROOT / "audit_tmp" / "remediation" / "F8" / "before" / "environments" / "dess_env.py"))
FEEDERS = (("ieee33", IEEE33_CONFIG), ("ieee69", IEEE69_CONFIG))
TERMS = ("grid_energy_cost", "grid_voltage_deviation", "grid_curtailment", "local", "kpi", "backup_penalty",
         "residual_penalty", "doe_penalty")
TIMES = ("time_doe_ms", "time_backup_search_ms", "time_pf_ms", "time_diagnostic_pf_ms", "time_env_other_ms")


def actions(env, t, rng):
    a = rng.uniform(-1, 1, env.num_agents).astype(np.float32)
    if t % 5 == 0:
        a[:] = -1.0   # herd charging: exercises DOE and the directional backup on IEEE69
    return a


class Diagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {n: make_primary_env(copy.deepcopy(c), mode="train", seed=5) for n, c in FEEDERS}

    def test_reward_terms_reconstruct_reward_and_timing_partition(self):
        for name, env in self.envs.items():
            env.reset(seed=5)
            rng, backups = np.random.default_rng(5), 0
            for t in range(60):
                _, r, te, tr, info = env.step(actions(env, t, rng))
                total = sum(info["reward_term_" + k] for k in TERMS)
                np.testing.assert_allclose(total, r, rtol=1e-6, atol=1e-4, err_msg=f"{name} step {t}")
                np.testing.assert_allclose(info["reward_terms_total"], total)
                backups += info["backup_activated"]
                if info["backup_activated"]:
                    self.assertTrue(np.all(info["reward_term_backup_penalty"] == env.reward_cfg["infeasible_action_penalty"]))
                parts = sum(info[k] for k in TIMES)
                self.assertTrue(all(info[k] >= 0 for k in TIMES))
                self.assertAlmostEqual(parts, info["time_env_step_ms"], delta=1e-6 + 1e-9 * info["time_env_step_ms"])
                self.assertGreaterEqual(info["pf_solves_in_step"], 2)
                if te or tr:
                    env.reset(seed=t)
            if name == "ieee69":
                self.assertGreater(backups, 0)

    def test_nonconvergence_terms_and_missing_executed_state(self):
        env = self.envs["ieee33"]
        env.reset(seed=1)
        with patch.object(env.power_flow, "run_power_flow", side_effect=RuntimeError("injected")):
            _, r, _, tr, info = env.step(np.zeros(env.num_agents, np.float32))
        self.assertTrue(tr)
        np.testing.assert_allclose(info["reward_terms_total"], r)
        np.testing.assert_array_equal(info["reward_term_nonconvergence_penalty"], env.reward_cfg["nonconvergence_penalty"])
        self.assertIsNone(info["executed_voltage_violation"])
        self.assertIsNone(info["executed_max_line_loading_pct"])

    def test_line_loading_and_violation_location(self):
        env = self.envs["ieee69"]
        env.reset(seed=5)
        rng, seen_violation = np.random.default_rng(5), False
        limits_a = env.grid.get_line_current_limits_pu() * env.grid.get_base_current_a()
        np.testing.assert_allclose(limits_a, 255.0)
        for t in range(40):
            _, _, te, tr, info = env.step(actions(env, t, rng))
            current_a = np.asarray(info["final_line_current_pu"]) * env.grid.get_base_current_a()
            np.testing.assert_allclose(info["executed_line_loading_pct"], 100 * current_a / limits_a, rtol=1e-6)
            self.assertAlmostEqual(info["executed_max_line_loading_pct"], 100 * current_a.max() / 255.0, places=4)
            for stage in ("doe_candidate_", "executed_"):
                viol = np.union1d(info[stage + "undervoltage_buses"], info[stage + "overvoltage_buses"])
                with_d, without = info[stage + "violating_buses_with_dess"], info[stage + "violating_buses_without_dess"]
                np.testing.assert_array_equal(np.union1d(with_d, without), viol)
                self.assertTrue(set(with_d) <= set(env.dess_buses))
                self.assertFalse(set(without) & set(env.dess_buses))
                self.assertEqual(info[stage + "violating_bus_has_dess"], bool(len(with_d)))
            seen_violation |= bool(info["doe_candidate_network_violation"])
            if info["doe_candidate_network_violation"] is not None:
                self.assertEqual(info["doe_candidate_network_violation"], bool(info["backup_activated"]))
            if te or tr:
                env.reset(seed=t)
        self.assertTrue(seen_violation, "herd charging should create DOE-candidate violations on IEEE69")

    def test_physical_proposal_diagnostic_is_isolated(self):
        env = self.envs["ieee33"]
        a = np.zeros(env.num_agents, np.float32)
        a[0] = -1.0
        results = {}
        for flag in (False, True):
            env.evaluation_diagnostics = flag
            env.reset(seed=123)
            env.last_pf_result["voltage_pu"][env.dess_buses[0]] = 0.955   # DOE limits this charge
            calls = {"n": 0}
            real = env.power_flow.run_power_flow

            def counting(**kw):
                calls["n"] += 1
                return real(**kw)
            with patch.object(env.power_flow, "run_power_flow", side_effect=counting):
                obs, r, te, tr, info = env.step(a.copy())
            results[flag] = (obs, r, info, calls["n"])
        env.evaluation_diagnostics = False
        (o0, r0, i0, n0), (o1, r1, i1, n1) = results[False], results[True]
        self.assertTrue(np.any(i0["doe_preventive_clipped"]))
        self.assertEqual(i0["physical_proposal_source"], "diagnostic_disabled")
        self.assertIsNone(i0["physical_proposal_feasible"])
        self.assertEqual(i1["physical_proposal_source"], "diagnostic_pf")
        self.assertIn(i1["physical_proposal_feasible"], (True, False))
        self.assertEqual(n1, n0 + 1)                                   # exactly one extra solve
        self.assertEqual(i1["pf_solves_in_step"], i0["pf_solves_in_step"])  # not counted as control PF
        self.assertEqual(i1["backup_pf_evaluations"], i0["backup_pf_evaluations"])
        self.assertEqual((i1["pf_failure_count"], i1["pf_failed_solve_count"]), (i0["pf_failure_count"], i0["pf_failed_solve_count"]))
        np.testing.assert_array_equal(r0, r1)
        for k in o0:
            np.testing.assert_array_equal(o0[k], o1[k])
        for k in ("final_battery_power_kw", "soc_after", "final_voltage_pu", "accepted_action", "doe_power_kw"):
            np.testing.assert_array_equal(np.asarray(i0[k]), np.asarray(i1[k]))

    def test_duration_summary(self):
        df = pd.DataFrame(dict(episode_id=[0, 0, 0, 0, 1, 1], delta_t_hours=[0.25] * 6,
                               executed_network_violation=[True, True, False, True, True, None],
                               executed_voltage_violation=[True, False, False, False, True, None],
                               executed_current_violation=[False] * 5 + [None]))
        s = diagnostic_summary(df)
        self.assertEqual((s["network_violation_steps"], s["network_violation_hours"]), (4, 1.0))
        self.assertEqual((s["network_violation_longest_run_steps"], s["network_violation_longest_run_hours"]), (2, 0.5))
        self.assertEqual(s["network_violation_measured_steps"], 5)   # the unmeasured step is excluded
        self.assertEqual((s["voltage_violation_steps"], s["voltage_violation_longest_run_steps"]), (2, 1))
        self.assertEqual(s["current_violation_hours"], 0.0)
        empty = diagnostic_summary(pd.DataFrame(dict(episode_id=[0])))
        self.assertTrue(np.isnan(empty["network_violation_hours"]))

    def test_exports_and_real_evaluator(self):
        env, scope = self.envs["ieee33"], evaluator(EVALUATORS[0])
        try:
            with patch.object(env, "episode_length", 3):
                rows = scope["evaluate_one_episode"](env, None, "random", 0, env.episode_start_indices[0], 7)
        finally:
            env.evaluation_diagnostics = False
        frame = pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)))
        for col in ("time_policy_ms", "time_env_step_ms", "time_pf_ms", "reward_term_kpi", "reward_terms_total",
                    "executed_max_line_loading_pct", "executed_violating_bus_has_dess", "physical_proposal_source",
                    "doe_candidate_network_violation", "delta_t_hours", "pf_solves_in_step"):
            self.assertIn(col, frame, col)
        self.assertEqual(len(json.loads(frame["executed_line_loading_pct"].iloc[0])), env.grid.get_num_lines())
        ep, agg = scope["summarize"](frame)
        for col in ("mean_time_policy_ms", "network_violation_hours", "max_executed_line_loading_pct"):
            self.assertIn(col, agg)

    def test_update_time_logged(self):
        import torch
        from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
        agent = MLPMADDPG(agent_obs_dim=10, num_agents=5, device="cpu", actor_hidden_dims=(8,), critic_hidden_dims=(8,))
        B = 4
        batch = dict(agent_obs=torch.randn(B, 5, 10), next_agent_obs=torch.randn(B, 5, 10),
                     actions=torch.zeros(B, 5, 1), rewards=torch.zeros(B, 5), dones=torch.zeros(B, 1))
        self.assertGreaterEqual(agent.update(batch)["update_time_ms"], 0.0)

    @unittest.skipUnless(BASELINE.is_file(), f"pre-F8 baseline not found: {BASELINE}")
    def test_physics_rewards_dispatch_unchanged_vs_pre_f8(self):
        spec = importlib.util.spec_from_file_location("dess_env_pre_f8", BASELINE)
        old_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_mod)
        import environments.dess_env as new_mod
        keys = ("final_battery_power_kw", "dess_power_kw", "soc_before", "soc_after", "final_voltage_pu",
                "final_line_current_pu", "accepted_action", "physical_limited_power_kw", "doe_power_kw",
                "backup_activated", "backup_outcome", "backup_pf_evaluations", "backup_penalty_applied",
                "major_backup_penalty_applied", "feasible", "kpis", "backup_cause", "curtailment_by_bus_kw")
        for name, cfg in FEEDERS:
            for diag in (False, True):
                new = new_mod.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=9)
                old = old_mod.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=9)
                new.evaluation_diagnostics = diag
                on, oo = new.reset(seed=9)[0], old.reset(seed=9)[0]
                rng = np.random.default_rng(9)
                for t in range(80):
                    for k in on:
                        if k == "agent_obs":   # F3: only the feeder-curtailment scale changed
                            assert_agent_obs_equal_except_f3_curtailment(on[k], oo[k], new, old)
                        else:
                            np.testing.assert_array_equal(on[k], oo[k])
                    a = actions(new, t, rng)
                    on, rn, tn, trn, i_new = new.step(a.copy())
                    oo, ro, to, tro, i_old = old.step(a.copy())
                    np.testing.assert_array_equal(rn, ro)
                    self.assertEqual((tn, trn), (to, tro))
                    for k in keys:
                        if isinstance(i_old[k], dict):
                            self.assertEqual(i_new[k], i_old[k])
                        else:
                            np.testing.assert_array_equal(np.asarray(i_new[k]), np.asarray(i_old[k]), f"{name} {t} {k}")
                    if tn or trn:
                        on, oo = new.reset(seed=100 + t)[0], old.reset(seed=100 + t)[0]


if __name__ == "__main__":
    unittest.main()
