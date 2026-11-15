"""F12: directional-backup candidate status, eligibility and caching; dispatch unchanged.

Unit tests drive the real search() with explicit power-flow rules (as test_directional_backup does).
The dispatch-invariance test compares against the pre-F12 dess_env.py and directional_backup.py
(F12_BASELINE_DIR, default audit_tmp/remediation_stageD/F12/before) and is skipped if absent.
"""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from environments.directional_backup import search, settings, violation_metrics
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.primary_marl import make_primary_env, primary_config
from test_directional_backup import pf

BASE = Path(os.environ.get("F12_BASELINE_DIR", ROOT / "audit_tmp" / "remediation_stageD" / "F12" / "before"))
PLACEHOLDER = dict(converged=False, voltage_pu=np.array([np.nan]), line_current_pu=np.array([np.nan]),
                   grid_import_kw=np.nan, grid_import_kvar=np.nan)


def run(fn, start=0., low=75., high=75., budget=20):
    """Real search(); fn(power) -> pf result (None = solver failure placeholder)."""
    calls = []

    def evaluate(p):
        calls.append(p.copy())
        res = fn(float(p[0]))
        return dict(pf_result=res if res is not None else dict(PLACEHOLDER), accepted_action=p / 75,
                    feasible=True, dess_power_kw=p.copy())      # caller's claim is never trusted
    initial = evaluate(np.array([start]))
    cfg = settings(dict(backup_mode="directional_incremental", pf_budget=budget))
    best, diag = search(initial, [start], [low], [high], evaluate, lambda r: violation_metrics(r, [1]), cfg)
    return best, diag, calls


class CandidateStatus(unittest.TestCase):
    def test_legitimate_zero_candidate_is_selected(self):
        # Only |P| <= 5 kW is feasible: the zero reference is the unique feasible evaluated candidate.
        best, diag, calls = run(lambda p: pf([1 if abs(p) <= 5 else .94]), start=60)
        self.assertTrue(best["feasible"])
        self.assertEqual(best["dess_power_kw"][0], 0)
        self.assertEqual(diag["backup_selected_role"], "zero_reference")
        self.assertTrue(diag["backup_selected_evaluated"] and diag["backup_selected_executed_zero"])
        self.assertTrue(best["correction_fallback_to_zero"])
        self.assertEqual(diag["backup_outcome"], "fixed")
        self.assertTrue(any(np.all(c == 0) for c in calls))     # zero was physically evaluated

    def test_diagnostic_placeholder_zero_is_never_eligible(self):
        # The zero action's power flow FAILS (placeholder); the evaluate() claim feasible=True is ignored.
        best, diag, _ = run(lambda p: None if p == 0 else pf([.94 + p * .0001]), start=60)
        self.assertNotEqual(diag["backup_selected_role"], "zero_reference")
        self.assertFalse(np.all(best["dess_power_kw"] == 0))
        self.assertFalse(best["correction_fallback_to_zero"])
        self.assertTrue(np.isinf(diag["backup_zero_score"]))
        self.assertFalse(diag["backup_zero_comparison_valid"])
        self.assertEqual(diag["backup_invalid_pf_evaluations"], 1)
        # Every candidate placeholder -> no dispatch candidate at all.
        best, diag, _ = run(lambda p: None, start=60)
        self.assertIsNone(best)
        self.assertEqual(diag["backup_selected_role"], "none")
        self.assertEqual(diag["backup_eligible_candidates"], 0)

    def test_cache_hit_preserves_evaluation_status(self):
        # Initial proposal is already zero and infeasible: the zero reference is a cache hit of the
        # initial candidate -- no extra PF, same measured status, recorded with both roles.
        best, diag, calls = run(lambda p: pf([.94 + p * .0004]), start=0)
        self.assertEqual(sum(np.all(c == 0) for c in calls), 1)   # zero solved once (the initial)
        self.assertGreaterEqual(diag["backup_cache_hits"], 1)
        self.assertEqual(diag["backup_zero_reference_role"], "initial|zero_reference")
        self.assertEqual(diag["backup_zero_score"], diag["backup_initial_score"])
        self.assertTrue(best["feasible"])                           # a discharge candidate fixes it
        self.assertFalse(best["correction_fallback_to_zero"])
        self.assertEqual(len(calls), diag["backup_pf_evaluations"])  # no PF for cache hits

    def test_infeasible_only_candidates_best_effort(self):
        best, diag, _ = run(lambda p: pf([.93 + p * .0001]), start=10, high=20)
        self.assertFalse(best["feasible"])
        self.assertTrue(diag["backup_best_effort"])
        self.assertEqual(diag["backup_outcome"], "partly fixed")
        self.assertEqual(diag["backup_selected_role"], "directional")
        self.assertEqual(diag["backup_eligible_candidates"], diag["backup_evaluated_candidates"])

    def test_exhausted_budget(self):
        for budget in (2, 3, 7, 20):
            best, diag, calls = run(lambda p: pf([.90 + p * 0.0]), start=30, budget=budget)
            self.assertLessEqual(len(calls), budget)
            self.assertEqual(diag["backup_pf_evaluations"], len(calls))
            self.assertEqual(diag["backup_evaluated_candidates"], len(calls))
            if budget == 2:                                         # initial + zero reference only
                self.assertEqual(diag["backup_selected_role"], "initial")
        with self.assertRaises(ValueError):
            settings(dict(pf_budget=21))


def load_pre_f12():
    mods = {}
    for name in ("directional_backup", "dess_env"):
        spec = importlib.util.spec_from_file_location(f"pre_f12_{name}", BASE / "environments" / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    for attr in ("search", "legacy_diagnostics", "backup_attribution", "valid_pf"):
        setattr(mods["dess_env"], attr, getattr(mods["directional_backup"], attr))
    return mods["dess_env"]


class RealAC(unittest.TestCase):
    def test_failed_dispatch_zero_is_not_reported_as_evaluated(self):
        for cfg in (IEEE33_CONFIG, IEEE69_CONFIG):
            env = make_primary_env(copy.deepcopy(cfg), mode="train", seed=2)
            env.reset(seed=2)
            env.last_pf_result["voltage_pu"][env.dess_buses] = .94
            real = env.power_flow.run_power_flow
            n = {"k": 0}

            def fail_after_first(**kw):
                n["k"] += 1
                if n["k"] > 1:
                    raise RuntimeError("injected PF failure")
                return real(**kw)
            with patch.object(env, "_is_feasible_pf_result", return_value=False), \
                    patch.object(env.power_flow, "run_power_flow", side_effect=fail_after_first):
                _, _, _, tr, info = env.step(-np.ones(env.num_agents, np.float32))
            self.assertTrue(info["episode_failed"])
            self.assertIsNone(info["zero_fallback_feasible"])
            self.assertFalse(info["zero_fallback_evaluated"])
            self.assertLessEqual(info["backup_pf_evaluations"], 20)

    def test_legacy_failed_zero_fallback_is_unknown_not_infeasible(self):
        """Legacy hard-feasibility path: the zero fallback's PF fails -> unknown, not 'infeasible zero'."""
        cfg = copy.deepcopy(IEEE33_CONFIG)
        cfg["backup"] = dict(backup_mode="legacy_scaling")
        cfg["doe"]["pf_failure_mode"] = "logged_fallback"
        from environments.dess_env import DESSEnv
        env = DESSEnv(config=cfg, mode="train", seed=2)
        env.reset(seed=2)
        real = env.power_flow.run_power_flow
        calls = {"k": 0}

        def fail_zero(**kw):
            calls["k"] += 1
            if not np.any(kw["dess_power_kw"]):        # the zero fallback's power flow fails
                raise RuntimeError("injected PF failure")
            return real(**kw)
        with patch.object(env, "_is_feasible_pf_result", return_value=False), \
                patch.object(env.power_flow, "run_power_flow", side_effect=fail_zero):
            _, _, _, _, info = env.step(-np.ones(env.num_agents, np.float32))
        self.assertTrue(info["correction_fallback_to_zero"])     # zero fallback was attempted
        self.assertTrue(info["episode_failed"])
        self.assertIsNone(info["zero_fallback_feasible"])        # pre-F12: False (placeholder as evidence)
        self.assertFalse(info["zero_fallback_evaluated"])
        self.assertFalse(info["backup_zero_evaluated"])          # placeholder is not a zero reference

    @unittest.skipUnless((BASE / "environments" / "dess_env.py").is_file(), f"pre-F12 baseline not found: {BASE}")
    def test_dispatch_unchanged_vs_pre_f12(self):
        old_mod = load_pre_f12()
        import environments.dess_env as new_mod
        keys = ("final_battery_power_kw", "accepted_action", "soc_after", "final_voltage_pu", "final_line_current_pu",
                "backup_activated", "backup_outcome", "backup_pf_evaluations", "backup_direction",
                "backup_penalty_applied", "feasible", "kpis", "backup_zero_score", "backup_cause")
        for name, cfg in (("ieee33", IEEE33_CONFIG), ("ieee69", IEEE69_CONFIG)):
            new = new_mod.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=12)
            old = old_mod.DESSEnv(primary_config(copy.deepcopy(cfg)), mode="train", seed=12)
            new.reset(seed=12), old.reset(seed=12)
            rng, backups, zero_selected, budget_max = np.random.default_rng(12), 0, 0, 0
            for t in range(192):
                a = np.full(new.num_agents, -1.0, np.float32) if (t // 6) % 2 == 0 else rng.uniform(-1, 1, new.num_agents).astype(np.float32)
                _, rn, tn, trn, i_new = new.step(a.copy())
                _, ro, to, tro, i_old = old.step(a.copy())
                np.testing.assert_array_equal(rn, ro)
                for k in keys:
                    if isinstance(i_old[k], dict):
                        self.assertEqual(i_new[k], i_old[k], k)
                    else:
                        np.testing.assert_array_equal(np.asarray(i_new[k]), np.asarray(i_old[k]), f"{name} {t} {k}")
                backups += bool(i_new["backup_activated"])
                zero_selected += i_new["backup_selected_role"] == "zero_reference"
                budget_max = max(budget_max, i_new["backup_pf_evaluations"])
                if i_new["backup_activated"]:
                    np.testing.assert_array_equal(i_new["reward_term_backup_penalty"], -20.0)
                if tn or trn:
                    new.reset(seed=100 + t), old.reset(seed=100 + t)
            self.assertLessEqual(budget_max, 20)
            if name == "ieee69":
                self.assertGreater(backups, 0, "IEEE69 all-charging blocks must activate the backup")
            print(f"F12_INVARIANCE {name}: steps=192 backups={backups} zero_reference_selected={zero_selected} "
                  f"max_backup_pf={budget_max}")


if __name__ == "__main__":
    unittest.main()
