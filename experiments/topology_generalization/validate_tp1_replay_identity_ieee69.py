"""
Validate an IEEE69 action-replay run (read only; writes a new report inside the replay directory).

    identity  (TP1 -> TP1): the replay must reproduce the ORIGINAL TP1 evaluation step by step.
        python -u experiments/topology_generalization/validate_tp1_replay_identity_ieee69.py \
            --kind identity --model gcn --replay_dir <replay run dir>
    target    (TP1 -> TP10 smoke/full): replayed actions == recorded TP1 actions, complete episodes,
              same days / timestamps / initial SOC (reset state) as the TP10 feedback evaluation.
        python -u experiments/topology_generalization/validate_tp1_replay_identity_ieee69.py \
            --kind target --model gcn --replay_dir <replay run dir>

Rows are aligned by (start_index, step) for the replayed episodes of the SAME model / selection /
source file (taken from replay_manifest.json and re-hashed). Per metric: matched steps, missing and
duplicate keys, max and mean absolute error, values beyond tolerance, PASS / FAIL / MISSING.
Discrete flags and the replayed raw action: exact equality. Continuous metrics: |replay - original|
<= atol + rtol*|original| (defaults 1e-8 / 1e-8; same code, data and inputs, so differences should
be zero or at round-off). A required metric absent from either file is MISSING and the result is
FAIL; optional metrics are reported when present. Timing columns are never compared.
"""

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.topology_generalization.replay_utils_ieee69 import (
    N_AGENTS, RAW_FIELD, STEPS_PER_EPISODE, ReplayError, learned_policy, load_source_actions, read_filtered,
    sha256_file, utc_stamp, write_json)

AGENT = lambda prefix: [f"{prefix}_{i}" for i in range(N_AGENTS)]
REQUIRED_EXACT = ["date_time", "record_type", "feasible", "converged", "raw_infeasible_requested", "requested_feasible",
                  "backup_activated", "backup_cause", "final_network_feasible", "episode_failed"]
REQUIRED_CLOSE = (AGENT("requested_action_agent") + AGENT("accepted_action_agent") + AGENT("soc_agent")
                  + AGENT("dess_power_kw_agent") + ["energy_cost", "grid_import_kw", "grid_import_mwh", "voltage_deviation",
                  "min_voltage_pu", "max_voltage_pu", "max_line_current_pu", "requested_min_voltage_pu",
                  "requested_max_voltage_pu", "throughput_mwh", "reward_sum"])
REQUIRED_VECTOR_EXACT = [RAW_FIELD]
REQUIRED_VECTOR_CLOSE = ["soc_before", "soc_after"]
OPTIONAL_EXACT = ["minor_corrected_action", "major_infeasible_action", "correction_iterations", "correction_fallback_to_zero",
                  "requested_pf_converged", "requested_battery_limited_count", "backup_outcome", "backup_pf_evaluations",
                  "backup_penalty_applied", "major_backup_penalty_applied", "physical_proposal_feasible",
                  "executed_network_violation", "executed_voltage_violation", "executed_current_violation",
                  "doe_candidate_network_violation", "requested_overloaded_branch_count", "requested_overloaded_branches",
                  "episode_completed", "infeasible_action"]
OPTIONAL_CLOSE = ["reward_mean", "curtailment_kw", "curtailment_mwh", "grid_stress", "max_voltage_violation",
                  "max_line_current_violation", "requested_max_voltage_violation", "requested_max_line_current_violation",
                  "requested_battery_clipping_kw", "requested_actual_charge_kw", "requested_actual_discharge_kw",
                  "requested_actual_net_injection_kw", "backup_initial_score", "backup_final_score", "backup_zero_score"]
OPTIONAL_VECTOR_CLOSE = ["marl_prepared_action", "doe_conditioned_action", "physical_limited_power_kw", "doe_power_kw",
                         "executed_battery_power_kw", "final_battery_power_kw", "pre_action_local_voltage_pu",
                         "final_voltage_pu", "final_line_current_pu"]


def as_list(v):
    if isinstance(v, str) and v.strip():
        out = ast.literal_eval(v.strip())
        return np.asarray(out, dtype=float) if isinstance(out, (list, tuple)) else None
    return None


def norm_exact(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
        return str(int(v))
    return str(v)


def compare(src, rep, column, kind, atol, rtol):
    """src/rep: aligned Series (same index). Returns (n, n_exceed, max_err, mean_err)."""
    if kind == "exact":
        a, b = src.map(norm_exact), rep.map(norm_exact)
        bad = (a != b)
        return len(a), int(bad.sum()), (np.nan if not len(a) else float(bad.any())), np.nan
    if kind == "close":
        a, b = pd.to_numeric(src, errors="coerce").to_numpy(float), pd.to_numeric(rep, errors="coerce").to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        err = np.where(both_nan, 0.0, np.abs(a - b))
        err = np.where(np.isnan(err), np.inf, err)
        bad = err > atol + rtol * np.abs(np.nan_to_num(a))
        return len(a), int(bad.sum()), float(err.max()) if len(err) else np.nan, float(err[np.isfinite(err)].mean()) if np.isfinite(err).any() else np.nan
    # vectors
    errs, n_bad = [], 0
    for x, y in zip(src, rep):
        xa, ya = as_list(x), as_list(y)
        if xa is None and ya is None:
            errs.append(0.0)
            continue
        if xa is None or ya is None or xa.shape != ya.shape:
            n_bad += 1
            errs.append(np.inf)
            continue
        if kind == "vector_exact":
            e = float(np.abs(xa.astype(np.float32) - ya.astype(np.float32)).max()) if xa.size else 0.0
            n_bad += int(e != 0.0)
        else:
            d = np.abs(xa - ya)
            e = float(d.max()) if d.size else 0.0
            n_bad += int((d > atol + rtol * np.abs(xa)).any())
        errs.append(e)
    errs = np.asarray(errs)
    fin = errs[np.isfinite(errs)]
    return len(errs), n_bad, float(errs.max()) if len(errs) else np.nan, float(fin.mean()) if len(fin) else np.nan


def key_report(df, label):
    dup = int(df.duplicated(["start_index", "step"]).sum())
    per_ep = df.groupby("start_index")["step"].agg(["count", "min", "max"])
    incomplete = per_ep[(per_ep["count"] != STEPS_PER_EPISODE) | (per_ep["min"] != 0) | (per_ep["max"] != STEPS_PER_EPISODE - 1)]
    return dict(rows=len(df), duplicate_keys=dup, episodes=len(per_ep), incomplete_episodes=incomplete.index.tolist(), label=label)


def identity(args, man, rep_df, policy, source_csv):
    starts = sorted(rep_df["start_index"].unique())
    groups = [("exact", REQUIRED_EXACT, True), ("close", REQUIRED_CLOSE, True), ("vector_exact", REQUIRED_VECTOR_EXACT, True),
              ("vector_close", REQUIRED_VECTOR_CLOSE, True), ("exact", OPTIONAL_EXACT, False), ("close", OPTIONAL_CLOSE, False),
              ("vector_close", OPTIONAL_VECTOR_CLOSE, False)]
    cols = [c for _, cs, _ in groups for c in cs]
    src_df, _ = read_filtered(source_csv, "TP1", policy, ["start_index", "step", *cols])
    src_df = src_df[src_df["start_index"].isin(starts)]
    keys = dict(source=key_report(src_df, "original TP1"), replay=key_report(rep_df, "TP1 identity replay"))
    merged = src_df.merge(rep_df, on=["start_index", "step"], how="outer", suffixes=("__src", "__rep"), indicator=True)
    missing = dict(only_in_source=int((merged["_merge"] == "left_only").sum()), only_in_replay=int((merged["_merge"] == "right_only").sum()))
    both = merged[merged["_merge"] == "both"]
    rows = []
    for kind, cs, required in groups:
        for c in cs:
            s_has, r_has = c in src_df.columns, c in rep_df.columns
            if not (s_has and r_has):
                rows.append(dict(metric=c, kind=kind, required=required, in_source=s_has, in_replay=r_has,
                                 status="MISSING" if required else "NOT OBSERVABLE"))
                continue
            n, bad, mx, mean = compare(both[f"{c}__src"], both[f"{c}__rep"], c, kind, args.atol, args.rtol)
            rows.append(dict(metric=c, kind=kind, required=required, in_source=True, in_replay=True, matched_steps=n,
                             n_exceed=bad, max_abs_err=mx, mean_abs_err=mean, status="PASS" if bad == 0 and n > 0 else "FAIL"))
    table = pd.DataFrame(rows)
    ok_keys = (not missing["only_in_source"] and not missing["only_in_replay"] and keys["source"]["duplicate_keys"] == 0
               and keys["replay"]["duplicate_keys"] == 0 and not keys["replay"]["incomplete_episodes"]
               and not keys["source"]["incomplete_episodes"])
    failing = table[table["status"].isin(["FAIL", "MISSING"])]
    status = "PASS" if ok_keys and failing.empty and len(both) else "FAIL"
    net = (man.get("network") or {})
    extra = dict(network_representation=dict(archived_tp1=net.get("archived_tp1"), input_files_vs_original_TP1=net.get(
        "input_files_vs_original_TP1"), backup_cross_check=net.get("backup_cross_check")))
    if not net.get("input_files_vs_original_TP1"):
        status = "FAIL"
        extra["network_problem"] = "replay manifest lacks the verified TP1 input-file identity"
    return status, table, keys, missing, len(starts), extra


def target(args, man, rep_df, policy, source_csv):
    actions, episodes, _ = load_source_actions(source_csv, args.model, args.expected_episodes)
    tgt = man["target_topology"]
    starts = sorted(rep_df["start_index"].unique())
    keys = dict(replay=key_report(rep_df, f"{tgt} replay"))
    rows = []
    bad_actions = 0
    for _, r in rep_df.iterrows():
        a = np.asarray(ast.literal_eval(r[RAW_FIELD]), dtype=np.float32)
        bad_actions += int(not np.array_equal(a, actions[int(r["start_index"])][int(r["step"])]))
    rows.append(dict(metric="replayed raw action == recorded TP1 raw action", matched_steps=len(rep_df), n_exceed=bad_actions,
                     status="PASS" if bad_actions == 0 and len(rep_df) else "FAIL"))
    calls_ok = int(man.get("actor_calls", -1)) == len(rep_df)
    rows.append(dict(metric="ReplayAgent calls == replayed steps (no new inference)", matched_steps=len(rep_df),
                     n_exceed=0 if calls_ok else 1, status="PASS" if calls_ok else "FAIL"))
    fb = Path((man.get("network") or {}).get("feedback_dir", ""))
    fb_df, fb_missing = read_filtered(fb / "step_metrics.csv", tgt, policy, ["start_index", "step", "date_time", "soc_before", "episode_id"])
    fb_df = fb_df[fb_df["start_index"].isin(starts)]
    keys["feedback"] = key_report(fb_df, f"{tgt} feedback")
    m = fb_df.merge(rep_df, on=["start_index", "step"], how="outer", suffixes=("__fb", "__rep"), indicator=True)
    both = m[m["_merge"] == "both"]
    for c, kind in (("date_time", "exact"), ("episode_id", "exact")):
        n, bad, _, _ = compare(both[f"{c}__fb"], both[f"{c}__rep"], c, kind, 0, 0)
        rows.append(dict(metric=f"{c} equals TP10 feedback", matched_steps=n, n_exceed=bad, status="PASS" if bad == 0 and n else "FAIL"))
    first = both[both["step"] == 0]
    if "soc_before" in fb_df and "soc_before" in rep_df:
        n, bad, mx, _ = compare(first["soc_before__fb"], first["soc_before__rep"], "soc_before", "vector_close", 0.0, 0.0)
        rows.append(dict(metric="initial SOC (step 0 soc_before) equals TP10 feedback", matched_steps=n, n_exceed=bad,
                         max_abs_err=mx, status="PASS" if bad == 0 and n else "FAIL"))
    else:
        rows.append(dict(metric="initial SOC equals TP10 feedback", status="MISSING"))
    missing = dict(only_in_feedback=int((m["_merge"] == "left_only").sum()), only_in_replay=int((m["_merge"] == "right_only").sum()))
    table = pd.DataFrame(rows)
    ok_keys = (not missing["only_in_feedback"] and not missing["only_in_replay"] and keys["replay"]["duplicate_keys"] == 0
               and not keys["replay"]["incomplete_episodes"] and keys["feedback"]["duplicate_keys"] == 0)
    status = "PASS" if ok_keys and (table["status"] == "PASS").all() else "FAIL"
    return status, table, keys, missing, len(starts), dict(tp10_checks=(man.get("network") or {}).get("tp10_checks"),
                                                            network_vs_feedback=(man.get("network") or {}).get("tp10_vs_feedback_run"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["identity", "target"])
    ap.add_argument("--model", required=True, choices=["gcn", "tagconv", "gat", "mlp"])
    ap.add_argument("--replay_dir", required=True)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--expected_episodes", type=int, default=31)
    args = ap.parse_args()
    rd = Path(args.replay_dir)
    man = json.loads((rd / "replay_manifest.json").read_text())
    if man.get("model") != args.model:
        raise SystemExit(f"replay is for model {man.get('model')}, not {args.model}")
    source_csv = Path(man["source"]["source_csv"])
    src_sha = sha256_file(source_csv)
    if src_sha != man["source"]["source_sha256"]:
        raise SystemExit("source step_metrics.csv changed since the replay (sha256 differs)")
    tgt = man["target_topology"]
    if (args.kind == "identity") != (tgt == "TP1"):
        raise SystemExit(f"--kind {args.kind} does not match replay target {tgt}")
    policy = learned_policy(args.model, source_csv, "TP1")
    rep_df = pd.read_csv(rd / "step_metrics.csv", low_memory=False)
    rep_df = rep_df[(rep_df["topology_case"] == tgt) & (rep_df["policy"] == policy)]
    try:
        fn = identity if args.kind == "identity" else target
        status, table, keys, missing, n_eps, extra = fn(args, man, rep_df, policy, source_csv)
    except ReplayError as exc:
        status, table, keys, missing, n_eps, extra = "FAIL", pd.DataFrame(), {}, {}, 0, dict(error=str(exc))
    stamp = utc_stamp()
    table.to_csv(rd / f"validation_{args.kind}_{stamp}.csv", index=False)
    report = dict(kind=args.kind, status=status, model=args.model, target_topology=tgt, replay_dir=str(rd.resolve()),
                  source_csv=str(source_csv.resolve()), source_sha256=src_sha, selection=man.get("selection"),
                  episodes_validated=n_eps if status == "PASS" else 0, episodes_in_replay=n_eps,
                  tolerances=dict(atol=args.atol, rtol=args.rtol, discrete="exact", raw_action="exact (float32)"),
                  keys=keys, missing_keys=missing, metrics=table.to_dict(orient="records"), **extra,
                  created_utc=datetime.now(timezone.utc).isoformat())
    path = rd / f"validation_{args.kind}_{stamp}.json"
    write_json(path, report)
    print(table.to_string(index=False) if len(table) else "(no metric table)")
    print(f"{args.kind.upper()} VALIDATION {status}  episodes={n_eps}  report={path}")
    sys.exit(0 if status == "PASS" else 4)


if __name__ == "__main__":
    main()
