"""
Validate an IEEE33 action-replay run (read only; writes a new report inside the replay directory).

    identity (TP1 -> TP1): the replay must reproduce the ORIGINAL TP1 final evaluation step by step.
        python -u experiments/topology_generalization/validate_tp1_replay_identity_ieee33.py \
            --kind identity --model gcn --replay_dir <replay run dir>
    target   (TP1 -> TP6 smoke/full): replayed actions == recorded TP1 actions; complete episodes; same
             episode ids, timestamps and initial SOC as the TP6 feedback rows of the same final run.
        python -u experiments/topology_generalization/validate_tp1_replay_identity_ieee33.py \
            --kind target --model gcn --replay_dir <replay run dir>

Alignment: (start_index, step) within the replayed episodes, same model/selection/source file (from
replay_manifest.json, re-hashed). Per metric: matched steps, missing/duplicate keys, max and mean absolute
error, number beyond tolerance, PASS / FAIL / MISSING / NOT OBSERVABLE. Discrete fields and the replayed raw
action: exact equality. Continuous: |replay - original| <= atol + rtol*|original| (default 1e-8 / 1e-8).
A required metric absent from either file is MISSING and the result is FAIL. Timing columns are never compared.
"""

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.topology_generalization.replay_utils_ieee33 import (
    N_AGENTS, RAW_FIELD, STEPS_PER_EPISODE, ReplayError, learned_policy, load_source_actions, read_filtered,
    sha256_file, utc_stamp, write_json)

AGENT = lambda prefix: [f"{prefix}_{i}" for i in range(N_AGENTS)]
REQUIRED_EXACT = ["date_time", "record_type", "feasible", "converged", "infeasible_action", "backup_activated",
                  "backup_cause", "final_network_feasible", "episode_failed"]
REQUIRED_CLOSE = (AGENT("requested_action_agent") + AGENT("accepted_action_agent") + AGENT("soc_agent")
                  + AGENT("dess_power_kw_agent") + ["energy_cost", "grid_import_kw", "grid_import_mwh", "voltage_deviation",
                  "min_voltage_pu", "max_voltage_pu", "max_line_current_pu", "throughput_mwh", "reward_sum"])
REQUIRED_VECTOR_EXACT = [RAW_FIELD]
REQUIRED_VECTOR_CLOSE = ["soc_before", "soc_after"]
OPTIONAL_EXACT = ["correction_iterations", "correction_fallback_to_zero", "backup_outcome", "backup_pf_evaluations",
                  "backup_penalty_applied", "major_backup_penalty_applied", "physical_proposal_feasible", "physical_clipping_any",
                  "executed_network_violation", "executed_voltage_violation", "executed_current_violation",
                  "doe_candidate_network_violation", "doe_intervened", "episode_completed", "backup_zero_infeasible"]
OPTIONAL_CLOSE = ["reward_mean", "curtailment_kw", "curtailment_mwh", "grid_stress", "max_voltage_violation",
                  "max_line_current_violation", "backup_initial_score", "backup_final_score", "backup_zero_score",
                  "backup_initial_min_voltage", "backup_final_min_voltage", "physical_clipped_kw_total",
                  "executed_max_line_loading_pct"]
OPTIONAL_VECTOR_CLOSE = ["marl_prepared_action", "physical_limited_action_normalized", "doe_conditioned_action",
                         "physical_limited_power_kw", "doe_power_kw", "executed_battery_power_kw", "final_battery_power_kw",
                         "physical_clipped_kw", "pre_action_local_voltage_pu", "final_voltage_pu", "final_line_current_pu"]


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


def compare(src, rep, kind, atol, rtol):
    """Aligned Series -> (n, n_exceed, max_abs_err, mean_abs_err)."""
    if kind == "exact":
        bad = src.map(norm_exact) != rep.map(norm_exact)
        return len(src), int(bad.sum()), (float(bad.any()) if len(src) else np.nan), np.nan
    if kind == "close":
        a, b = pd.to_numeric(src, errors="coerce").to_numpy(float), pd.to_numeric(rep, errors="coerce").to_numpy(float)
        err = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        err = np.where(np.isnan(err), np.inf, err)
        bad = err > atol + rtol * np.abs(np.nan_to_num(a))
        fin = err[np.isfinite(err)]
        return len(a), int(bad.sum()), float(err.max()) if len(err) else np.nan, float(fin.mean()) if len(fin) else np.nan
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
    per_ep = df.groupby("start_index")["step"].agg(["count", "min", "max"])
    bad = per_ep[(per_ep["count"] != STEPS_PER_EPISODE) | (per_ep["min"] != 0) | (per_ep["max"] != STEPS_PER_EPISODE - 1)]
    return dict(label=label, rows=len(df), duplicate_keys=int(df.duplicated(["start_index", "step"]).sum()),
                episodes=len(per_ep), incomplete_episodes=bad.index.tolist())


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
            n, bad, mx, mean = compare(both[f"{c}__src"], both[f"{c}__rep"], kind, args.atol, args.rtol)
            rows.append(dict(metric=c, kind=kind, required=required, in_source=True, in_replay=True, matched_steps=n,
                             n_exceed=bad, max_abs_err=mx, mean_abs_err=mean, status="PASS" if bad == 0 and n > 0 else "FAIL"))
    table = pd.DataFrame(rows)
    ok_keys = (not any(missing.values()) and not keys["source"]["duplicate_keys"] and not keys["replay"]["duplicate_keys"]
               and not keys["replay"]["incomplete_episodes"] and not keys["source"]["incomplete_episodes"])
    status = "PASS" if ok_keys and not table["status"].isin(["FAIL", "MISSING"]).any() and len(both) else "FAIL"
    net = man.get("network") or {}
    extra = dict(network_representation=dict(archived=net.get("archived_network"), topology_checks=net.get("topology_checks"),
                                             input_files_vs_original_run=net.get("input_files_vs_original_run"),
                                             regenerated_from_live_csvs=net.get("regenerated_from_live_csvs")),
                 initialisation=man.get("reset"))
    if not net.get("input_files_vs_original_run"):
        status, extra["network_problem"] = "FAIL", "replay manifest lacks the verified input-file identity"
    return status, table, keys, missing, len(starts), extra


def target(args, man, rep_df, policy, source_csv):
    actions, _, _ = load_source_actions(source_csv, args.model, args.expected_episodes)
    tgt = man["target_topology"]
    starts = sorted(rep_df["start_index"].unique())
    keys = dict(replay=key_report(rep_df, f"{tgt} replay"))
    bad_actions = sum(int(not np.array_equal(np.asarray(ast.literal_eval(r[RAW_FIELD]), dtype=np.float32),
                                             actions[int(r["start_index"])][int(r["step"])])) for _, r in rep_df.iterrows())
    rows = [dict(metric="replayed raw action == recorded TP1 raw action", matched_steps=len(rep_df), n_exceed=bad_actions,
                 status="PASS" if bad_actions == 0 and len(rep_df) else "FAIL")]
    calls_ok = int(man.get("actor_calls", -1)) == len(rep_df)
    rows.append(dict(metric="ReplayAgent calls == replayed steps (no new inference)", matched_steps=len(rep_df),
                     n_exceed=int(not calls_ok), status="PASS" if calls_ok else "FAIL"))
    fb_df, _ = read_filtered(source_csv, tgt, policy, ["start_index", "step", "date_time", "soc_before", "episode_id"])
    fb_df = fb_df[fb_df["start_index"].isin(starts)]
    keys["feedback"] = key_report(fb_df, f"{tgt} feedback (same final run)")
    m = fb_df.merge(rep_df, on=["start_index", "step"], how="outer", suffixes=("__fb", "__rep"), indicator=True)
    both = m[m["_merge"] == "both"]
    for c in ("date_time", "episode_id"):
        n, bad, _, _ = compare(both[f"{c}__fb"], both[f"{c}__rep"], "exact", 0, 0)
        rows.append(dict(metric=f"{c} equals {tgt} feedback", matched_steps=n, n_exceed=bad, status="PASS" if bad == 0 and n else "FAIL"))
    first = both[both["step"] == 0]
    if "soc_before__fb" in first and "soc_before__rep" in first:
        n, bad, mx, _ = compare(first["soc_before__fb"], first["soc_before__rep"], "vector_close", 0.0, 0.0)
        rows.append(dict(metric=f"initial SOC (step 0 soc_before) equals {tgt} feedback", matched_steps=n, n_exceed=bad,
                         max_abs_err=mx, status="PASS" if bad == 0 and n else "FAIL"))
    else:
        rows.append(dict(metric="initial SOC equals feedback", status="MISSING"))
    missing = dict(only_in_feedback=int((m["_merge"] == "left_only").sum()), only_in_replay=int((m["_merge"] == "right_only").sum()))
    table = pd.DataFrame(rows)
    ok_keys = (not any(missing.values()) and not keys["replay"]["duplicate_keys"] and not keys["replay"]["incomplete_episodes"]
               and not keys["feedback"]["duplicate_keys"])
    status = "PASS" if ok_keys and (table["status"] == "PASS").all() else "FAIL"
    net = man.get("network") or {}
    return status, table, keys, missing, len(starts), dict(topology_checks=net.get("topology_checks"),
                                                            archived_network=net.get("archived_network"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["identity", "target"])
    ap.add_argument("--model", required=True, choices=["gcn", "gat", "tagconv", "mlp"])
    ap.add_argument("--replay_dir", required=True)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--expected_episodes", type=int, default=31)
    args = ap.parse_args()
    rd = Path(args.replay_dir)
    man = json.loads((rd / "replay_manifest.json").read_text())
    if man.get("model") != args.model or man.get("feeder") != "ieee33":
        raise SystemExit(f"replay is {man.get('feeder')}/{man.get('model')}, not ieee33/{args.model}")
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
        status, table, keys, missing, n_eps, extra = (identity if args.kind == "identity" else target)(args, man, rep_df, policy, source_csv)
    except ReplayError as exc:
        status, table, keys, missing, n_eps, extra = "FAIL", pd.DataFrame(), {}, {}, 0, dict(error=str(exc))
    stamp = utc_stamp()
    table.to_csv(rd / f"validation_{args.kind}_{stamp}.csv", index=False)
    report = dict(feeder="ieee33", kind=args.kind, status=status, model=args.model, target_topology=tgt, replay_dir=str(rd.resolve()),
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
