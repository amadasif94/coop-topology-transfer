"""
Physical-clipping diagnostics for the IEEE33 TP1 -> TP6 action replay (read only; separate from the
original metric, which is NOT changed).

    python -u experiments/topology_generalization/diagnose_physical_clipping_ieee33.py \
        --pair gcn 0 <TP1 identity replay dir or -> <TP6 replay dir or -> ... --out <new dir>

How the ORIGINAL metric is computed (inspected source):
  environments/dess_env.py DESSEnv._step_impl
    marl_raw_action  = the action handed to env.step (actor output, normalized, float32)
    marl_prepared_action = _prepare_action(raw) = clip(raw, action.low, action.high) = clip(raw, -1, 1)
    marl_requested_power_kw[i] = prepared[i] * (max_discharge_kw if prepared[i] >= 0 else max_charge_kw)   [kW]
    physical_limited_power_kw[i] = Battery.step(prepared[i])["power_kw"] on a SOC snapshot (restored)   [kW]
        battery.py: discharge = min(requested, max_discharge_kw, (soc - soc_min)*capacity*eta_dis/dt)
                    charge    = min(|requested|, max_charge_kw, (soc_max - soc)*capacity/eta_ch/dt)
    physical_clipped_kw[i] = |marl_requested_power_kw[i] - physical_limited_power_kw[i]|      (line 1073)
    physical_clipped[i]    = physical_clipped_kw[i] > 1e-6 kW                                 (line 1075, explicit tolerance)
    physical_clipping_any  = any_i physical_clipped[i]                                        (line 1076, one flag per STEP)
  utility/evaluation_reporting.py diagnostic_summary: physical_clipping_rate = mean over steps of
    physical_clipping_any  -> the reported rate is the fraction of STEPS in which ANY of the 5 agents is clipped.
  Because requested power is built from the [-1,1]-clipped action, requested <= rating always, so the
  max_charge/max_discharge term never binds in that metric: it measures ENERGY/SOC feasibility only.
  Clipping of the raw output to the rating ([-1,1]) happens earlier (_prepare_action) and is NOT part of it.
  physical_limited_action_normalized (evaluation_reporting.step_fields) = _power_to_action(physical power)
    = power / (max_discharge_kw if power >= 0 else max_charge_kw): same normalized units as the raw action.

Diagnostics per controller / run / condition (original TP1, TP1 identity replay, TP6 fixed-action replay,
TP6 feedback), per agent-step unless stated:
  clip magnitude |raw - physical_limited_action_normalized| (mean, median, p95, max; all and clipped-only);
  its parts: rating part |raw - prepared| (raw outside [-1,1]) and SOC/energy part |prepared - physical|;
  fraction |raw| > 0.99; step fraction with any agent clipped (logged and recomputed, consistency check);
  per-agent clipping frequency; attribution: rating (raw outside [-1,1]), SOC lower bound (discharge request
  reduced), SOC upper bound (charge request reduced) - power-rating limits inside the physical stage cannot
  bind (requested <= rating) and are reported as such, not as a separate cause;
  stage differences |raw - physical|, |physical - DOE|, |DOE - executed|, |raw - executed| (normalized,
  executed power / rating), and the evaluator's |requested_action_agent - accepted_action_agent|
  (DOE-conditioned candidate vs backup-selected action);
  relationship: backup-activation and executed-violation rates for steps with / without clipping, mean
  step clip magnitude for backup / no-backup steps, Spearman correlation (step clip sum, backup activated).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.topology_generalization.replay_utils_ieee33 import (
    N_AGENTS, matrix, read_filtered, read_manifest, sha256_file, write_json)

TOL_KW = 1e-6
COLUMNS = ["start_index", "step", "policy_proposal_action_normalized", "marl_raw_action", "marl_prepared_action",
           "physical_limited_action_normalized", "doe_action_normalized", "doe_conditioned_action", "policy_requested_power_kw",
           "marl_requested_power_kw", "physical_limited_power_kw", "executed_battery_power_kw", "final_battery_power_kw",
           "physical_clipped", "physical_clipped_kw", "physical_clipping_any", "backup_activated", "executed_network_violation",
           "soc_before"] + [f"requested_action_agent_{i}" for i in range(N_AGENTS)] + [f"accepted_action_agent_{i}" for i in range(N_AGENTS)]


def first(df, *names):
    for n in names:
        if n in df:
            return n
    return None


def flag(s):
    return s.map(lambda v: {"True": 1.0, "False": 0.0}.get(v, v) if isinstance(v, str) else v).astype(float)


def bool_matrix(series):
    """JSON list of booleans per row (as logged by evaluation_fields) -> float matrix; bad rows NaN."""
    out = np.full((len(series), N_AGENTS), np.nan)
    for i, v in enumerate(series):
        try:
            x = json.loads(v) if isinstance(v, str) else None
        except json.JSONDecodeError:
            x = None
        if isinstance(x, list) and len(x) == N_AGENTS and all(isinstance(b, bool) for b in x):
            out[i] = np.asarray(x, float)
    return out


def ratings(run_dir):
    envs = read_manifest(run_dir).get("environments") or []
    b = (envs[0].get("battery_constraints") or []) if envs else []
    if len(b) != N_AGENTS:
        raise SystemExit(f"{run_dir}: battery constraints not in run manifest")
    return np.array([x["max_charge_kw"] for x in b], float), np.array([x["max_discharge_kw"] for x in b], float)


def stats(x):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=np.nan, median=np.nan, p95=np.nan, max=np.nan)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def diagnose(df, chg, dis):
    raw = matrix(df[first(df, "policy_proposal_action_normalized", "marl_raw_action")])
    prep = matrix(df["marl_prepared_action"]) if "marl_prepared_action" in df else np.clip(raw, -1, 1)
    phys_n = matrix(df["physical_limited_action_normalized"]) if "physical_limited_action_normalized" in df else np.full_like(raw, np.nan)
    doe_col = first(df, "doe_action_normalized", "doe_conditioned_action")
    doe_n = matrix(df[doe_col]) if doe_col else np.full_like(raw, np.nan)
    req_kw = matrix(df[first(df, "policy_requested_power_kw", "marl_requested_power_kw")]) if first(df, "policy_requested_power_kw", "marl_requested_power_kw") else np.full_like(raw, np.nan)
    phys_kw = matrix(df["physical_limited_power_kw"]) if "physical_limited_power_kw" in df else np.full_like(raw, np.nan)
    ex_col = first(df, "executed_battery_power_kw", "final_battery_power_kw")
    ex_kw = matrix(df[ex_col]) if ex_col else np.full_like(raw, np.nan)
    ex_n = np.where(ex_kw >= 0, ex_kw / dis, ex_kw / chg)
    clip = np.abs(raw - phys_n)
    clipped = np.abs(req_kw - phys_kw) > TOL_KW
    logged = bool_matrix(df["physical_clipped"]) if "physical_clipped" in df else None
    rating_part = np.abs(raw - prep)
    soc_lower = (prep > 0) & (phys_kw < req_kw - TOL_KW)
    soc_upper = (prep < 0) & (phys_kw > req_kw + TOL_KW)
    step_any = clipped.any(axis=1)
    out = dict(steps=len(df), agent_steps=int(np.isfinite(raw).all(axis=1).sum() * N_AGENTS),
               clip_magnitude_all=stats(clip), clip_magnitude_clipped_only=stats(clip[clipped]),
               rating_part=stats(rating_part), soc_energy_part=stats(np.abs(prep - phys_n)),
               frac_abs_raw_gt_0_99=float(np.nanmean(np.abs(raw) > 0.99)),
               frac_steps_any_clipped_recomputed=float(step_any.mean()) if len(df) else np.nan,
               frac_steps_any_clipped_logged=float(flag(df["physical_clipping_any"]).mean()) if "physical_clipping_any" in df else np.nan,
               logged_vs_recomputed_agent_flags_agree=(bool(np.array_equal(logged.astype(bool), clipped)) if logged is not None
                                                       and np.isfinite(logged).all() else None),
               per_agent_clip_frequency=clipped.mean(axis=0).round(6).tolist(),
               attribution=dict(rating_raw_outside_bounds=int((rating_part > 0).sum()), soc_lower_bound_discharge=int(soc_lower.sum()),
                                soc_upper_bound_charge=int(soc_upper.sum()), clipped_agent_steps=int(clipped.sum()),
                                unattributed=int((clipped & ~soc_lower & ~soc_upper).sum()),
                                note="power-rating limits cannot bind inside the physical stage (requested <= rating); "
                                     "rating clipping happens only when |raw| > 1 (_prepare_action)"),
               stage_abs_diff=dict(raw_to_physical=stats(clip), physical_to_doe=stats(np.abs(phys_n - doe_n)),
                                   doe_to_executed=stats(np.abs(doe_n - ex_n)), raw_to_executed=stats(np.abs(raw - ex_n))))
    req_cols = [f"requested_action_agent_{i}" for i in range(N_AGENTS)]
    acc_cols = [f"accepted_action_agent_{i}" for i in range(N_AGENTS)]
    if all(c in df for c in req_cols + acc_cols):
        out["evaluator_requested_vs_accepted"] = stats(np.abs(df[req_cols].to_numpy(float) - df[acc_cols].to_numpy(float)))
    if "backup_activated" in df:
        b = flag(df["backup_activated"]).to_numpy()
        step_clip = np.nansum(clip, axis=1)
        rel = dict(backup_rate_steps_clipped=float(b[step_any].mean()) if step_any.any() else np.nan,
                   backup_rate_steps_not_clipped=float(b[~step_any].mean()) if (~step_any).any() else np.nan,
                   mean_step_clip_backup=float(step_clip[b == 1].mean()) if (b == 1).any() else np.nan,
                   mean_step_clip_no_backup=float(step_clip[b == 0].mean()) if (b == 0).any() else np.nan,
                   spearman_step_clip_vs_backup=float(pd.Series(step_clip).corr(pd.Series(b), method="spearman")))
        if "executed_network_violation" in df:
            v = flag(df["executed_network_violation"]).to_numpy()
            rel.update(violation_rate_steps_clipped=float(np.nanmean(v[step_any])) if step_any.any() else np.nan,
                       violation_rate_steps_not_clipped=float(np.nanmean(v[~step_any])) if (~step_any).any() else np.nan)
        out["relationship"] = rel
    soc = matrix(df["soc_before"]) if "soc_before" in df else None
    if soc is not None:
        out["soc_before_at_soc_lower_events"] = stats(soc[soc_lower])
        out["soc_before_at_soc_upper_events"] = stats(soc[soc_upper])
    return out


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif not isinstance(v, list):
            out[f"{prefix}{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=4, action="append", required=True, metavar=("MODEL", "RUN", "IDENTITY_DIR", "TP6_REPLAY_DIR"),
                    help="use - for a replay directory that is not available yet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    results, prov = [], []
    for model, run, ident, tp6 in a.pair:
        dirs = [Path(d) for d in (ident, tp6) if d != "-"]
        if not dirs:
            raise SystemExit(f"{model} run {run}: at least one replay directory is needed (to locate the source run)")
        man = json.loads((dirs[0] / "replay_manifest.json").read_text())
        src_csv, src_dir, policy = Path(man["source"]["source_csv"]), Path(man["source"]["source_dir"]), man["policy_label"]
        chg, dis = ratings(src_dir)
        conds = [("original_TP1", src_csv, "TP1"), ("TP6_feedback", src_csv, "TP6")]
        if ident != "-":
            conds.append(("TP1_identity_replay", Path(ident) / "step_metrics.csv", "TP1"))
        if tp6 != "-":
            conds.append(("TP6_fixed_action_replay", Path(tp6) / "step_metrics.csv", "TP6"))
        for cond, csv, tp in conds:
            df, missing = read_filtered(csv, tp, policy, COLUMNS)
            res = dict(model=model, run=str(run), condition=cond, topology=tp, missing_columns=missing, **diagnose(df, chg, dis))
            results.append(res)
            prov.append(dict(model=model, run=str(run), condition=cond, file=str(csv.resolve()), sha256=sha256_file(csv)))
    pd.DataFrame([flatten(r) for r in results]).to_csv(out / "physical_clipping_diagnostics.csv", index=False)
    write_json(out / "physical_clipping_diagnostics.json", dict(results=results, inputs=prov, definition=__doc__))
    cols = ["model", "run", "condition", "frac_steps_any_clipped_logged", "frac_steps_any_clipped_recomputed",
            "frac_abs_raw_gt_0_99", "clip_magnitude_all.mean", "clip_magnitude_all.p95", "clip_magnitude_all.max",
            "attribution.soc_lower_bound_discharge", "attribution.soc_upper_bound_charge", "attribution.rating_raw_outside_bounds",
            "attribution.unattributed", "relationship.backup_rate_steps_clipped", "relationship.backup_rate_steps_not_clipped"]
    flat = pd.DataFrame([flatten(r) for r in results])
    pd.set_option("display.width", 250)
    print(flat[[c for c in cols if c in flat]].to_string(index=False))
    print("written", out.resolve())


if __name__ == "__main__":
    main()
