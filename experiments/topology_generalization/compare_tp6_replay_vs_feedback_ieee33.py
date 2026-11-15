"""
Paired comparison on IEEE33 TP6: TP1 fixed-action REPLAY minus TP6 closed-loop FEEDBACK (read only).

    python -u experiments/topology_generalization/compare_tp6_replay_vs_feedback_ieee33.py \
        --pair gcn 0 <TP6 full replay dir> --pair gcn 1 <dir> ... (12 pairs) --out <new dir>

Feedback = the TP6 rows of the SAME final-test run that supplied the TP1 actions (same selection,
checkpoint, network inputs, reset seeds and December days; taken from the replay manifest).

Sign convention:  delta = replay - feedback  (per matched December day).
  cost / adjusted cost : delta > 0  -> fixed-action replay is MORE expensive.
  total reward         : delta < 0  -> fixed-action replay has LOWER reward.
  rates / counts       : delta > 0  -> more frequent under replay.

Daily metrics (both conditions, identical definitions):
  energy_cost (sum of evaluator energy_cost); stored_energy_adjusted_cost = energy_cost - p_day *
  (E_terminal - E_initial) / 1000 with p_day = mean scaled tariff of the day's 96 steps and
  E = sum_i SOC_i * capacity_i [kWh] (same equation and sign as the IEEE69 comparison: energy left
  stored is credited at the day's mean price); terminal_energy_kwh; total_reward (sum of reward_sum,
  team reward); mean_voltage_deviation; min_candidate_voltage_pu = minimum voltage of the AC power flow
  of the DOE-conditioned PRE-BACKUP candidate dispatch (evaluator field requested_min_voltage_pu if present,
  else backup_initial_min_voltage; a solved candidate network, NOT the executed network and NOT a pre-action
  voltage); min_executed_voltage_pu (min_voltage_pu, executed network); backup_activation_rate;
  raw_proposal_feasible_rate (physical_proposal_feasible: AC PF of the physically limited, pre-DOE proposal);
  executed network / voltage / current violation counts and rates (executed_*_violation);
  throughput_mwh; requested_to_executed_abs (mean over agent-steps of |raw action - executed power/rating|);
  clip_magnitude (mean |raw - physical_limited_action_normalized|); clipping_step_rate (physical_clipping_any).
Uncertainty (fixed seed, n_boot replicates), never read as significance tests:
  per run          : bootstrap over the 31 matched days (episode alignment preserved).
  per architecture : (a) day-level CONDITIONAL interval - the same resampled day set applied to all runs,
                         statistic = mean over runs (conditional on these trained runs);
                     (b) between-run variability - run means, their SD and range (3 runs);
                     (c) two-level bootstrap - runs with replacement, then days within run (as IEEE69).
                     93 run-days are never treated as 93 independent training runs.
Every replay must carry a PASS full 'target' validation and the identical set of 31 days across all pairs.
"""

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.topology_generalization.replay_utils_ieee33 import (
    EXPECTED_EPISODES, N_AGENTS, STEPS_PER_EPISODE, matrix, read_filtered, read_manifest, sha256_file, write_json)

COLUMNS = (["start_index", "step", "date_time", "episode_completed", "energy_cost", "reward_sum", "voltage_deviation",
            "requested_min_voltage_pu", "backup_initial_min_voltage", "min_voltage_pu", "backup_activated", "physical_proposal_feasible",
            "executed_network_violation", "executed_voltage_violation", "executed_current_violation", "throughput_mwh",
            "policy_proposal_action_normalized", "marl_raw_action", "physical_limited_action_normalized", "executed_battery_power_kw",
            "final_battery_power_kw", "physical_clipping_any", "soc_before"] + [f"soc_agent_{i}" for i in range(N_AGENTS)])
METRICS = ["energy_cost", "stored_energy_adjusted_cost", "terminal_energy_kwh", "total_reward", "mean_voltage_deviation",
           "min_candidate_voltage_pu", "min_executed_voltage_pu", "backup_activation_rate", "raw_proposal_feasible_rate",
           "executed_network_violation_count", "executed_network_violation_rate", "executed_voltage_violation_count",
           "executed_current_violation_count", "throughput_mwh", "requested_to_executed_abs", "clip_magnitude", "clipping_step_rate"]


def num(s):
    if s.dtype == object:
        s = s.map(lambda v: {"True": 1.0, "False": 0.0}.get(v, v) if isinstance(v, str) else v)
    return pd.to_numeric(s, errors="coerce")


def battery(run_dir):
    envs = read_manifest(run_dir).get("environments") or []
    b = (envs[0].get("battery_constraints") or []) if envs else []
    if len(b) != N_AGENTS:
        raise SystemExit(f"{run_dir}: battery constraints not in the run manifest")
    return (np.array([x["capacity_kwh"] for x in b], float), np.array([x["max_charge_kw"] for x in b], float),
            np.array([x["max_discharge_kw"] for x in b], float))


def day_prices(starts):
    from configs.ieee33_config import IEEE33_CONFIG
    from utility.data_loader import load_processed_node_time_series
    ts = load_processed_node_time_series(IEEE33_CONFIG["paths"]["processed_time_series"], 33, allow_empty=False)
    scale = float(IEEE33_CONFIG.get("data_scaling", {}).get("price_scaling_factor", 1.0))
    price = ts["df"][ts["price_col"]].to_numpy(float) * scale
    return {s: float(price[s:s + STEPS_PER_EPISODE].mean()) for s in starts}


def first(df, *names):
    return next((n for n in names if n in df), None)


def day_table(df, cap, chg, dis, prices):
    rows = []
    for start, g in df.groupby("start_index"):
        g = g.sort_values("step")
        complete = len(g) == STEPS_PER_EPISODE and num(g["episode_completed"]).fillna(0).eq(1).all()
        soc_t = np.array([float(g.iloc[-1][f"soc_agent_{i}"]) for i in range(N_AGENTS)])
        soc_0 = np.asarray(ast.literal_eval(g["soc_before"].iloc[0]), float) if "soc_before" in g else np.full(N_AGENTS, np.nan)
        e_t, e_0 = float(soc_t @ cap), float(soc_0 @ cap)
        cost = float(num(g["energy_cost"]).sum())
        raw = matrix(g[first(g, "policy_proposal_action_normalized", "marl_raw_action")])
        exk = matrix(g[first(g, "executed_battery_power_kw", "final_battery_power_kw")])
        exn = np.where(exk >= 0, exk / dis, exk / chg)
        phys = matrix(g["physical_limited_action_normalized"]) if "physical_limited_action_normalized" in g else np.full_like(raw, np.nan)
        cand = first(g, "requested_min_voltage_pu", "backup_initial_min_voltage")
        viol = lambda c: num(g[c]) if c in g else pd.Series(np.nan, index=g.index)
        rows.append(dict(start_index=int(start), date=str(g["date_time"].iloc[0])[:10], complete=bool(complete),
                         energy_cost=cost, stored_energy_adjusted_cost=cost - prices[int(start)] * (e_t - e_0) / 1000.0,
                         terminal_energy_kwh=e_t, initial_energy_kwh=e_0, total_reward=float(num(g["reward_sum"]).sum()),
                         mean_voltage_deviation=float(num(g["voltage_deviation"]).mean()),
                         min_candidate_voltage_pu=float(num(g[cand]).min()) if cand else np.nan,
                         min_executed_voltage_pu=float(num(g["min_voltage_pu"]).min()),
                         backup_activation_rate=float(num(g["backup_activated"]).mean()),
                         raw_proposal_feasible_rate=float(viol("physical_proposal_feasible").mean()),
                         executed_network_violation_count=float(viol("executed_network_violation").sum(min_count=1)),
                         executed_network_violation_rate=float(viol("executed_network_violation").mean()),
                         executed_voltage_violation_count=float(viol("executed_voltage_violation").sum(min_count=1)),
                         executed_current_violation_count=float(viol("executed_current_violation").sum(min_count=1)),
                         throughput_mwh=float(num(g["throughput_mwh"]).sum()),
                         requested_to_executed_abs=float(np.nanmean(np.abs(raw - exn))),
                         clip_magnitude=float(np.nanmean(np.abs(raw - phys))),
                         clipping_step_rate=float(viol("physical_clipping_any").mean()),
                         candidate_voltage_field=cand))
    return pd.DataFrame(rows).set_index("start_index")


def boot_mean(d, n_boot, rng):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n_days=0, mean=np.nan, median=np.nan, ci95_lo=np.nan, ci95_hi=np.nan)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return dict(n_days=len(d), mean=float(d.mean()), median=float(np.median(d)), ci95_lo=float(np.percentile(m, 2.5)),
                ci95_hi=float(np.percentile(m, 97.5)))


def pooled(mat, n_boot, rng):
    """mat: runs x days (aligned days). (a) conditional day bootstrap, (b) between-run spread, (c) two-level bootstrap."""
    mat = np.asarray(mat, float)
    runs, days = mat.shape
    run_means = np.nanmean(mat, axis=1)
    idx = rng.integers(0, days, (n_boot, days))
    cond = np.nanmean(np.nanmean(mat[:, idx], axis=2), axis=0)            # same day resample for every run
    two = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, runs, runs)
        two[b] = np.mean([np.nanmean(mat[i, rng.integers(0, days, days)]) for i in pick])
    return dict(n_runs=runs, n_days=days, mean_of_run_means=float(np.mean(run_means)), run_means=run_means.round(6).tolist(),
                between_run_sd=float(np.std(run_means, ddof=1)) if runs > 1 else np.nan,
                between_run_range=[float(run_means.min()), float(run_means.max())],
                day_conditional_ci95=[float(np.percentile(cond, 2.5)), float(np.percentile(cond, 97.5))],
                two_level_ci95=[float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))])


def validated(replay_dir, expected):
    for p in sorted(Path(replay_dir).glob("validation_target_*.json"), reverse=True):
        r = json.loads(p.read_text())
        if r.get("status") == "PASS" and int(r.get("episodes_validated", 0)) >= expected:
            return str(p)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("MODEL", "RUN", "TP6_REPLAY_DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=20261025)
    ap.add_argument("--expected_episodes", type=int, default=EXPECTED_EPISODES)
    ap.add_argument("--allow_unvalidated", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.boot_seed)
    per_day, prov, day_sets = [], [], {}
    for model, run, rdir in a.pair:
        rdir = Path(rdir)
        man = json.loads((rdir / "replay_manifest.json").read_text())
        if man.get("feeder") != "ieee33" or man.get("model") != model or man.get("target_topology") != "TP6":
            raise SystemExit(f"{rdir}: replay is {man.get('feeder')}/{man.get('model')}/{man.get('target_topology')}")
        if str(man.get("training_run")) not in (str(run), f"run_{run}"):
            raise SystemExit(f"{rdir}: selection run {man.get('training_run')} != {run}")
        val = validated(rdir, a.expected_episodes)
        if val is None and not a.allow_unvalidated:
            raise SystemExit(f"{rdir}: no PASS full 'target' validation report")
        src_dir, src_csv, policy = Path(man["source"]["source_dir"]), Path(man["source"]["source_csv"]), man["policy_label"]
        if sha256_file(src_csv) != man["source"]["source_sha256"]:
            raise SystemExit(f"{src_csv} changed since the replay")
        cap, chg, dis = battery(src_dir)
        rep, _ = read_filtered(rdir / "step_metrics.csv", "TP6", policy, COLUMNS)
        fbk, fb_missing = read_filtered(src_csv, "TP6", policy, COLUMNS)
        prices = day_prices(sorted(set(rep["start_index"].astype(int)) | set(fbk["start_index"].astype(int))))
        r_days, f_days = day_table(rep, cap, chg, dis, prices), day_table(fbk, cap, chg, dis, prices)
        ok = sorted(s for s in r_days.index.intersection(f_days.index) if r_days.loc[s, "complete"] and f_days.loc[s, "complete"])
        day_sets[(model, run)] = tuple(ok)
        diff = (r_days.loc[ok, METRICS] - f_days.loc[ok, METRICS]).assign(model=model, run=str(run), date=r_days.loc[ok, "date"])
        for m in METRICS:
            diff[f"replay_{m}"], diff[f"feedback_{m}"] = r_days.loc[ok, m], f_days.loc[ok, m]
        per_day.append(diff.reset_index())
        prov.append(dict(model=model, run=str(run), selection=man.get("selection"), source_dir=str(src_dir.resolve()),
                         source_step_metrics_sha256=man["source"]["source_sha256"],
                         source_manifest_sha256=man["source"].get("source_manifest_sha256"),
                         replay_dir=str(rdir.resolve()), replay_step_metrics_sha256=sha256_file(rdir / "step_metrics.csv"),
                         replay_manifest_sha256=sha256_file(rdir / "replay_manifest.json"), target_validation=val or "UNVALIDATED",
                         identity_gate=man.get("gates"), archived_network=(man.get("network") or {}).get("archived_network"),
                         input_files=man.get("target_input_files"), reset=man.get("reset"), matched_days=len(ok),
                         replay_days=len(r_days), feedback_days=len(f_days), feedback_missing_columns=fb_missing,
                         candidate_voltage_field=str(r_days["candidate_voltage_field"].iloc[0]) if len(r_days) else None))
    if len(set(day_sets.values())) != 1:
        raise SystemExit(f"pairs do not cover the identical set of days: { {k: len(v) for k, v in day_sets.items()} }")
    days = pd.concat(per_day, ignore_index=True)
    run_rows, model_rows = [], []
    for (model, run), g in days.groupby(["model", "run"]):
        for m in METRICS:
            run_rows.append(dict(model=model, run=run, metric=m, replay_mean=float(g[f"replay_{m}"].mean()),
                                 feedback_mean=float(g[f"feedback_{m}"].mean()), **boot_mean(g[m], a.n_boot, rng)))
    for model, g in days.groupby("model"):
        piv = {m: g.pivot(index="run", columns="start_index", values=m) for m in METRICS}
        for m in METRICS:
            model_rows.append(dict(model=model, metric=m, **pooled(piv[m].to_numpy(), a.n_boot, rng)))
    days.to_csv(out / "per_day_paired_differences.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out / "per_run_summary.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out / "architecture_pooled_summary.csv", index=False)
    write_json(out / "comparison.json", dict(definition="delta = replay - feedback (TP6, matched December days)",
               sign_convention=dict(cost="delta > 0: replay more expensive", reward="delta < 0: replay lower reward"),
               stored_energy_adjustment="cost - p_day_mean * (E_terminal - E_initial)/1000 (IEEE69 comparison equation)",
               candidate_voltage="pre-backup DOE-conditioned candidate AC PF minimum voltage; not executed, not pre-action",
               n_boot=a.n_boot, boot_seed=a.boot_seed, days=list(next(iter(day_sets.values()))),
               uncertainty="per run: day bootstrap; per architecture: day-conditional, between-run spread, two-level bootstrap",
               interpretation="intervals are descriptive; they are not significance tests", created_utc=datetime.now(timezone.utc).isoformat()))
    write_json(out / "provenance.json", dict(pairs=prov, outputs=dict(per_day=str((out / "per_day_paired_differences.csv").resolve()),
                                                                      per_run=str((out / "per_run_summary.csv").resolve()),
                                                                      pooled=str((out / "architecture_pooled_summary.csv").resolve()))))
    pd.set_option("display.width", 250)
    show = pd.DataFrame(model_rows).drop(columns=["run_means"])
    print(show.to_string(index=False))
    print("written", out.resolve())


if __name__ == "__main__":
    main()
