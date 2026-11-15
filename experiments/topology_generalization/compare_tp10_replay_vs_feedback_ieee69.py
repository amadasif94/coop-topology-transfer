"""
Paired comparison on IEEE69 TP10: TP1 raw-action REPLAY minus TP10 normal FEEDBACK (read only).

    python -u experiments/topology_generalization/compare_tp10_replay_vs_feedback_ieee69.py \
        --pair mlp 0 <replay run dir> job:99786 \
        --pair gcn 0 <replay run dir> <feedback dir> ... \
        --out results/topology_generalization/csv_ieee69/action_replay/comparison/<name>

What is compared: closed-loop feedback (actions recomputed from TP10 observations) versus the recorded
open-loop TP1 action schedule applied to TP10. This does NOT by itself prove topology-specific
adaptation; it measures the value of feedback relative to the recorded open-loop schedule.

Per controller and training run: the two evaluations are aligned by December test day (start index);
only days completed in both are paired. Daily metrics (replay - feedback):
    energy cost, mean voltage deviation, backup-activation rate, raw-proposal infeasibility rate
    (raw_infeasible_requested) and physical-proposal feasibility rate (if recorded), minimum requested
    and executed voltage, battery throughput, terminal SOC per battery (soc_agent_i after the last step),
    total terminal stored energy (sum SOC_i x capacity_i, capacities from the run manifest).
Optional terminal-energy adjusted cost (--terminal_energy_valuation day_mean_price): cost minus the
change of stored energy over the day valued at that day's mean tariff (scaled price from the dataset
used by the environment), i.e. energy left in the batteries is credited at the average price of the
day. This is a sensitivity convention, reported separately, never replacing raw cost.
Uncertainty: per training run, bootstrap over matched DAYS as blocks (n_boot, fixed seed). Pooled per
controller: two-level bootstrap (resample training runs with replacement, then days within each run;
statistic = mean of run means). With three runs this interval is coarse and descriptive.
Replay runs must carry a PASS 'target' validation covering all episodes (--allow_unvalidated to override,
labelled UNVALIDATED).
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
    N_AGENTS, STEPS_PER_EPISODE, read_filtered, read_manifest, resolve_feedback_dir, sha256_file, write_json)

METRICS = ["energy_cost", "mean_voltage_deviation", "backup_activation_rate", "raw_infeasible_rate",
           "physical_proposal_feasible_rate", "min_requested_voltage_pu", "min_executed_voltage_pu",
           "throughput_mwh", "terminal_energy_kwh"] + [f"terminal_soc_agent_{i}" for i in range(N_AGENTS)]
COLUMNS = ["start_index", "step", "date_time", "episode_completed", "energy_cost", "voltage_deviation", "backup_activated",
           "raw_infeasible_requested", "physical_proposal_feasible", "requested_min_voltage_pu", "min_voltage_pu",
           "throughput_mwh", "soc_before"] + [f"soc_agent_{i}" for i in range(N_AGENTS)]


def num(s):
    if s.dtype == object:
        s = s.map(lambda v: {"True": 1.0, "False": 0.0}.get(v, v) if isinstance(v, str) else v)
    return pd.to_numeric(s, errors="coerce")


def capacities(run_dir):
    envs = read_manifest(run_dir).get("environments") or []
    caps = [b.get("capacity_kwh") for b in (envs[0].get("battery_constraints") or [])] if envs else []
    return np.asarray(caps, dtype=float) if len(caps) == N_AGENTS and all(c is not None for c in caps) else None


def day_prices(starts):
    from configs.ieee69_config import IEEE69_CONFIG
    from utility.data_loader import load_processed_node_time_series
    ts = load_processed_node_time_series(IEEE69_CONFIG["paths"]["processed_time_series"], 69, allow_empty=False)
    scale = float(IEEE69_CONFIG.get("data_scaling", {}).get("price_scaling_factor", 1.0))
    price = ts["df"][ts["price_col"]].to_numpy(float) * scale
    return {s: float(price[s:s + STEPS_PER_EPISODE].mean()) for s in starts}


def day_table(df, caps):
    out = []
    for start, g in df.groupby("start_index"):
        g = g.sort_values("step")
        complete = len(g) == STEPS_PER_EPISODE and num(g["episode_completed"]).fillna(0).eq(1).all()
        last = g.iloc[-1]
        soc_t = np.array([float(last[f"soc_agent_{i}"]) for i in range(N_AGENTS)]) if all(f"soc_agent_{i}" in g for i in range(N_AGENTS)) else None
        soc_0 = np.asarray(ast.literal_eval(g["soc_before"].iloc[0]), dtype=float) if "soc_before" in g and isinstance(g["soc_before"].iloc[0], str) else None
        row = dict(start_index=int(start), date=str(g["date_time"].iloc[0])[:10], complete=bool(complete),
                   energy_cost=float(num(g["energy_cost"]).sum()), mean_voltage_deviation=float(num(g["voltage_deviation"]).mean()),
                   backup_activation_rate=float(num(g["backup_activated"]).mean()) if "backup_activated" in g else np.nan,
                   raw_infeasible_rate=float(num(g["raw_infeasible_requested"]).mean()) if "raw_infeasible_requested" in g else np.nan,
                   physical_proposal_feasible_rate=float(num(g["physical_proposal_feasible"]).mean()) if "physical_proposal_feasible" in g else np.nan,
                   min_requested_voltage_pu=float(num(g["requested_min_voltage_pu"]).min()) if "requested_min_voltage_pu" in g else np.nan,
                   min_executed_voltage_pu=float(num(g["min_voltage_pu"]).min()), throughput_mwh=float(num(g["throughput_mwh"]).sum()))
        for i in range(N_AGENTS):
            row[f"terminal_soc_agent_{i}"] = float(soc_t[i]) if soc_t is not None else np.nan
        row["terminal_energy_kwh"] = float(soc_t @ caps) if (soc_t is not None and caps is not None) else np.nan
        row["initial_energy_kwh"] = float(soc_0 @ caps) if (soc_0 is not None and caps is not None and soc_0.size == N_AGENTS) else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("start_index")


def boot_days(d, n_boot, rng):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n_days=0, mean=np.nan, median=np.nan, ci95_lo=np.nan, ci95_hi=np.nan)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return dict(n_days=len(d), mean=float(d.mean()), median=float(np.median(d)),
                ci95_lo=float(np.percentile(means, 2.5)), ci95_hi=float(np.percentile(means, 97.5)))


def boot_hierarchical(per_run, n_boot, rng):
    runs = [np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for v in per_run]
    runs = [r for r in runs if len(r)]
    if not runs:
        return dict(n_runs=0)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(runs), len(runs))
        stats[b] = np.mean([runs[i][rng.integers(0, len(runs[i]), len(runs[i]))].mean() for i in pick])
    return dict(n_runs=len(runs), n_days_total=int(sum(len(r) for r in runs)), mean_of_run_means=float(np.mean([r.mean() for r in runs])),
                median_of_run_means=float(np.median([r.mean() for r in runs])), run_means=[float(r.mean()) for r in runs],
                ci95_lo=float(np.percentile(stats, 2.5)), ci95_hi=float(np.percentile(stats, 97.5)),
                method="two-level bootstrap: training runs with replacement, then days within each run; mean of run means")


def validated(replay_dir, expected):
    reps = sorted(Path(replay_dir).glob("validation_target_*.json"))
    for p in reversed(reps):
        r = json.loads(p.read_text())
        if r.get("status") == "PASS" and int(r.get("episodes_validated", 0)) >= expected:
            return str(p)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=4, action="append", required=True, metavar=("MODEL", "RUN", "REPLAY_DIR", "FEEDBACK"),
                    help="FEEDBACK = TP10 feedback result dir or job:<final_test_learned job id>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=20261024)
    ap.add_argument("--expected_episodes", type=int, default=31)
    ap.add_argument("--terminal_energy_valuation", choices=["none", "day_mean_price"], default="none")
    ap.add_argument("--allow_unvalidated", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.boot_seed)
    per_day, seed_rows, pairs_meta, limitations = [], [], [], []
    seen = set()
    for model, run, replay_dir, fb in a.pair:
        key = (model, str(run))
        if key in seen:
            raise SystemExit(f"duplicate pair {key}")
        seen.add(key)
        replay_dir = Path(replay_dir)
        fb_dir = resolve_feedback_dir(fb.split(":", 1)[1]) if fb.startswith("job:") else Path(fb)
        man = json.loads((replay_dir / "replay_manifest.json").read_text())
        if man.get("model") != model or man.get("target_topology") != "TP10":
            raise SystemExit(f"{replay_dir}: replay is {man.get('model')}/{man.get('target_topology')}, not {model}/TP10")
        if str(man.get("training_run")) not in (str(run), f"run_{run}", "None"):
            raise SystemExit(f"{replay_dir}: training run {man.get('training_run')} != {run}")
        if Path((man.get("network") or {}).get("feedback_dir", "")).resolve() != fb_dir.resolve():
            raise SystemExit(f"{replay_dir}: replay was matched to feedback {man.get('network', {}).get('feedback_dir')}, not {fb_dir}")
        val = validated(replay_dir, a.expected_episodes)
        if val is None and not a.allow_unvalidated:
            raise SystemExit(f"{replay_dir}: no PASS full 'target' validation report (run the validator first)")
        policy = man["policy_label"]
        rep, _ = read_filtered(replay_dir / "step_metrics.csv", "TP10", policy, COLUMNS)
        fbk, fb_missing = read_filtered(fb_dir / "step_metrics.csv", "TP10", policy, COLUMNS)
        caps = capacities(replay_dir)
        if caps is None:
            limitations.append(f"{model} run {run}: battery capacities not in manifest; terminal energy not computed")
        r_days, f_days = day_table(rep, caps), day_table(fbk, caps)
        common = r_days.index.intersection(f_days.index)
        ok = [s for s in common if r_days.loc[s, "complete"] and f_days.loc[s, "complete"]]
        if set(r_days.index) != set(f_days.index):
            limitations.append(f"{model} run {run}: day sets differ (replay {len(r_days)}, feedback {len(f_days)})")
        diff = (r_days.loc[ok, METRICS] - f_days.loc[ok, METRICS])
        if a.terminal_energy_valuation == "day_mean_price":
            prices = day_prices(ok)
            adj = lambda d: d.loc[ok, "energy_cost"] - pd.Series(prices) * (d.loc[ok, "terminal_energy_kwh"] - d.loc[ok, "initial_energy_kwh"]) / 1000.0
            diff["terminal_energy_adjusted_cost"] = adj(r_days) - adj(f_days)
        diff = diff.assign(model=model, run=str(run), date=r_days.loc[ok, "date"]).reset_index()
        per_day.append(diff)
        for m in [c for c in diff.columns if c not in ("start_index", "model", "run", "date")]:
            seed_rows.append(dict(model=model, run=str(run), metric=m, **boot_days(diff[m], a.n_boot, rng)))
        pairs_meta.append(dict(model=model, run=str(run), replay_dir=str(replay_dir.resolve()), feedback_dir=str(fb_dir.resolve()),
                               replay_step_metrics_sha256=sha256_file(replay_dir / "step_metrics.csv"),
                               feedback_step_metrics_sha256=sha256_file(fb_dir / "step_metrics.csv"),
                               validation_report=val or "UNVALIDATED", matched_complete_days=len(ok),
                               replay_days=len(r_days), feedback_days=len(f_days), feedback_missing_columns=fb_missing,
                               selection_id=(man.get("selection") or {}).get("selection_id")))
    days = pd.concat(per_day, ignore_index=True)
    seeds = pd.DataFrame(seed_rows)
    model_rows = []
    for model, g in days.groupby("model"):
        for m in [c for c in days.columns if c not in ("start_index", "model", "run", "date")]:
            model_rows.append(dict(model=model, metric=m, **boot_hierarchical([gg[m].to_numpy() for _, gg in g.groupby("run")], a.n_boot, rng)))
    models = pd.DataFrame(model_rows)
    days.to_csv(out / "per_day_differences.csv", index=False)
    seeds.to_csv(out / "run_summary.csv", index=False)
    models.to_csv(out / "model_pooled_summary.csv", index=False)
    write_json(out / "comparison.json", dict(
        definition="replay minus feedback, per matched December test day, TP10", n_boot=a.n_boot, boot_seed=a.boot_seed,
        bootstrap="per run: days resampled as blocks; per model: two-level (runs, then days)",
        terminal_energy_valuation=a.terminal_energy_valuation, pairs=pairs_meta, limitations=limitations,
        interpretation=("measures closed-loop feedback versus a recorded open-loop TP1 action schedule on TP10; it does not by "
                        "itself prove topology-specific adaptation; three training runs per controller give coarse uncertainty"),
        created_utc=datetime.now(timezone.utc).isoformat()))
    pd.set_option("display.width", 200)
    print(models.drop(columns=["run_means", "method"], errors="ignore").to_string(index=False))
    print("written", out.resolve())


if __name__ == "__main__":
    main()
