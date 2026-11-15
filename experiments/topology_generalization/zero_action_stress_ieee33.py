"""
Zero-action baseline completeness and pre-existing network stress for the IEEE33 final evaluations (read only).

    python -u experiments/topology_generalization/zero_action_stress_ieee33.py --out <new dir> \
        [--csv_root results/topology_generalization/csv] [--source_glob '*_ieee33_tp_full_eval_run*_best'] [--extra_dir <dir> ...]

1. Searches the original IEEE33 TP1-TP7 evaluation directories <csv_root>/<source_glob>/ (e.g.
   gcn_ieee33_tp_full_eval_run0_best) plus any --extra_dir (e.g. a later baselines-only run) for
   step_metrics.csv rows with policy == 'zero'.
2. Completeness per topology TP1..TP7: 31 completed December episodes x 96 dispatch steps, test split,
   final-test protocol, and configuration consistency (per-topology input-file sha256, seed and episode
   argument identical to the learned final runs found under the same root). Duplicated complete sources must
   agree day by day; the first consistent complete source is used. Nothing is re-simulated.
3. Network stress per topology from the zero-action rows: executed network / voltage / current violation
   steps, min / max bus voltage, max line loading (% of the 255 A rating) and max line current (p.u.), final
   network feasibility, power-flow convergence, backup activations, DOE interventions, and the evaluator's
   pre-existing-infeasibility indicators (physical_proposal_feasible, backup_zero_infeasible).
   Topologies are ranked by idle-battery infeasible steps (physical_proposal_feasible == False), then by
   executed violation steps.

Zero-action baseline vs the per-step pre-existing-infeasibility diagnostic:
  * The zero-action POLICY proposes 0 kW all day. Its physically limited proposal IS the idle-battery network
    (physical_proposal_feasible), but with DOE enabled the corrective DOE can still force non-zero battery
    power when a local voltage is outside [0.95, 1.05]; the executed zero-action state can therefore differ
    from the idle network (and SOC can drift from 0.5), and the backup can activate.
  * The evaluator's per-step diagnostic (backup_zero_score / backup_cause / backup_zero_infeasible) is the
    zero-battery reference solved by the directional backup at the SAME timestep of a given policy's run; it
    exists only on steps where the backup ran, is scored with the backup's aggregate score, and conditions on
    nothing else (zero DESS power, same load/PV). Its timestep-wise network is the same physical case as the
    zero-action proposal, but its coverage and classification are different, so the two are not equivalent
    summaries.
If records are missing, the exact existing-evaluator command for the missing topologies is printed (it runs
zero AND random baselines; --baselines_only does not load a checkpoint).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.topology_generalization.replay_utils_ieee33 import (
    DEFAULT_CSV_ROOT, EXPECTED_EPISODES, SOURCE_GLOB, STEPS_PER_EPISODE, input_shas, sha256_file, utc_stamp, write_json)

TOPOLOGIES = [f"TP{i}" for i in range(1, 8)]
COLS = ["topology_case", "policy", "episode_id", "start_index", "step", "record_type", "episode_completed", "energy_cost",
        "executed_network_violation", "executed_voltage_violation", "executed_current_violation", "min_voltage_pu",
        "max_voltage_pu", "executed_max_line_loading_pct", "max_line_current_pu", "final_network_feasible", "converged",
        "backup_activated", "doe_intervened", "physical_proposal_feasible", "backup_zero_infeasible", "backup_cause"]


def flag(s):
    return s.map(lambda v: {"True": 1.0, "False": 0.0}.get(v, v) if isinstance(v, str) else v).astype(float)


def read_zero(csv):
    header = set(pd.read_csv(csv, nrows=0).columns)
    keep = [c for c in COLS if c in header]
    parts = [c[c["policy"] == "zero"] for c in pd.read_csv(csv, usecols=keep, chunksize=20000, low_memory=False, dtype={"policy": str})]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)


def manifest(d):
    p = Path(d) / "run_manifest.json"
    return json.loads(p.read_text()) if p.is_file() else None


def env_shas(man, tp):
    for e in (man or {}).get("environments", []):
        if str(e.get("topology", "")).upper() == tp:
            return input_shas(e)
    return None


def stress(z):
    n = len(z)
    g = lambda c: flag(z[c]) if c in z else pd.Series(np.nan, index=z.index)
    num = lambda c: pd.to_numeric(z[c], errors="coerce") if c in z else pd.Series(np.nan, index=z.index)
    out = dict(steps=n,
               executed_network_violation_steps=int(g("executed_network_violation").sum()) if "executed_network_violation" in z else None,
               executed_voltage_violation_steps=int(g("executed_voltage_violation").sum()) if "executed_voltage_violation" in z else None,
               executed_current_violation_steps=int(g("executed_current_violation").sum()) if "executed_current_violation" in z else None,
               min_bus_voltage_pu=float(num("min_voltage_pu").min()), max_bus_voltage_pu=float(num("max_voltage_pu").max()),
               max_line_loading_pct=float(num("executed_max_line_loading_pct").max()) if "executed_max_line_loading_pct" in z else None,
               max_line_current_pu=float(num("max_line_current_pu").max()),
               final_network_feasible_rate=float(g("final_network_feasible").mean()), converged_rate=float(g("converged").mean()),
               backup_activation_steps=int(g("backup_activated").sum()) if "backup_activated" in z else None,
               doe_intervention_steps=int(g("doe_intervened").sum()) if "doe_intervened" in z else None,
               idle_battery_infeasible_steps=(int((g("physical_proposal_feasible") == 0).sum()) if "physical_proposal_feasible" in z else None),
               backup_zero_infeasible_steps=(int((g("backup_zero_infeasible") == 1).sum()) if "backup_zero_infeasible" in z else None),
               backup_cause_counts=z["backup_cause"].value_counts().to_dict() if "backup_cause" in z else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_root", default=str(DEFAULT_CSV_ROOT))
    ap.add_argument("--source_glob", default=SOURCE_GLOB)
    ap.add_argument("--extra_dir", action="append", default=[], help="additional evaluation directory to search (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = sorted({p.parent for p in Path(a.csv_root).glob(f"{a.source_glob}/step_metrics.csv")}
                  | {Path(d) for d in a.extra_dir if (Path(d) / "step_metrics.csv").is_file()})
    if not runs:
        raise SystemExit(f"no step_metrics.csv under {Path(a.csv_root) / a.source_glob} (or --extra_dir)")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    print("searched directories:", *[str(d) for d in runs], sep="\n  ")
    learned_ref = {}
    for d in runs:
        man = manifest(d)
        if not (man or {}).get("checkpoint_file") or ((man or {}).get("arguments") or {}).get("baselines_only"):
            continue                                              # reference configuration = the learned final runs
        for tp in TOPOLOGIES:
            s = env_shas(man, tp)
            if s:
                learned_ref.setdefault(tp, []).append(dict(dir=str(d), shas=s, seed=(man or {}).get("seed"),
                                                           episodes=((man or {}).get("arguments") or {}).get("episodes")))
    rows, chosen, problems = [], {}, []
    for d in runs:
        z = read_zero(d / "step_metrics.csv")
        if z.empty:
            continue
        man = manifest(d)
        for tp in TOPOLOGIES:
            zt = z[z["topology_case"] == tp]
            if zt.empty:
                continue
            disp = zt[zt.get("record_type", "dispatch_attempt") == "dispatch_attempt"] if "record_type" in zt else zt
            per_ep = disp.groupby("start_index")["step"].count()
            completed = int(disp.groupby("start_index")["episode_completed"].agg(lambda s: s.astype(str).eq("True").all()).sum()) \
                if "episode_completed" in disp else 0
            prot = (man or {}).get("evaluation_protocol") or {}
            shas = env_shas(man, tp)
            refs = learned_ref.get(tp, [])
            consistent = bool(shas) and all(r["shas"] == shas for r in refs) and len({r["seed"] for r in refs} | {(man or {}).get("seed")}) == 1
            complete = (len(per_ep) == EXPECTED_EPISODES and (per_ep == STEPS_PER_EPISODE).all() and completed == EXPECTED_EPISODES
                        and (man or {}).get("mode") == "test" and prot.get("final_test") is not False)
            rows.append(dict(topology=tp, source_dir=str(d), episodes=len(per_ep), completed_episodes=completed,
                             all_96_steps=bool((per_ep == STEPS_PER_EPISODE).all()) if len(per_ep) else False,
                             mode=(man or {}).get("mode"), final_test=prot.get("final_test"), seed=(man or {}).get("seed"),
                             consistent_with_learned_runs=consistent, learned_runs_compared=len(refs), complete=complete,
                             step_metrics_sha256=sha256_file(d / "step_metrics.csv")))
            if complete and consistent:
                cost = disp.groupby("start_index")["energy_cost"].sum()
                if tp in chosen:
                    if not np.allclose(chosen[tp]["cost"].to_numpy(), cost.reindex(chosen[tp]["cost"].index).to_numpy(), rtol=0, atol=1e-9):
                        problems.append(f"{tp}: zero-action sources {chosen[tp]['dir']} and {d} disagree day by day")
                else:
                    chosen[tp] = dict(dir=str(d), cost=cost, rows=disp)
    completeness = pd.DataFrame(rows)
    completeness.to_csv(out / "zero_action_completeness.csv", index=False)
    missing = [tp for tp in TOPOLOGIES if tp not in chosen]
    stress_rows = []
    for tp, c in chosen.items():
        stress_rows.append(dict(topology=tp, source_dir=c["dir"], mean_daily_cost=float(c["cost"].mean()), **stress(c["rows"])))
    stress_df = pd.DataFrame(stress_rows)
    if len(stress_df):
        stress_df = stress_df.sort_values([c for c in ("idle_battery_infeasible_steps", "executed_network_violation_steps")
                                           if c in stress_df], ascending=False, na_position="last")
        stress_df["stress_rank"] = range(1, len(stress_df) + 1)
    stress_df.to_csv(out / "zero_action_network_stress.csv", index=False)
    command = None
    if missing:
        name = "ieee33_tp_baselines_zero_random_" + utc_stamp()
        command = ("python -u experiments/topology_generalization/run_tp_eval.py --baselines_only --final --topologies "
                   + " ".join(missing) + f" --episodes all --run_name {name} --cpu"
                   + f"   # then re-run this script with --extra_dir {Path(a.csv_root) / name}")
    write_json(out / "zero_action_report.json", dict(searched=[str(d) for d in runs], topologies=TOPOLOGIES,
               complete_and_consistent=sorted(chosen), missing_or_incomplete=missing, problems=problems,
               command_for_missing=command, note=__doc__))
    pd.set_option("display.width", 250)
    print(completeness.to_string(index=False) if len(completeness) else "no zero-action rows found in the searched directories")
    print()
    print(stress_df.drop(columns=["backup_cause_counts", "source_dir"], errors="ignore").to_string(index=False) if len(stress_df) else "")
    if problems:
        print("PROBLEMS:", problems)
    if command:
        print("MISSING/INCOMPLETE:", missing, "\nrun (existing evaluator, original 31-day final protocol):\n" + command)
    print("written", out.resolve())


if __name__ == "__main__":
    main()
