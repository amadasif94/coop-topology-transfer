"""
IEEE33 configuration actually used by the final evaluations and by the action replay (read only).

    python -u experiments/topology_generalization/inspect_ieee33_replay_config.py --out <new dir> \
        [--csv_root results/topology_generalization/csv] [--source_glob '*_ieee33_tp_full_eval_run*_best'] [--no_saved_runs]

Saved final runs are discovered as <csv_root>/<source_glob>/run_manifest.json (the original IEEE33 TP1-TP7
evaluation directories, e.g. gcn_ieee33_tp_full_eval_run0_best); the list of discovered directories is
reported, and an empty discovery is an error unless --no_saved_runs is given.

Reports, from code (configs/ieee33_config.py, environments/primary_marl.py) and from the EFFECTIVE
environment built exactly like the evaluators (make_primary_env on the validation split - no test data),
and from every discovered original evaluation run_manifest.json:
  DESS capacity, charge/discharge power, efficiencies, SOC min/max/initial, number and buses of DESS;
  control timestep and episode length; voltage limits; effective line-current ratings (A); DOE switch and
  parameters; backup mode, PF budget, refinement steps, scales, residual penalty; reward penalties;
  constraint handling; and every difference between the base configuration and each saved final run.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments.topology_generalization.replay_utils_ieee33 import DEFAULT_CSV_ROOT, SOURCE_GLOB, write_json


def env_summary(env):
    from environments.primary_marl import effective_current_ratings_a
    amps = effective_current_ratings_a(env)
    b = [vars(x.config) if not hasattr(x.config, "__dataclass_fields__") else {k: getattr(x.config, k) for k in x.config.__dataclass_fields__}
         for x in env.batteries]
    return dict(dess_buses_zero_based=[int(x) for x in env.dess_buses], dess_buses_one_based=[int(x) + 1 for x in env.dess_buses],
                num_dess=int(env.num_agents), battery_constraints=b, delta_t_hours=float(env.delta_t_hours),
                episode_length=int(env.episode_length), v_min=float(env.v_min), v_max=float(env.v_max),
                current_ratings_a=dict(min=float(amps.min()), max=float(amps.max()), n_active=int(amps.size)),
                doe=dict(enabled=bool(env.doe_enabled), w=env.doe_w, delta_c=env.doe_delta_c, pf_failure_mode=env.doe_pf_failure_mode,
                         penalty_coef=getattr(env, "doe_penalty_coef", None)),
                backup_mode=env.backup_mode, backup=dict(env.backup_cfg), reward=dict(env.reward_cfg),
                constraints=dict(env.constraint_cfg), grid_export_limit_kw=float(env.grid_export_limit_kw))


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not np.isclose(a, b, rtol=0, atol=1e-12):
            out.append(dict(field=path, base=a, saved=b))
    elif a != b:
        out.append(dict(field=path, base=a, saved=b))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_root", default=str(DEFAULT_CSV_ROOT))
    ap.add_argument("--source_glob", default=SOURCE_GLOB)
    ap.add_argument("--no_saved_runs", action="store_true", help="report the code/effective configuration only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifests = [] if a.no_saved_runs else sorted(Path(a.csv_root).glob(f"{a.source_glob}/run_manifest.json"))
    if not a.no_saved_runs and not manifests:
        raise SystemExit(f"no run_manifest.json found under {Path(a.csv_root) / a.source_glob}")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    from configs.ieee33_config import IEEE33_CONFIG
    from environments.primary_marl import make_primary_env, primary_config
    effective = env_summary(make_primary_env(copy.deepcopy(IEEE33_CONFIG), mode="val", seed=0))
    base_cfg = primary_config(copy.deepcopy(IEEE33_CONFIG))
    report = dict(effective_environment=effective,
                  base_config_sections={k: base_cfg[k] for k in ("dess", "time", "network", "doe", "backup", "constraints", "reward")},
                  line_current_note=("configured line_current_limit_mode / rho are reported as configured; the effective ratings "
                                     "above (and in every run manifest) are what the environment enforces"),
                  discovered_source_dirs=[str(m.parent) for m in manifests], saved_final_runs=[])
    if manifests:
        for man_path in manifests:
            man = json.loads(man_path.read_text())
            for e in man.get("environments", []):
                saved = dict(battery_constraints=e.get("battery_constraints"), doe=e.get("doe"), backup_mode=e.get("backup_mode"),
                             backup=e.get("backup"), network=e.get("network"), reward=e.get("reward"),
                             current_ratings_a=dict(min=e.get("min_current_rating_a"), max=e.get("max_current_rating_a")))
                ref = dict(battery_constraints=effective["battery_constraints"],
                           doe=dict(enabled=effective["doe"]["enabled"], w=effective["doe"]["w"], delta_c=effective["doe"]["delta_c"],
                                    pf_failure_mode=effective["doe"]["pf_failure_mode"], penalty_coef=effective["doe"]["penalty_coef"]),
                           backup_mode=effective["backup_mode"], backup=effective["backup"], network=json.loads(json.dumps(base_cfg["network"], default=str)),
                           reward=effective["reward"], current_ratings_a=dict(min=effective["current_ratings_a"]["min"],
                                                                              max=effective["current_ratings_a"]["max"]))
                report["saved_final_runs"].append(dict(manifest=str(man_path), controller=man.get("controller"), mode=man.get("mode"),
                                                       seed=man.get("seed"), gnn_type=(man.get("arguments") or {}).get("gnn_type"),
                                                       checkpoint_sha256=(man.get("checkpoint_file") or {}).get("sha256"),
                                                       topology=e.get("topology"), differences_vs_base=diff(ref, json.loads(json.dumps(saved, default=str)))))
    write_json(out / "ieee33_config_report.json", report)
    b0 = effective["battery_constraints"][0]
    print(json.dumps(dict(dess_buses_one_based=effective["dess_buses_one_based"], battery=b0, delta_t_hours=effective["delta_t_hours"],
                          v_limits=[effective["v_min"], effective["v_max"]], current_ratings_a=effective["current_ratings_a"],
                          doe=effective["doe"], backup_mode=effective["backup_mode"], backup=effective["backup"],
                          backup_penalty=effective["reward"].get("infeasible_action_penalty"),
                          constraints=effective["constraints"]), indent=1, default=str))
    n_diff = sum(len(r["differences_vs_base"]) for r in report["saved_final_runs"])
    print(f"discovered source directories: {len(manifests)}")
    for m in manifests:
        print("  ", m.parent)
    print(f"saved final-run environment entries: {len(report['saved_final_runs'])}, total differences vs base: {n_diff}")
    for r in report["saved_final_runs"]:
        if r["differences_vs_base"]:
            print("  DIFF", r["manifest"], r["topology"], r["differences_vs_base"][:5])
    print("written", out.resolve())


if __name__ == "__main__":
    main()
