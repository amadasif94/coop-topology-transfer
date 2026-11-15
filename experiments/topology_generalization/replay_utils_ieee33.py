"""Shared utilities for the IEEE33 TP1 action-replay experiment (open-loop replay of recorded TP1 actions).

Self-contained (does not import the IEEE69 replay code). Conditions:
    A  original TP1 feedback evaluation       (final-test results, read only)
    B  TP1 -> TP1 identity replay              (implementation validation; must reproduce A)
    C  TP1 -> TP6 fixed-action replay          (recorded TP1 raw actor actions applied to TP6)
    D  TP6 closed-loop feedback                (TP6 rows of the SAME final-test run as A, read only)

Replay semantics: the recorded 5-dim actor output `marl_raw_action` (what agent.select_action(obs,
noise_std=0.0) returned, float32) is returned by ReplayAgent.select_action inside the ORIGINAL
evaluator's evaluate_one_episode (experiments/topology_generalization/run_tp_eval.py or
experiments/no_gnn_maddpg/run_tp_eval_mlp.py, unmodified). Reset, env.step(), physical/SOC limits,
DOE, directional backup, power flow, SOC evolution and every metric are the original code paths.
Network files: the ARCHIVED network_inputs/<TP>/ of the original final run (sha256-verified against its
run_manifest.json) are copied into the replay directory; live CSVs are never touched.
Initialisation (verified by the TP1 identity replay): Battery.reset() sets soc_init for every battery;
the reset seed is <manifest seed> + episode index (utility.evaluation_protocol.episode_plan); the start
index is pinned by utility.evaluation_reporting.reset_for_evaluation; exogenous load/PV/price come from
the time-series row start_index + step. Sign convention (dess_env/battery): action > 0 = discharge.
"""
import ast
import copy
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEEDER = "ieee33"
N_AGENTS = 5
EXPECTED_EPISODES = 31
STEPS_PER_EPISODE = 96
ACTION_BOUNDS = (-1.0, 1.0)
MODELS = ("gcn", "gat", "tagconv", "mlp")
LEARNED_POLICY = dict(gcn="maddpg", gat="maddpg", tagconv="maddpg", mlp="mlp_maddpg")
NON_LEARNED = ("zero", "random")
RAW_FIELD = "marl_raw_action"
PROPOSAL_FIELD = "policy_proposal_action_normalized"
PREPARED_FIELD = "marl_prepared_action"
TP6 = dict(open_edges=[(30, 31)], close_edges=[(17, 32)])
DEFAULT_CSV_ROOT = Path("results/topology_generalization/csv")
DEFAULT_OUTPUT_ROOT = DEFAULT_CSV_ROOT / "action_replay_ieee33"
# Original IEEE33 final (TP1-TP7, 31 December days) evaluation directories on RIT:
SOURCE_DIR_PATTERN = "{model}_ieee33_tp_full_eval_run{run}_best"
SOURCE_GLOB = "*_ieee33_tp_full_eval_run*_best"
LINE_KEYS = ("line_data", "gridtensor_line_data")
NETWORK_FILES = dict(line_data="line_data.csv", gridtensor_line_data="gridtensor_line.csv")


class ReplayError(RuntimeError):
    """Any provenance, alignment or validation failure; the replay must stop."""


# ------------------------------------------------------------------------------- basic helpers
def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_json(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=json_default)


def parse_vector(text, n=N_AGENTS, bounds=None):
    """Safe parse (ast.literal_eval, never eval) of a stored list -> float64 array of exactly n finite numbers."""
    if isinstance(text, (list, tuple, np.ndarray)):
        value = list(text)
    else:
        if not isinstance(text, str) or not text.strip():
            raise ReplayError(f"empty or non-text vector field: {text!r}")
        try:
            value = ast.literal_eval(text.strip())
        except (ValueError, SyntaxError) as exc:
            raise ReplayError(f"unparseable vector field {text[:80]!r}: {exc}") from exc
    if not isinstance(value, (list, tuple)) or (n is not None and len(value) != n):
        raise ReplayError(f"expected {n} values, got {value!r}"[:200])
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ReplayError(f"non-numeric entries: {value!r}"[:200])
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ReplayError(f"non-finite values: {value!r}"[:200])
    if bounds is not None and ((arr < bounds[0]).any() or (arr > bounds[1]).any()):
        raise ReplayError(f"values outside {bounds}: {value!r}"[:200])
    return arr


def matrix(series, n=N_AGENTS):
    """JSON-list column -> (rows, n) float array; unparseable/missing rows become NaN (diagnostics only)."""
    out = np.full((len(series), n), np.nan)
    for i, v in enumerate(series):
        try:
            out[i] = parse_vector(v, n)
        except ReplayError:
            pass
    return out


def csv_header(path):
    return list(pd.read_csv(path, nrows=0).columns)


def read_filtered(path, topology, policy, columns, chunksize=20000):
    """Memory-safe read of `columns` for rows with topology_case == topology and policy == policy."""
    header = set(csv_header(path))
    for c in ("topology_case", "policy"):
        if c not in header:
            raise ReplayError(f"{path}: required column {c!r} missing")
    keep = [c for c in dict.fromkeys(["topology_case", "policy", *columns]) if c in header]
    parts = [chunk[(chunk["topology_case"] == topology) & (chunk["policy"] == policy)]
             for chunk in pd.read_csv(path, usecols=keep, chunksize=chunksize, dtype={"topology_case": str, "policy": str},
                                      low_memory=False)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)
    return out, sorted(set(columns) - header)


def policies_present(path, topology, chunksize=50000):
    seen = set()
    for chunk in pd.read_csv(path, usecols=["topology_case", "policy"], chunksize=chunksize, dtype=str):
        seen |= set(chunk.loc[chunk["topology_case"] == topology, "policy"].dropna())
    return sorted(seen)


def learned_policy(model, csv_path, topology):
    """The learned-policy label, confirmed against the actual CSV (zero/random rows never used)."""
    expected = LEARNED_POLICY[model]
    present = policies_present(csv_path, topology)
    learned = [p for p in present if p not in NON_LEARNED]
    if learned != [expected]:
        raise ReplayError(f"{csv_path}: {topology} learned policies {learned} (present {present}); expected exactly [{expected!r}]")
    return expected


# ------------------------------------------------------------------------------- source actions
SOURCE_COLUMNS = ["episode_id", "step", "start_index", "date_time", "record_type", "episode_completed",
                  RAW_FIELD, PROPOSAL_FIELD, PREPARED_FIELD]


def load_source_actions(csv_path, model, expected_episodes=EXPECTED_EPISODES, steps=STEPS_PER_EPISODE,
                        topology="TP1", bounds=ACTION_BOUNDS):
    """Recorded raw actor actions of ONE controller on `topology`, fully validated.

    Returns (actions {start_index: float32 (steps, 5)}, episodes [(episode_id, start_index)], report)."""
    csv_path = Path(csv_path)
    policy = learned_policy(model, csv_path, topology)
    df, missing = read_filtered(csv_path, topology, policy, SOURCE_COLUMNS)
    need = [c for c in ("episode_id", "step", "start_index", "date_time", RAW_FIELD) if c in missing]
    if need:
        raise ReplayError(f"{csv_path}: required columns missing {need}")
    if "record_type" in df and (df["record_type"] != "dispatch_attempt").any():
        raise ReplayError(f"{csv_path}: non-dispatch rows (reset failures) in source {topology}/{policy}")
    report = dict(source_csv=str(csv_path.resolve()), source_sha256=sha256_file(csv_path), topology=topology, policy=policy,
                  rows=len(df), raw_field=RAW_FIELD, missing_optional_columns=missing)
    if len(df) != expected_episodes * steps:
        raise ReplayError(f"{csv_path}: {len(df)} {topology}/{policy} rows, expected {expected_episodes}x{steps}")
    for c in ("episode_id", "step", "start_index"):
        df[c] = df[c].astype(int)
    if df.duplicated(["start_index", "step"]).any():
        raise ReplayError(f"{csv_path}: duplicate (start_index, step) keys")
    pairs = df[["episode_id", "start_index"]].drop_duplicates()
    if pairs["episode_id"].duplicated().any() or pairs["start_index"].duplicated().any() or len(pairs) != expected_episodes:
        raise ReplayError(f"{csv_path}: inconsistent episode_id <-> start_index mapping ({len(pairs)} pairs)")
    actions, episodes, times = {}, [], {}
    raw_eq_prop = prep_eq = 0
    for (ep, start), g in df.groupby(["episode_id", "start_index"], sort=True):
        g = g.sort_values("step")
        if g["step"].tolist() != list(range(steps)):
            raise ReplayError(f"episode {ep} (start {start}): steps not exactly 0..{steps - 1}")
        t = pd.to_datetime(g["date_time"], utc=True, errors="coerce")
        if t.isna().any() or not (t.diff().dropna() == pd.Timedelta(minutes=15)).all():
            raise ReplayError(f"episode {ep} (start {start}): timestamps not consecutive 15-min steps")
        if "episode_completed" in g and not g["episode_completed"].astype(str).eq("True").all():
            raise ReplayError(f"episode {ep} (start {start}): source episode not completed")
        raw = np.vstack([parse_vector(v, bounds=bounds) for v in g[RAW_FIELD]])
        if PROPOSAL_FIELD in g:
            prop = np.vstack([parse_vector(v) for v in g[PROPOSAL_FIELD]])
            if not np.array_equal(raw, prop):
                raise ReplayError(f"episode {ep}: {RAW_FIELD} != {PROPOSAL_FIELD} (max |diff| {np.abs(raw - prop).max()})")
            raw_eq_prop += len(g)
        if PREPARED_FIELD in g:
            prep = np.vstack([parse_vector(v) for v in g[PREPARED_FIELD]])
            if not np.array_equal(np.clip(raw.astype(np.float32), *bounds), prep.astype(np.float32)):
                raise ReplayError(f"episode {ep}: {PREPARED_FIELD} != clip({RAW_FIELD})")
            prep_eq += len(g)
        actions[int(start)] = raw.astype(np.float32)
        episodes.append((int(ep), int(start)))
        times[int(start)] = str(g["date_time"].iloc[0])
    if [s for _, s in episodes] != sorted(s for _, s in episodes):
        raise ReplayError("episode ids and start indices are not in the same chronological order")
    report.update(episodes=len(episodes), steps_per_episode=steps, records=len(df),
                  raw_equals_proposal_steps=raw_eq_prop, prepared_equals_clip_raw_steps=prep_eq,
                  raw_equals_proposal_checked_all=(raw_eq_prop == len(df)) if PROPOSAL_FIELD in df else None,
                  start_indices=[s for _, s in episodes], first_timestamps=times)
    return actions, episodes, report


class ReplayAgent:
    """Stands in for the actor: returns the recorded raw action for the env's current (start_index, step)."""

    def __init__(self, env, actions):
        self.env, self.actions = env, actions
        self.start, self.k, self.calls = None, 0, 0

    def begin_episode(self, start_index):
        if int(start_index) not in self.actions:
            raise ReplayError(f"no recorded actions for start_index {start_index}")
        self.start, self.k = int(start_index), 0

    def select_action(self, obs, noise_std=0.0):
        del obs                                                  # open loop: the observation is NOT used
        if noise_std != 0.0:
            raise ReplayError("replay must be noise-free")
        seq = self.actions[self.start]
        if self.k >= len(seq):
            raise ReplayError(f"more env steps than recorded actions for start {self.start}")
        idx, step = int(self.env.current_index), int(self.env.current_step)
        if idx != self.start + self.k or step != self.k:
            raise ReplayError(f"alignment: env at index {idx} step {step}, replay expects {self.start + self.k} / {self.k}")
        action = seq[self.k].copy()
        self.k += 1
        self.calls += 1
        return action

    def end_episode(self):
        if self.k != len(self.actions[self.start]):
            raise ReplayError(f"episode {self.start}: replayed {self.k} actions, recorded {len(self.actions[self.start])}")


# ------------------------------------------------------------------------------- manifests and provenance
def read_manifest(run_dir):
    p = Path(run_dir) / "run_manifest.json"
    if not p.is_file():
        raise ReplayError(f"run manifest missing: {p}")
    return json.loads(p.read_text())


def manifest_env(manifest, topology):
    entries = [e for e in manifest.get("environments", []) if str(e.get("topology", "")).upper() == topology.upper()]
    if not entries:
        raise ReplayError(f"manifest has no {topology} environment entry")
    shas = {json.dumps(input_shas(e), sort_keys=True) for e in entries}
    if len(shas) != 1:
        raise ReplayError(f"manifest {topology} entries disagree on input files")
    return entries[0]


def input_shas(entry_or_files):
    files = entry_or_files.get("input_files", entry_or_files)
    return {k: (v or {}).get("sha256") for k, v in files.items()}


def selection_info(selection_path, model):
    p = Path(selection_path)
    rec = json.loads(p.read_text())
    if rec.get("feeder") != FEEDER or rec.get("model") != model:
        raise ReplayError(f"selection {p} is {rec.get('feeder')}/{rec.get('model')}, not {FEEDER}/{model}")
    sel = rec.get("selected") or {}
    return dict(path=str(p.resolve()), sha256=sha256_file(p), selection_id=rec["selection_id"], run=rec.get("run"),
                checkpoint_path=sel.get("checkpoint_path"), checkpoint_sha256=sel.get("checkpoint_sha256"),
                checkpoint_step=sel.get("step"))


def run_number(sel):
    """Training-run number from the selection's 'run' field (e.g. 'run_0' -> '0')."""
    m = re.search(r"(\d+)\s*$", str((sel or {}).get("run", "")))
    if not m:
        raise ReplayError(f"cannot read a run number from selection run {sel.get('run')!r}; pass --source_dir explicitly")
    return m.group(1)


def resolve_source_dir(model, sel, csv_root=DEFAULT_CSV_ROOT):
    """Original final-evaluation directory: <csv_root>/<model>_ieee33_tp_full_eval_run<run>_best."""
    d = Path(csv_root) / SOURCE_DIR_PATTERN.format(model=model, run=run_number(sel))
    missing = [n for n in ("run_manifest.json", "step_metrics.csv", "network_inputs") if not (d / n).exists()]
    if missing:
        raise ReplayError(f"original evaluation directory {d} lacks {missing} (pass --source_dir explicitly)")
    return d


def check_source_selection(manifest, sel, model):
    """The frozen selection must identify the checkpoint the source run actually evaluated.

    Uses every provenance record available in the run manifest: the F2 protocol selection record (final runs made
    with --selection) and the evaluated checkpoint file record (sha256 of the loaded weights, F14). At least one must
    exist and all that exist must agree; the model identity must match. Returns the checks for the replay manifest."""
    checks = {}
    rec = (manifest.get("evaluation_protocol") or {}).get("selection")
    if isinstance(rec, dict):
        checks.update(protocol_selection_sha256=rec.get("selection_sha256") == sel["sha256"],
                      protocol_selection_id=rec.get("selection_id") == sel["selection_id"],
                      protocol_checkpoint_sha256=rec.get("checkpoint_sha256") == sel["checkpoint_sha256"])
    evaluated = (manifest.get("checkpoint_file") or {}).get("sha256")
    if evaluated:
        checks["evaluated_checkpoint_sha256_equals_selection"] = evaluated == sel["checkpoint_sha256"]
    if not checks:
        raise ReplayError("source run manifest records neither a selection nor the evaluated checkpoint sha256")
    args = manifest.get("arguments") or {}
    if model == "mlp":
        checks["model_identity"] = str(manifest.get("controller", "")).upper() == "MLP" and not args.get("gnn_type")
    else:
        checks["model_identity"] = args.get("gnn_type") == model
    if not all(checks.values()):
        raise ReplayError(f"source run does not match selection {sel['selection_id']} / model {model}: {checks}")
    return dict(checks, evaluated_checkpoint_sha256=evaluated, selection_checkpoint_sha256=sel["checkpoint_sha256"])


def episode_args(manifest):
    args = manifest.get("arguments") or {}
    mode = manifest.get("mode")
    flag = (manifest.get("evaluation_protocol") or {}).get("final_test")
    return dict(seed=int(manifest["seed"]), episodes=str(args.get("episodes", "all")), mode=mode,
                final=bool(flag) or mode in ("test", "all"))


def verify_source_records(source_csv, source_dir, manifest, model, start_indices, topologies=("TP1", "TP6"),
                          expected_episodes=EXPECTED_EPISODES):
    """Original learned-policy records and archived networks per topology (read only; report)."""
    out = {}
    policy = LEARNED_POLICY[model]
    for tp in topologies:
        r = dict(topology=tp)
        try:
            df, _ = read_filtered(source_csv, tp, policy, ["start_index", "step", "record_type", "episode_completed"])
            per = df.groupby("start_index")["step"].count()
            done = df.groupby("start_index")["episode_completed"].agg(lambda s: s.astype(str).eq("True").all()) if len(df) else per
            r.update(rows=len(df), episodes=len(per), all_96_steps=bool((per == STEPS_PER_EPISODE).all()) if len(per) else False,
                     completed_episodes=int(done.sum()) if len(df) else 0,
                     same_days_as_TP1=sorted(int(s) for s in per.index) == list(start_indices))
            recorded = input_shas(manifest_env(manifest, tp))
            files = {k: sha256_file(Path(source_dir) / "network_inputs" / tp / n) if (Path(source_dir) / "network_inputs" / tp / n).is_file()
                     else None for k, n in NETWORK_FILES.items()}
            r["archived_network_matches_manifest"] = all(files[k] is not None and files[k] == recorded.get(k) for k in NETWORK_FILES)
        except ReplayError as exc:
            r["error"] = str(exc)
        r["ok"] = bool(r.get("rows") == expected_episodes * STEPS_PER_EPISODE and r.get("completed_episodes") == expected_episodes
                       and r.get("all_96_steps") and r.get("same_days_as_TP1") and r.get("archived_network_matches_manifest"))
        out[tp] = r
    return out


def compare_input_files(env_files, reference, keys, label):
    diff = {k: (env_files.get(k), reference.get(k)) for k in keys if env_files.get(k) != reference.get(k)}
    if diff:
        raise ReplayError(f"{label}: input files differ {diff}")
    return {k: env_files.get(k) for k in keys}


# ------------------------------------------------------------------------------- networks
def archived_config(base_config, source_dir, source_manifest, topology, out_dir):
    """Config whose line files are the ARCHIVED network_inputs/<topology>/ of the original final run,
    sha256-verified against that run's manifest and copied into out_dir. Live CSVs are never touched."""
    from experiments.topology_generalization.tp_configs import get_topology_case
    src = Path(source_dir) / "network_inputs" / topology
    recorded = input_shas(manifest_env(source_manifest, topology))
    dst = Path(out_dir) / "network_inputs" / topology
    dst.mkdir(parents=True, exist_ok=False)
    report = {}
    for key, name in NETWORK_FILES.items():
        p = src / name
        if not p.is_file():
            raise ReplayError(f"archived {topology} network input missing: {p}")
        sha = sha256_file(p)
        if sha != recorded.get(key):
            raise ReplayError(f"archived {p} sha256 {sha} != original run manifest {recorded.get(key)}")
        shutil.copyfile(p, dst / name)
        if sha256_file(dst / name) != sha:
            raise ReplayError(f"copy of {p} changed its content")
        report[key] = dict(archived=str(p.resolve()), copy=str((dst / name).resolve()), sha256=sha)
    cfg = copy.deepcopy(base_config)
    case = get_topology_case(topology)
    cfg["case_name"] = f"ieee33_{topology}"
    cfg["paths"]["line_data"] = dst / NETWORK_FILES["line_data"]
    cfg["paths"]["gridtensor_line_data"] = dst / NETWORK_FILES["gridtensor_line_data"]
    cfg["topology_case"] = dict(case_name=topology, description=case["description"], open_edges=list(case["open_edges"]),
                                close_edges=list(case["close_edges"]), generated_dir=dst)
    return cfg, report


def regenerated_cross_check(base_config, topology, archived_report, out_dir):
    """Report only: regenerate the topology from the CURRENT live source CSVs and compare with the archive."""
    from experiments.topology_generalization.topology_utils import build_topology_config
    try:
        built = build_topology_config(topology, base_config=copy.deepcopy(base_config),
                                      output_root=Path(out_dir) / "regenerated_check", validate=True)
    except (ValueError, OSError, KeyError) as exc:
        return dict(status=f"NOT CHECKED ({type(exc).__name__}: {exc})")
    res = {k: sha256_file(built["paths"][k]) == archived_report[k]["sha256"] for k in LINE_KEYS}
    return dict(status="MATCH" if all(res.values()) else "DIFFERS (live source CSVs changed since the final run)", per_file=res)


def active_edges(env):
    df = env.grid.in_service_line_df
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(df["from_bus"], df["to_bus"])}


def check_topology(env, topology):
    """Energized network matches the TP definition (radial, 33 buses, 32 active branches, 5 batteries)."""
    from experiments.topology_generalization.tp_configs import get_topology_case
    case = get_topology_case(topology)
    act = active_edges(env)
    checks = dict(num_buses=int(env.num_buses) == 33, active_branches=len(act) == 32,
                  batteries=len(env.batteries) == N_AGENTS == int(env.num_agents),
                  opened_edges_inactive=all(tuple(e) not in act for e in case["open_edges"]),
                  closed_edges_active=all(tuple(e) in act for e in case["close_edges"]))
    if topology == "TP6":
        checks["tp6_definition"] = (sorted(map(tuple, case["open_edges"])) == TP6["open_edges"]
                                    and sorted(map(tuple, case["close_edges"])) == TP6["close_edges"])
    if not all(checks.values()):
        raise ReplayError(f"{topology} network check failed: {checks}")
    return checks


# ------------------------------------------------------------------------------- execution-order gate
def check_gate(report_path, kind, model, source_csv, episodes, target=None):
    if not report_path:
        raise ReplayError(f"a passing {kind} validation report is required before this replay "
                          f"(--{'identity' if kind == 'identity' else 'smoke'}_report)")
    rep = json.loads(Path(report_path).read_text())
    problems = []
    if rep.get("kind") != kind or rep.get("status") != "PASS":
        problems.append(f"report kind/status {rep.get('kind')}/{rep.get('status')}")
    if rep.get("model") != model:
        problems.append(f"model {rep.get('model')}")
    if rep.get("source_sha256") != sha256_file(source_csv):
        problems.append("different source step_metrics.csv")
    if int(rep.get("episodes_validated", 0)) < episodes:
        problems.append(f"{rep.get('episodes_validated')} episodes validated, need {episodes}")
    if target and rep.get("target_topology") != target:
        problems.append(f"target {rep.get('target_topology')}")
    if problems:
        raise ReplayError(f"gate {report_path} not satisfied: " + "; ".join(problems))
    return dict(report=str(Path(report_path).resolve()), sha256=sha256_file(report_path), status="PASS")


# ------------------------------------------------------------------------------- the replay run (both evaluators)
def run_replay(module, model, args, policy):
    from configs.ieee33_config import IEEE33_CONFIG
    from environments.primary_marl import make_primary_env
    from utility.evaluation_protocol import authorize, protocol_config, episode_plan
    from utility.provenance import input_files
    from utility.run_manifest import RunManifest

    target = args.target_topology.upper()
    sel = selection_info(args.selection, model) if args.selection else None
    if args.source_dir:
        source_dir = Path(args.source_dir)
    elif sel:
        source_dir = resolve_source_dir(model, sel, args.csv_root)
    else:
        raise ReplayError("--source_dir or --selection is required")
    source_csv = source_dir / "step_metrics.csv"
    src_manifest = read_manifest(source_dir)
    ep_args = episode_args(src_manifest)
    if ep_args["final"] and sel is None:
        raise ReplayError("the source is a final test: --selection (frozen selection.json) is required")
    provenance = check_source_selection(src_manifest, sel, model) if sel else None
    gate = None
    if target != "TP1":
        gate = check_gate(args.identity_report, "identity", model, source_csv, args.expected_episodes)
        if args.max_episodes is None or args.max_episodes > 1:
            gate = dict(identity=gate, smoke=check_gate(args.smoke_report, "target", model, source_csv, 1, target=target))
    actions, episodes, src_report = load_source_actions(source_csv, model, args.expected_episodes)
    records = verify_source_records(source_csv, source_dir, src_manifest, model, src_report["start_indices"],
                                    topologies=tuple(dict.fromkeys(("TP1", "TP6", target))), expected_episodes=args.expected_episodes)
    for tp in ("TP1", target):
        if not records[tp]["ok"]:
            raise ReplayError(f"original {tp} records / archived network not verified: {records[tp]}")
    n_run = len(episodes) if args.max_episodes is None else min(args.max_episodes, len(episodes))
    sel_id = sel["selection_id"] if sel else source_dir.name
    run_id = args.run_id or f"{'smoke' if n_run < len(episodes) else 'full'}_{utc_stamp()}"
    out = Path(args.output_root) / target / model / sel_id / run_id
    if out.exists() and any(out.iterdir()):
        raise ReplayError(f"refusing to reuse non-empty output directory {out}")
    protocol = authorize(ep_args["mode"], ep_args["final"], "final_test" if ep_args["final"] else "development",
                         learned=True, topologies=[target], selection=sel["path"] if (sel and ep_args["final"]) else None)
    run = RunManifest(out, f"REPLAY_{model}", ep_args["seed"], ep_args["mode"], None,
                      dict(vars(args), replay=True, source_dir=str(source_dir), policy_label=policy))
    run.record_protocol(protocol, [policy])
    base = copy.deepcopy(IEEE33_CONFIG)
    net = {}
    cfg, net["archived_network"] = archived_config(base, source_dir, src_manifest, target, out)
    net["regenerated_from_live_csvs"] = regenerated_cross_check(base, target, net["archived_network"], out)
    cfg = protocol_config(cfg, protocol)
    env = make_primary_env(config=cfg, mode=ep_args["mode"], seed=ep_args["seed"])
    net["topology_checks"] = check_topology(env, target)
    files = {k: (v or {}).get("sha256") for k, v in input_files(env).items()}
    recorded = input_shas(manifest_env(src_manifest, target))
    net["input_files_vs_original_run"] = compare_input_files(files, recorded, sorted(recorded),
                                                            f"{target} replay vs the original final run's {target} environment")
    run.add_environment(env, topology=target)
    starts, seeds, _ = episode_plan(env, ep_args["episodes"], ep_args["seed"], protocol)
    if [int(s) for s in starts] != src_report["start_indices"]:
        raise ReplayError("episode plan of the replay environment differs from the source episodes")
    if target != "TP1":                                          # feedback = target rows of the same final run
        fb, _ = read_filtered(source_csv, target, policy, ["start_index"])
        if sorted(int(s) for s in fb["start_index"].unique()) != src_report["start_indices"]:
            raise ReplayError(f"{target} feedback start indices differ from the TP1 source")
    agent = ReplayAgent(env, actions)
    rows = []
    condition = "TP1_to_TP1_identity_replay" if target == "TP1" else f"TP1_to_{target}_fixed_action_replay"
    for ep_id, start in episodes[:n_run]:
        if int(starts[ep_id]) != start:
            raise ReplayError(f"episode {ep_id}: plan start {starts[ep_id]} != source {start}")
        agent.begin_episode(start)
        ep_rows = module.evaluate_one_episode(env=env, agent=agent, policy=policy, topology_case=target,
                                              episode_id=ep_id, start_index=start, seed=seeds[ep_id])
        agent.end_episode()
        if len(ep_rows) != STEPS_PER_EPISODE or not all(r.get("episode_completed") for r in ep_rows):
            raise ReplayError(f"episode {ep_id}: replay produced {len(ep_rows)} rows / incomplete episode")
        for r in ep_rows:
            r.update(replay_condition=condition, replay_source_topology="TP1", replay_raw_field=RAW_FIELD,
                     replay_source_episode_id=ep_id, replay_source_start_index=start)
        rows.extend(ep_rows)
        print(f"{condition} model={model} episode={ep_id} start={start} seed={seeds[ep_id]} "
              f"cost={sum(r['energy_cost'] for r in ep_rows):.4f}", flush=True)
    step_df = pd.DataFrame(rows)
    episode_summary, aggregate_summary = module.summarize(step_df)
    step_df.to_csv(out / "step_metrics.csv", index=False)
    episode_summary.to_csv(out / "episode_summary.csv", index=False)
    aggregate_summary.to_csv(out / "aggregate_summary.csv", index=False)
    write_json(out / "replay_manifest.json", dict(
        feeder=FEEDER, condition=condition, model=model, policy_label=policy, target_topology=target, source_topology="TP1",
        replay_semantics=("recorded TP1 actor output (marl_raw_action, float32) returned by ReplayAgent.select_action inside the "
                          "original evaluate_one_episode; no actor inference; observations ignored; env.step recomputes "
                          "physical/SOC limits, DOE, backup, power flow and SOC"),
        source=dict(src_report, source_dir=str(source_dir.resolve()), source_manifest_sha256=sha256_file(source_dir / "run_manifest.json"),
                    selection_provenance=provenance, original_records=records),
        selection=sel, training_run=(sel or {}).get("run"),
        reset=dict(seed=ep_args["seed"], episodes_arg=ep_args["episodes"], reset_seeds=[int(seeds[e]) for e, _ in episodes[:n_run]],
                   derivation="episode_plan(env, episodes, seed, protocol): seed + episode index; initial SOC = soc_init (Battery.reset)"),
        mode=ep_args["mode"], final_test=ep_args["final"],
        episode_index_mapping=[dict(episode_id=e, start_index=s) for e, s in episodes[:n_run]], episodes_replayed=n_run,
        episodes_available=len(episodes), actor_calls=agent.calls, network=net, gates=gate, target_input_files=files,
        created_utc=datetime.now(timezone.utc).isoformat(), evaluator=module.__name__, output_dir=str(out.resolve())))
    run.data["replay"] = dict(condition=condition, episodes_replayed=n_run)
    run.finish()
    print("REPLAY OUTPUT", out.resolve(), flush=True)
    return out


def add_replay_arguments(parser):
    parser.add_argument("--selection", help="frozen selection.json (required for final-test sources)")
    parser.add_argument("--source_dir", help="original final evaluation directory "
                        "(default <csv_root>/<model>_ieee33_tp_full_eval_run<run>_best, run from the selection)")
    parser.add_argument("--csv_root", default=str(DEFAULT_CSV_ROOT), help="root holding the original evaluation directories")
    parser.add_argument("--target_topology", required=True, help="TP1 (identity) or TP6")
    parser.add_argument("--max_episodes", type=int, default=None, help="replay only the first N episodes (smoke test: 1)")
    parser.add_argument("--expected_episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_id", default=None, help="output sub-directory name (default smoke_/full_<UTC>)")
    parser.add_argument("--identity_report", default=None, help="PASS full identity validation (required for TP6)")
    parser.add_argument("--smoke_report", default=None, help="PASS TP6 smoke validation (required for the full TP6 replay)")
    parser.add_argument("--cpu", action="store_true", help="accepted for symmetry; no network is evaluated")
    return parser
