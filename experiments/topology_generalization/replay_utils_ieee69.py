"""Shared utilities for the IEEE69 TP1 action-replay experiment (open-loop replay of recorded TP1 actions).

Conditions
    A  original TP1 feedback evaluation          (existing final-test results, read only)
    B  TP1 -> TP1 identity replay                 (implementation validation; must reproduce A)
    C  TP1 -> TP10 action replay                  (recorded TP1 raw actor actions applied to TP10)
    D  TP10 normal feedback                       (existing TP10 final-test results, read only)

Replay semantics: the recorded 9-dim actor output (`marl_raw_action`, i.e. what
`agent.select_action(obs, noise_std=0.0)` returned) is handed to the ORIGINAL evaluator's
`evaluate_one_episode` through ReplayAgent.select_action, so reset, env.step(), physical limits,
DOE, directional backup, power flow, SOC evolution and every metric are the original code paths.
Nothing here modifies existing files; network files are copied/generated only inside the replay
output directory. Sign convention (environments/dess_env.py): action > 0 = discharge, < 0 = charge.
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

N_AGENTS = 9
EXPECTED_EPISODES = 31
STEPS_PER_EPISODE = 96
ACTION_BOUNDS = (-1.0, 1.0)
LEARNED_POLICY = dict(gcn="maddpg", tagconv="maddpg", gat="maddpg", mlp="mlp_maddpg")
NON_LEARNED = ("zero", "random")
RAW_FIELD = "marl_raw_action"
PROPOSAL_FIELD = "policy_proposal_action_normalized"
PREPARED_FIELD = "marl_prepared_action"
TP10 = dict(open_edges=[(62, 63)], close_edges=[(26, 64)], r_ohm=1.041, x_ohm=0.5302, max_current_a=255.0,
            num_buses=69, active_branches=68)
DEFAULT_CSV_ROOT = Path("results/topology_generalization/csv_ieee69")
DEFAULT_OUTPUT_ROOT = DEFAULT_CSV_ROOT / "action_replay"
LINE_KEYS = ("line_data", "gridtensor_line_data")


class ReplayError(RuntimeError):
    """Any provenance, alignment or validation failure; replay must stop."""


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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, set, tuple)):
        return str(o) if isinstance(o, Path) else list(o)
    return str(o)


def write_json(path, obj, exclusive=True):
    with open(path, "x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=json_default)


def parse_vector(text, n=N_AGENTS, bounds=None):
    """Safe parse of a stored list (JSON / Python literal) -> float64 array of exactly n finite values."""
    if isinstance(text, (list, tuple, np.ndarray)):
        value = list(text)
    else:
        if not isinstance(text, str) or not text.strip():
            raise ReplayError(f"empty or non-text action field: {text!r}")
        try:
            value = ast.literal_eval(text.strip())
        except (ValueError, SyntaxError) as exc:
            raise ReplayError(f"unparseable action field {text[:80]!r}: {exc}") from exc
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ReplayError(f"expected {n} values, got {value!r}"[:200])
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ReplayError(f"non-numeric action entries: {value!r}"[:200])
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ReplayError(f"non-finite action values: {value!r}"[:200])
    if bounds is not None and ((arr < bounds[0]).any() or (arr > bounds[1]).any()):
        raise ReplayError(f"action outside bounds {bounds}: {value!r}"[:200])
    return arr


def csv_header(path):
    return list(pd.read_csv(path, nrows=0).columns)


def read_filtered(path, topology, policy, columns, chunksize=20000):
    """Memory-safe read of only `columns` for rows with topology_case == topology and policy == policy."""
    header = set(csv_header(path))
    for c in ("topology_case", "policy"):
        if c not in header:
            raise ReplayError(f"{path}: required column {c!r} missing")
    keep = [c for c in dict.fromkeys(["topology_case", "policy", *columns]) if c in header]
    parts = []
    for chunk in pd.read_csv(path, usecols=keep, chunksize=chunksize, dtype={"topology_case": str, "policy": str},
                             low_memory=False):
        parts.append(chunk[(chunk["topology_case"] == topology) & (chunk["policy"] == policy)])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)
    return out, sorted(set(columns) - header)


def policies_present(path, topology, chunksize=50000):
    seen = set()
    for chunk in pd.read_csv(path, usecols=["topology_case", "policy"], chunksize=chunksize, dtype=str):
        seen |= set(chunk.loc[chunk["topology_case"] == topology, "policy"].dropna())
    return sorted(seen)


def learned_policy(model, csv_path, topology):
    """The learned-policy identifier, confirmed against the actual CSV (zero/random rows are never used)."""
    expected = LEARNED_POLICY[model]
    present = policies_present(csv_path, topology)
    learned = [p for p in present if p not in NON_LEARNED]
    if learned != [expected]:
        raise ReplayError(f"{csv_path}: {topology} learned policies {learned} (all present {present}); expected exactly [{expected!r}]")
    return expected


# ------------------------------------------------------------------------------- source actions
SOURCE_COLUMNS = ["episode_id", "step", "start_index", "date_time", "record_type", "episode_completed",
                  RAW_FIELD, PROPOSAL_FIELD, PREPARED_FIELD]


def load_source_actions(csv_path, model, expected_episodes=EXPECTED_EPISODES, steps=STEPS_PER_EPISODE,
                        topology="TP1", bounds=ACTION_BOUNDS):
    """Recorded TP1 raw actor actions of ONE controller, fully validated.

    Returns (actions {start_index: float32 (steps, 9)}, episodes [(episode_id, start_index)] in order, report)."""
    csv_path = Path(csv_path)
    policy = learned_policy(model, csv_path, topology)
    df, missing = read_filtered(csv_path, topology, policy, SOURCE_COLUMNS)
    need = [c for c in ("episode_id", "step", "start_index", "date_time", RAW_FIELD) if c in missing]
    if need:
        raise ReplayError(f"{csv_path}: required columns missing {need}")
    if "record_type" in df:
        bad = df[df["record_type"] != "dispatch_attempt"]
        if len(bad):
            raise ReplayError(f"{csv_path}: {len(bad)} non-dispatch rows (reset failures) in source {topology}/{policy}")
    report = dict(source_csv=str(csv_path.resolve()), source_sha256=sha256_file(csv_path), topology=topology, policy=policy,
                  rows=len(df), raw_field=RAW_FIELD, missing_optional_columns=missing)
    if len(df) != expected_episodes * steps:
        raise ReplayError(f"{csv_path}: {len(df)} {topology}/{policy} rows, expected {expected_episodes}x{steps}={expected_episodes * steps}")
    df["episode_id"] = df["episode_id"].astype(int)
    df["step"] = df["step"].astype(int)
    df["start_index"] = df["start_index"].astype(int)
    if df.duplicated(["start_index", "step"]).any():
        raise ReplayError(f"{csv_path}: duplicate (start_index, step) keys")
    pairs = df[["episode_id", "start_index"]].drop_duplicates()
    if pairs["episode_id"].duplicated().any() or pairs["start_index"].duplicated().any() or len(pairs) != expected_episodes:
        raise ReplayError(f"{csv_path}: inconsistent episode_id <-> start_index mapping ({len(pairs)} pairs)")
    actions, episodes = {}, []
    raw_eq_prop = prep_eq_raw = 0
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
            prep_eq_raw += len(g)
        actions[int(start)] = raw.astype(np.float32)          # the actor returned float32 (models/*.select_action)
        episodes.append((int(ep), int(start)))
    if [e for e, _ in episodes] != sorted(e for e, _ in episodes) or [s for _, s in episodes] != sorted(s for _, s in episodes):
        raise ReplayError("episode ids and start indices are not in the same chronological order")
    report.update(episodes=len(episodes), steps_per_episode=steps, records=len(df),
                  raw_equals_proposal_steps=raw_eq_prop, prepared_equals_clip_raw_steps=prep_eq_raw,
                  raw_equals_proposal_checked_all=(raw_eq_prop == len(df)) if PROPOSAL_FIELD in df else None,
                  start_indices=[s for _, s in episodes])
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
        n = len(self.actions[self.start])
        if self.k != n:
            raise ReplayError(f"episode {self.start}: replayed {self.k} actions, recorded {n}")


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
    shas = {json.dumps({k: (v or {}).get("sha256") for k, v in e.get("input_files", {}).items()}, sort_keys=True) for e in entries}
    if len(shas) != 1:
        raise ReplayError(f"manifest {topology} entries disagree on input files")
    return entries[0]


def input_shas(entry_or_files):
    files = entry_or_files.get("input_files", entry_or_files)
    return {k: (v or {}).get("sha256") for k, v in files.items()}


def selection_info(selection_path, model, feeder="ieee69"):
    p = Path(selection_path)
    rec = json.loads(p.read_text())
    if rec.get("feeder") != feeder or rec.get("model") != model:
        raise ReplayError(f"selection {p} is {rec.get('feeder')}/{rec.get('model')}, not {feeder}/{model}")
    return dict(path=str(p.resolve()), sha256=sha256_file(p), selection_id=rec["selection_id"], run=rec.get("run"),
                checkpoint_path=(rec.get("selected") or {}).get("checkpoint_path"),
                checkpoint_sha256=(rec.get("selected") or {}).get("checkpoint_sha256"),
                checkpoint_step=(rec.get("selected") or {}).get("step"))


def check_source_selection(manifest, sel):
    """The source run must be the final test of exactly this frozen selection."""
    prot = manifest.get("evaluation_protocol") or {}
    rec = prot.get("selection") or {}
    problems = []
    if rec.get("selection_sha256") != sel["sha256"]:
        problems.append(f"selection sha256 {rec.get('selection_sha256')} != {sel['sha256']}")
    if rec.get("selection_id") != sel["selection_id"]:
        problems.append(f"selection id {rec.get('selection_id')} != {sel['selection_id']}")
    if rec.get("checkpoint_sha256") != sel["checkpoint_sha256"]:
        problems.append("checkpoint sha256 differs from the selection")
    if problems:
        raise ReplayError("source/selection mismatch: " + "; ".join(problems))


def resolve_feedback_dir(job_id, log_dir="results/logs/slurm", csv_root=DEFAULT_CSV_ROOT):
    """TP10 feedback result directory of a final_test_learned.slurm job, from its log's run_name line."""
    log = Path(log_dir) / f"final_test_learned_{job_id}.out"
    if not log.is_file():
        raise ReplayError(f"SLURM log not found: {log} (pass --feedback_dir explicitly)")
    m = re.search(r"run_name=(\S+)", log.read_text(errors="replace"))
    if not m:
        raise ReplayError(f"{log}: no 'run_name=' line (pass --feedback_dir explicitly)")
    d = Path(csv_root) / m.group(1)
    if not (d / "step_metrics.csv").is_file():
        raise ReplayError(f"resolved feedback dir {d} has no step_metrics.csv")
    return d


def episode_args(manifest):
    args = manifest.get("arguments") or {}
    return dict(seed=int(manifest["seed"]), episodes=str(args.get("episodes", "all")), mode=manifest.get("mode"),
                final=bool((manifest.get("evaluation_protocol") or {}).get("final_test")))


# ------------------------------------------------------------------------------- networks
def topology_case_dict(case_name, description, open_edges, close_edges, generated_dir):
    return dict(case_name=case_name, description=description, open_edges=list(open_edges),
                close_edges=list(close_edges), generated_dir=generated_dir)


def archived_tp1_config(base_config, source_dir, source_manifest, out_dir):
    """Config whose TP1 line files are the ARCHIVED inputs of the original TP1 evaluation (copied into out_dir
    after verifying their sha256 against the original run manifest). Live CSVs are never touched."""
    from experiments.topology_generalization.tp_configs_ieee69 import get_topology_case
    src = Path(source_dir) / "network_inputs" / "TP1"
    files = dict(line_data=src / "line_data.csv", gridtensor_line_data=src / "gridtensor_line.csv")
    recorded = input_shas(manifest_env(source_manifest, "TP1"))
    dst = Path(out_dir) / "network_inputs" / "TP1"
    dst.mkdir(parents=True, exist_ok=False)
    report = {}
    for key, p in files.items():
        if not p.is_file():
            raise ReplayError(f"archived TP1 network input missing: {p}")
        sha = sha256_file(p)
        if sha != recorded.get(key):
            raise ReplayError(f"archived {p.name} sha256 {sha} != original run manifest {recorded.get(key)}")
        target = dst / p.name
        shutil.copyfile(p, target)
        if sha256_file(target) != sha:
            raise ReplayError(f"copy of {p} changed its content")
        report[key] = dict(archived=str(p.resolve()), copy=str(target.resolve()), sha256=sha)
    cfg = copy.deepcopy(base_config)
    case = get_topology_case("TP1")
    cfg["case_name"] = "ieee69_TP1"
    cfg["paths"]["line_data"] = dst / "line_data.csv"
    cfg["paths"]["gridtensor_line_data"] = dst / "gridtensor_line.csv"
    cfg["topology_case"] = topology_case_dict("TP1", case["description"], case["open_edges"], case["close_edges"], dst)
    return cfg, report


def backup_cross_check(base_config, backup_dir, archived_report, out_dir):
    """Optional: regenerate TP1 from the pre-TP10 backup of the source CSVs and compare with the archive."""
    from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
    b = Path(backup_dir)
    cfg = copy.deepcopy(base_config)
    cfg["paths"]["line_data"] = b / "line_data.csv"
    cfg["paths"]["gridtensor_line_data"] = b / "gridtensor_line.csv"
    for p in (cfg["paths"]["line_data"], cfg["paths"]["gridtensor_line_data"]):
        if not Path(p).is_file():
            return dict(status="NOT CHECKED", reason=f"{p} missing")
    built = build_topology_config("TP1", base_config=cfg, output_root=Path(out_dir) / "backup_regenerated", validate=True)
    res = {k: sha256_file(built["paths"][k]) == archived_report[k]["sha256"] for k in LINE_KEYS}
    return dict(status="MATCH" if all(res.values()) else "DIFFERS", per_file=res,
                backup_files={k: sha256_file(cfg["paths"][k]) for k in LINE_KEYS})


def active_edges(env):
    df = env.grid.in_service_line_df
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(df["from_bus"], df["to_bus"])}


def check_tp10(config, env):
    """TP10 as specified: open (62,63), close (26,64); tie R/X/rating; 69 buses, 68 active, 9 batteries."""
    tc = config["topology_case"]
    norm = lambda edges: sorted(tuple(sorted(map(int, e))) for e in edges)
    line = pd.read_csv(config["paths"]["line_data"])
    gt = pd.read_csv(config["paths"]["gridtensor_line_data"])
    a, b = TP10["close_edges"][0]
    tie = line[((line["from_bus"] == a) & (line["to_bus"] == b)) | ((line["from_bus"] == b) & (line["to_bus"] == a))]
    gtie = gt[((gt["FROM"] == a + 1) & (gt["TO"] == b + 1)) | ((gt["FROM"] == b + 1) & (gt["TO"] == a + 1))]
    act = active_edges(env)
    checks = dict(
        open_edges=norm(tc["open_edges"]) == norm(TP10["open_edges"]),
        close_edges=norm(tc["close_edges"]) == norm(TP10["close_edges"]),
        num_buses=int(env.num_buses) == TP10["num_buses"],
        active_branches=len(act) == TP10["active_branches"],
        batteries=len(env.batteries) == N_AGENTS == int(env.num_agents),
        opened_edge_inactive=tuple(TP10["open_edges"][0]) not in act,
        closed_edge_active=tuple(TP10["close_edges"][0]) in act,
        tie_line_data=(len(tie) == 1 and np.isclose(tie["r_ohm"].iloc[0], TP10["r_ohm"]) and np.isclose(tie["x_ohm"].iloc[0], TP10["x_ohm"])
                       and np.isclose(tie["max_current_a"].iloc[0], TP10["max_current_a"]) and float(tie["status"].iloc[0]) == 1.0),
        tie_gridtensor=(len(gtie) == 1 and np.isclose(gtie["R"].iloc[0], TP10["r_ohm"]) and np.isclose(gtie["X"].iloc[0], TP10["x_ohm"])
                        and float(gtie["STATUS"].iloc[0]) == 1.0))
    if not all(checks.values()):
        raise ReplayError(f"TP10 network check failed: {checks}")
    return checks


def compare_input_files(env_files, reference, keys, label):
    diff = {k: (env_files.get(k), reference.get(k)) for k in keys if env_files.get(k) != reference.get(k)}
    if diff:
        raise ReplayError(f"{label}: input files differ {diff}")
    return {k: env_files.get(k) for k in keys}


# ------------------------------------------------------------------------------- the replay run (both evaluators)
def run_replay(module, model, args, policy):
    """Shared driver for the GNN and MLP replay evaluators (see their --help)."""
    from configs.ieee69_config import IEEE69_CONFIG
    from environments.primary_marl import make_primary_env
    from utility.evaluation_protocol import authorize, protocol_config, episode_plan
    from utility.provenance import input_files
    from utility.run_manifest import RunManifest

    target = args.target_topology.upper()
    sel = selection_info(args.selection, model) if args.selection else None
    source_dir = Path(args.source_dir) if args.source_dir else (
        DEFAULT_CSV_ROOT / "final" / "ieee69" / model / sel["selection_id"] if sel else None)
    if source_dir is None:
        raise ReplayError("--source_dir or --selection is required")
    source_csv = source_dir / "step_metrics.csv"
    src_manifest = read_manifest(source_dir)
    ep_args = episode_args(src_manifest)
    if ep_args["final"]:
        if sel is None:
            raise ReplayError("the source is a final test: --selection (the frozen selection.json) is required")
        check_source_selection(src_manifest, sel)
    # gates: TP10 replay only after a passing FULL identity validation of the same controller/selection
    gate = None
    if target != "TP1":
        gate = check_gate(args.identity_report, "identity", model, source_csv, args.expected_episodes)
        if args.max_episodes is None or args.max_episodes > 1:
            gate = dict(identity=gate, smoke=check_gate(args.smoke_report, "target", model, source_csv, 1, target=target))
    actions, episodes, src_report = load_source_actions(source_csv, model, args.expected_episodes)
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
    base = copy.deepcopy(IEEE69_CONFIG)
    net, feedback = {}, None
    if target == "TP1":
        cfg, net["archived_tp1"] = archived_tp1_config(base, source_dir, src_manifest, out)
        if args.pre_tp10_backup:
            net["backup_cross_check"] = backup_cross_check(base, args.pre_tp10_backup, net["archived_tp1"], out)
    else:
        from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
        cfg = build_topology_config(case_name=target, base_config=base, output_root=out / "network_inputs", validate=True)
        if args.feedback_dir:
            feedback = Path(args.feedback_dir)
        elif args.feedback_job:
            feedback = resolve_feedback_dir(args.feedback_job)
        else:
            raise ReplayError("TP10 replay needs --feedback_dir or --feedback_job (to match its network and reset settings)")
    cfg = protocol_config(cfg, protocol)
    env = make_primary_env(config=cfg, mode=ep_args["mode"], seed=ep_args["seed"])
    files = {k: (v or {}).get("sha256") for k, v in input_files(env).items()}
    src_tp1 = input_shas(manifest_env(src_manifest, "TP1"))
    non_line = sorted(set(src_tp1) - set(LINE_KEYS))
    if target == "TP1":
        net["input_files_vs_original_TP1"] = compare_input_files(files, src_tp1, sorted(src_tp1), "TP1 identity vs original TP1")
    else:
        net["tp10_checks"] = check_tp10(cfg, env)
        net["non_line_inputs_vs_original_TP1"] = compare_input_files(files, src_tp1, non_line, "TP10 replay vs original TP1 (data)")
        fb_manifest = read_manifest(feedback)
        fb_args = episode_args(fb_manifest)
        if sel:
            check_source_selection(fb_manifest, sel)
        if (fb_args["seed"], fb_args["episodes"], fb_args["mode"]) != (ep_args["seed"], ep_args["episodes"], ep_args["mode"]):
            raise ReplayError(f"feedback reset settings {fb_args} differ from source {ep_args}")
        fb_tp = input_shas(manifest_env(fb_manifest, target))
        net["tp10_vs_feedback_run"] = compare_input_files(files, fb_tp, sorted(fb_tp), f"{target} replay vs {target} feedback network")
        net["feedback_dir"] = str(feedback.resolve())
    run.add_environment(env, topology=target)
    starts, seeds, _ = episode_plan(env, ep_args["episodes"], ep_args["seed"], protocol)
    if [int(s) for s in starts] != src_report["start_indices"]:
        raise ReplayError("episode plan of the replay environment differs from the source episodes")
    if feedback is not None:
        fb_starts = sorted(int(s) for s in read_filtered(feedback / "step_metrics.csv", target, policy, ["start_index"])[0]["start_index"].unique())
        if fb_starts != src_report["start_indices"]:
            raise ReplayError("TP10 feedback start indices differ from the TP1 source")
    agent = ReplayAgent(env, actions)
    rows = []
    condition = "TP1_to_TP1_identity_replay" if target == "TP1" else f"TP1_to_{target}_action_replay"
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
        condition=condition, model=model, policy_label=policy, target_topology=target, source_topology="TP1",
        replay_semantics=("recorded TP1 actor output (marl_raw_action, float32) returned by ReplayAgent.select_action inside the "
                          "original evaluate_one_episode; no actor inference; observations ignored; env.step recomputes "
                          "physical clipping, DOE, backup, power flow and SOC"),
        source=dict(src_report, source_dir=str(source_dir.resolve()), source_manifest_sha256=sha256_file(source_dir / "run_manifest.json")),
        selection=sel, training_run=(sel or {}).get("run"), reset=dict(seed=ep_args["seed"], episodes_arg=ep_args["episodes"],
        reset_seeds=[int(seeds[e]) for e, _ in episodes[:n_run]]), mode=ep_args["mode"], final_test=ep_args["final"],
        episode_index_mapping=[dict(episode_id=e, start_index=s) for e, s in episodes[:n_run]], episodes_replayed=n_run,
        episodes_available=len(episodes), actor_calls=agent.calls, network=net, gates=gate,
        target_input_files=files, created_utc=datetime.now(timezone.utc).isoformat(), evaluator=module.__name__,
        output_dir=str(out.resolve())))
    run.data["replay"] = dict(condition=condition, episodes_replayed=n_run)
    run.finish()
    print("REPLAY OUTPUT", out.resolve(), flush=True)
    return out


def check_gate(report_path, kind, model, source_csv, episodes, target=None):
    """Execution-order gate: a PASS validation report of the required kind for the same source actions."""
    if not report_path:
        raise ReplayError(f"a passing {kind} validation report is required before this replay (--{'identity' if kind == 'identity' else 'smoke'}_report)")
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


def add_replay_arguments(parser):
    parser.add_argument("--selection", help="frozen selection.json of the controller (required for final-test sources)")
    parser.add_argument("--source_dir", help="original TP1 evaluation directory (default csv_ieee69/final/ieee69/<model>/<selection_id>)")
    parser.add_argument("--target_topology", required=True, help="TP1 (identity) or TP10")
    parser.add_argument("--max_episodes", type=int, default=None, help="replay only the first N episodes (smoke test: 1)")
    parser.add_argument("--expected_episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_id", default=None, help="output sub-directory name (default smoke_/full_<UTC>)")
    parser.add_argument("--pre_tp10_backup", default=None, help="optional misc/ieee69_before_tp10_<stamp>/ cross-check (TP1 only)")
    parser.add_argument("--feedback_dir", default=None, help="TP10 feedback result directory (TP10 only)")
    parser.add_argument("--feedback_job", default=None, help="or its final_test_learned SLURM job id (resolved from the log)")
    parser.add_argument("--identity_report", default=None, help="PASS identity validation (full) required for TP10")
    parser.add_argument("--smoke_report", default=None, help="PASS TP10 smoke validation required for the full TP10 replay")
    parser.add_argument("--cpu", action="store_true", help="accepted for symmetry; no network is evaluated")
    return parser
