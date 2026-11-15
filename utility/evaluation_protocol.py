"""Evaluation protocol (F2): data-split authorization, run purpose and frozen checkpoint selection.

Rules (enforced here and, for the split, again inside DESSEnv so no entry point can bypass them):
* The December test split (mode 'test', and mode 'all' which contains it) is used ONLY by an
  explicit final test (--final). Development runs of every controller use train/val data.
* A final test of a LEARNED controller requires a frozen selection.json produced by
  scripts/select_checkpoint.py; the evaluated checkpoint must be byte-identical (sha256) to the
  selected one. Non-learning controllers (zero, random, MPC, SMPC) record 'not_applicable'.
* Checkpoint selection uses TP1 validation only, on the fixed S-SEL-v1 schedule (F16).
  Held-out topologies (TP2 onward) are reserved for zero-shot final assessment: any non-final
  run that names them, and any selection record built from them, is refused.
* Selection records are never overwritten. A new record for the same feeder/model/run needs an
  explicit --supersede of the latest record plus a reason, and gets a new versioned id.
"""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from utility.validation_schedule import SELECTION_SCHEDULE_ID, selection_schedule

RUN_PURPOSES = ("development", "checkpoint_selection", "baseline_validation", "final_test")
SELECTION_SCHEMA = "gmarl-checkpoint-selection/1"
NOT_APPLICABLE = "not_applicable"
LEARNED_POLICIES = ("maddpg", "mlp_maddpg")


class ProtocolError(ValueError):
    pass


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_protocol_arguments(parser, learned=True):
    parser.add_argument("--final", action="store_true",
                        help="final test on the December test split (explicit authorization)")
    parser.add_argument("--run_purpose", default=None, choices=RUN_PURPOSES,
                        help="default: final_test with --final, otherwise development")
    if learned:
        parser.add_argument("--selection", default=None,
                            help="frozen selection.json; required with --final for learned controllers")


def authorize(mode=None, final=False, run_purpose=None, learned=True, topologies=("TP1",), selection=None,
              checkpoint=None):
    """Return the protocol record for one evaluation, or raise ProtocolError."""
    mode = mode or ("test" if final else "val")
    purpose = run_purpose or ("final_test" if final else "development")
    topologies = [str(t).upper() for t in topologies]
    if final:
        if mode != "test":
            raise ProtocolError(f"--final evaluates the December test split; mode {mode!r} is not allowed")
        if purpose != "final_test":
            raise ProtocolError(f"--final requires run_purpose 'final_test' (got {purpose!r})")
    else:
        if mode in ("test", "all"):
            raise ProtocolError("The December test split requires explicit final-test authorization (--final)")
        if purpose == "final_test":
            raise ProtocolError("run_purpose 'final_test' requires --final")
        held_out = [t for t in topologies if t != "TP1"]
        if held_out:
            raise ProtocolError(f"Held-out topologies {held_out} are reserved for zero-shot final assessment; "
                                "development, tuning and selection use TP1 only")
        if purpose in ("checkpoint_selection", "baseline_validation") and mode != "val":
            raise ProtocolError(f"run_purpose {purpose!r} uses the validation split (mode 'val')")
        if purpose == "checkpoint_selection" and not learned:
            raise ProtocolError("checkpoint selection applies to learned controllers only")
    if learned and final:
        if not selection:
            raise ProtocolError("A final test of a learned controller requires --selection <frozen selection.json>")
        sel = verify_selection(selection, checkpoint)
    elif learned and selection:
        raise ProtocolError("--selection is used only with --final")
    else:
        sel = None if learned else NOT_APPLICABLE
    return dict(run_purpose=purpose, data_split=mode, final_test=bool(final), topologies=topologies,
                learned_controller=bool(learned), selection=sel)


def authorize_args(args, learned=True, topologies=("TP1",)):
    """authorize() from parsed evaluator arguments; fills --checkpoint from the selection."""
    protocol = authorize(getattr(args, "mode", None), getattr(args, "final", False), getattr(args, "run_purpose", None),
                         learned, topologies, getattr(args, "selection", None), getattr(args, "checkpoint", None))
    if learned and protocol["final_test"] and not getattr(args, "checkpoint", None):
        args.checkpoint = protocol["selection"]["checkpoint_path"]
    if learned and not getattr(args, "checkpoint", None):
        raise ProtocolError("--checkpoint is required (or --selection with --final)")
    args.mode = protocol["data_split"]
    return protocol


def protocol_config(config, protocol):
    """Configuration for DESSEnv; final tests carry the explicit test-split authorization."""
    config = copy.deepcopy(config)
    if protocol["final_test"]:
        sel = protocol["selection"]
        config["evaluation_protocol"] = dict(final_test=True, run_purpose="final_test",
                                             selection_sha256=sel["selection_sha256"] if isinstance(sel, dict) else sel)
    return config


def selection_provenance(protocol, policies):
    """Per-policy selection provenance: the frozen record for learned policies, else not_applicable."""
    return {p: (protocol["selection"] if p in LEARNED_POLICIES and protocol["learned_controller"] else NOT_APPLICABLE)
            for p in policies}


def episode_plan(env, episodes, seed, protocol):
    """(start indices, reset seeds, schedule or None).

    Selection and baseline-validation runs (and --episodes schedule) use the fixed S-SEL-v1
    schedule; other runs keep the existing first-N / all behaviour with seed+i.
    """
    if protocol["run_purpose"] in ("checkpoint_selection", "baseline_validation") or str(episodes) == "schedule":
        if str(episodes) not in ("schedule", "all"):
            raise ProtocolError(f"{protocol['run_purpose']} evaluates the full {SELECTION_SCHEDULE_ID} schedule; "
                                f"use --episodes schedule (got {episodes!r})")
        schedule = selection_schedule(env)
        eps = schedule["episodes"]
        return [e["start_index"] for e in eps], [e["seed"] for e in eps], schedule
    starts = list(env.episode_start_indices) if str(episodes) == "all" else list(env.episode_start_indices[:int(episodes)])
    return starts, [int(seed) + i for i in range(len(starts))], None


# ----------------------------------------------------------------------------- frozen selections
def verify_selection(path, checkpoint=None):
    """Validate a frozen selection.json (and optionally the checkpoint to be evaluated)."""
    path = Path(path)
    raw = path.read_bytes()
    record = json.loads(raw.decode("utf-8"))
    problems = []
    if record.get("schema") != SELECTION_SCHEMA:
        problems.append(f"schema {record.get('schema')!r}")
    if str(record.get("topology", "")).upper() != "TP1":
        problems.append(f"topology {record.get('topology')!r} (selection must use TP1 validation only)")
    if record.get("data_split") != "val" or record.get("run_purpose") != "checkpoint_selection":
        problems.append("not a validation checkpoint-selection record")
    if (record.get("schedule") or {}).get("schedule_id") != SELECTION_SCHEDULE_ID:
        problems.append(f"schedule {(record.get('schedule') or {}).get('schedule_id')!r}")
    selected = record.get("selected") or {}
    ck_path, ck_sha = selected.get("checkpoint_path"), selected.get("checkpoint_sha256")
    if not ck_path or not ck_sha:
        problems.append("no selected checkpoint")
    elif not Path(ck_path).is_file() or sha256_file(ck_path) != ck_sha:
        problems.append(f"selected checkpoint {ck_path} is missing or no longer matches sha256 {ck_sha}")
    if checkpoint is not None and ck_sha and (not Path(checkpoint).is_file() or sha256_file(checkpoint) != ck_sha):
        problems.append(f"--checkpoint {checkpoint} is not the frozen selected checkpoint (sha256 {ck_sha})")
    if problems:
        raise ProtocolError(f"Invalid selection record {path}: " + "; ".join(problems))
    return dict(selection_path=str(path.resolve()), selection_sha256=hashlib.sha256(raw).hexdigest(),
                selection_id=record["selection_id"], version=record["version"],
                checkpoint_path=ck_path, checkpoint_sha256=ck_sha, checkpoint_step=selected.get("step"),
                metric=record.get("metric"), metric_value=selected.get("metric_value"))


def selection_key_dir(root, feeder, model, run):
    return Path(root) / str(feeder).lower() / str(model).lower() / str(run)


def existing_selections(key_dir):
    return sorted(Path(key_dir).glob("*/selection.json"), key=lambda p: json.loads(p.read_text())["version"])


def write_selection(record, root, supersede=None, reason=None):
    """Write a NEW frozen selection record; never overwrite or silently replace an existing one."""
    key = selection_key_dir(root, record["feeder"], record["model"], record["run"])
    existing = existing_selections(key)
    if existing:
        latest = existing[-1]
        if supersede is None:
            raise FileExistsError(f"A frozen selection already exists: {latest}. Re-running selection does not "
                                  "replace it; pass --supersede <that file> --reason <text> to record a new version.")
        if Path(supersede).resolve() != latest.resolve():
            raise ProtocolError(f"--supersede must name the latest record {latest}")
        if not reason:
            raise ProtocolError("--supersede requires --reason")
        previous = json.loads(latest.read_text())
        record["version"] = int(previous["version"]) + 1
        record["supersedes"] = dict(selection_id=previous["selection_id"], path=str(latest),
                                    sha256=sha256_file(latest), reason=reason)
    elif supersede is not None:
        raise ProtocolError("--supersede given but no previous selection exists")
    else:
        record["version"] = 1
        record["supersedes"] = None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record["selection_id"] = "%s_%s_%s_v%d_%s" % (record["feeder"], record["model"], record["run"],
                                                 record["version"], stamp)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    target = key / record["selection_id"]
    target.mkdir(parents=True, exist_ok=False)
    with (target / "selection.json").open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=True, allow_nan=False)
    return target / "selection.json"
