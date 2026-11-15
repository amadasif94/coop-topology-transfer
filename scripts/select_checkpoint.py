"""Frozen, validation-only checkpoint selection (F2, schedule F16).

    python scripts/select_checkpoint.py --feeder ieee33 --model gcn --run run_0 \
        --sweep_dir results/csv/selection/gcn/run_0 [--output_root results/selections]
        [--supersede <latest selection.json> --reason "..."]

Reads every step_*/ result of ONE training run's checkpoint-selection sweep and accepts a result
only if its run manifest shows: finished, run_purpose 'checkpoint_selection', split 'val', not a
final test, TP1 only, the S-SEL-v1 schedule with the sha256 recomputed here, and a checkpoint file
whose sha256 still equals the one recorded at evaluation time. Any TP2+/test-split/other-schedule
result is REJECTED (the whole selection fails), never silently skipped.

Eligibility: all 61 scheduled episodes completed and failure-free (F13 economic_totals_valid) and
the pre-existing aggregate safety rule (eligible_checkpoint_rows: no failed/incomplete episodes,
final network feasibility rate 1, no failed PF solves). Metric: mean episode reward
(total_reward_mean) over the 61 days; ties go to the earliest step. Reliability: day-bootstrap
top-1 frequency and the paired gap to the runner-up. The record is written once, under a new
versioned id; an existing record is never overwritten or silently replaced.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.primary_marl import make_primary_env
from utility.evaluation_protocol import SELECTION_SCHEMA, ProtocolError, sha256_file, write_selection
from utility.evaluation_reporting import eligible_checkpoint_rows
from utility.validation_schedule import SELECTION_SCHEDULE_ID, selection_schedule

FEEDERS = dict(ieee33=IEEE33_CONFIG, ieee69=IEEE69_CONFIG)
LEARNED = ("maddpg", "mlp_maddpg")
METRIC = "mean over the S-SEL-v1 validation days of the episode total_reward_mean (valid episodes only)"


def load_result(step_dir, schedule):
    """Return (step, checkpoint record, per-episode rewards, eligible, reason); raise on protocol violations."""
    manifest_path = step_dir / "run_manifest.json"
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    proto = man.get("evaluation_protocol") or {}
    where = str(step_dir)
    if man.get("status") != "finished":
        raise ProtocolError(f"{where}: evaluation did not finish")
    if proto.get("run_purpose") != "checkpoint_selection" or proto.get("data_split") != "val" or proto.get("final_test"):
        raise ProtocolError(f"{where}: not a validation checkpoint-selection run (protocol={proto})")
    topologies = [str(t).upper() for t in proto.get("topologies", [])]
    env_topologies = [str(e.get("topology", "")).upper() for e in man.get("environments", [])]
    if topologies != ["TP1"] or any(t != "TP1" for t in env_topologies) or not env_topologies:
        raise ProtocolError(f"{where}: selection results must come from TP1 only "
                            f"(held-out topologies are reserved for zero-shot assessment): {topologies or env_topologies}")
    if any(e.get("mode") != "val" for e in man["environments"]):
        raise ProtocolError(f"{where}: environment not on the validation split")
    sched = man.get("validation_schedule") or {}
    if sched.get("schedule_id") != SELECTION_SCHEDULE_ID or sched.get("sha256") != schedule["sha256"]:
        raise ProtocolError(f"{where}: schedule {sched.get('schedule_id')} / {sched.get('sha256')} "
                            f"is not {SELECTION_SCHEDULE_ID} / {schedule['sha256']}")
    ck = man.get("checkpoint_file") or {}
    if not ck.get("exists") or not Path(ck["path"]).is_file() or sha256_file(ck["path"]) != ck.get("sha256"):
        raise ProtocolError(f"{where}: checkpoint {ck.get('path')} is missing or changed since evaluation")
    ep = pd.read_csv(step_dir / "episode_summary.csv")
    ep = ep[ep["policy"].astype(str).str.lower().isin(LEARNED)]
    agg = pd.read_csv(step_dir / "aggregate_summary.csv")
    agg = agg[agg["policy"].astype(str).str.lower().isin(LEARNED)]
    expected = [e["start_index"] for e in schedule["episodes"]]
    ep = ep.set_index("start_index").reindex(expected)
    rewards = ep["total_reward_mean"].to_numpy(dtype=float)
    reasons = []
    if ep["economic_totals_valid"].isna().any() or not ep["economic_totals_valid"].astype(bool).all():
        reasons.append("failed/incomplete validation episode(s)")
    if len(agg) != 1 or eligible_checkpoint_rows(agg).empty:
        reasons.append("aggregate safety rule (eligible_checkpoint_rows) not met")
    if not np.isfinite(rewards).all():
        reasons.append("non-finite episode reward")
    step = int(step_dir.name.replace("step_", ""))
    env0 = man["environments"][0]
    provenance = dict(configuration_sha256=env0.get("configuration_sha256"),
                      normalization_sha256=(env0.get("normalization_identity") or {}).get("sha256"),
                      interface_identity=env0.get("interface_identity"),
                      source_files_sha256=(env0.get("source_files") or {}).get("sha256"),
                      input_files={k: v.get("sha256") for k, v in (env0.get("input_files") or {}).items()},
                      evaluation_manifest=str(manifest_path), evaluation_manifest_sha256=sha256_file(manifest_path),
                      source_revision=man.get("source_revision"))
    return dict(step=step, checkpoint_path=ck["path"], checkpoint_sha256=ck["sha256"], rewards=rewards,
                eligible=not reasons, reasons=reasons, result_dir=where, provenance=provenance)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feeder", required=True, choices=sorted(FEEDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sweep_dir", required=True, type=Path)
    ap.add_argument("--output_root", default="results/selections", type=Path)
    ap.add_argument("--supersede", default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=20261021)
    a = ap.parse_args(argv)
    env = make_primary_env(copy.deepcopy(FEEDERS[a.feeder]), mode="val", seed=0)
    schedule = selection_schedule(env)
    step_dirs = sorted((p for p in a.sweep_dir.glob("step_*") if p.is_dir()), key=lambda p: int(p.name[5:]))
    if not step_dirs:
        raise ProtocolError(f"no step_* results under {a.sweep_dir}")
    results = [load_result(p, schedule) for p in step_dirs]
    eligible = [r for r in results if r["eligible"]]
    if not eligible:
        raise ProtocolError("no eligible checkpoint (every candidate failed an episode or the safety rule)")
    means = np.array([r["rewards"].mean() for r in eligible])
    best = int(np.flatnonzero(means == means.max())[0])        # earliest step among ties
    chosen = eligible[best]
    R = np.stack([r["rewards"] for r in eligible])
    rng = np.random.default_rng(a.bootstrap_seed)
    n = R.shape[1]
    top = [int(np.argmax(R[:, rng.integers(0, n, n)].mean(axis=1))) for _ in range(a.bootstrap)]
    reliability = dict(bootstrap_resamples=a.bootstrap, bootstrap_seed=a.bootstrap_seed,
                       bootstrap_top1_frequency=float(np.mean(np.array(top) == best)))
    if len(eligible) > 1:
        second = int(np.argsort(-means, kind="stable")[1])
        diff = R[best] - R[second]
        reliability.update(runner_up_step=eligible[second]["step"], runner_up_gap=float(diff.mean()),
                           runner_up_paired_se=float(diff.std(ddof=1) / np.sqrt(n)))
    record = dict(
        schema=SELECTION_SCHEMA, feeder=a.feeder, model=a.model.lower(), run=a.run, topology="TP1",
        data_split="val", run_purpose="checkpoint_selection", metric=METRIC,
        eligibility="all scheduled episodes completed and failure-free; eligible_checkpoint_rows aggregate rule",
        schedule=dict(schedule_id=schedule["schedule_id"], sha256=schedule["sha256"], episodes=schedule["episodes"]),
        selected=dict(step=chosen["step"], checkpoint_path=chosen["checkpoint_path"],
                      checkpoint_sha256=chosen["checkpoint_sha256"], metric_value=float(means[best]),
                      result_dir=chosen["result_dir"]),
        configuration_provenance=chosen["provenance"], reliability=reliability,
        candidates=[dict(step=r["step"], checkpoint_sha256=r["checkpoint_sha256"], eligible=r["eligible"],
                         reasons=r["reasons"], metric_value=float(r["rewards"].mean()) if r["eligible"] else None,
                         result_dir=r["result_dir"]) for r in results])
    path = write_selection(record, a.output_root, a.supersede, a.reason)
    print("FROZEN SELECTION:", path)
    print("  selected step %d  metric %.6f  checkpoint sha256 %s" % (chosen["step"], means[best], chosen["checkpoint_sha256"]))
    print("  bootstrap top-1 frequency %.3f over %d resamples" % (reliability["bootstrap_top1_frequency"], a.bootstrap))
    print("  selection.json sha256", sha256_file(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
