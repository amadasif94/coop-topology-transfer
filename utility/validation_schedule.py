"""Fixed validation episode schedules (F16). Validation split only; TP1 only.

S-SEL-v1  checkpoint selection: ALL validation days (Oct 1 - Nov 30, 61 per feeder), shared
          by every checkpoint and every comparable controller.
S-PER-v1  periodic in-training validation: 8 fixed days per feeder (4 October, 4 November) at
          the 12.5/37.5/62.5/87.5 % rank positions of dataset-only daily PV energy within each
          month. Frozen below as dates (approved 2026-10-21); S-PER-v1 is a subset of S-SEL-v1.

Each episode records its id, start index, first/last timestamp and reset seed. The seed is
SEED_BASE + the day's position in the 61-day pool, so a day has the same seed in both schedules.
Held-out topologies (TP2 onward) are refused: they are reserved for zero-shot final assessment.
"""
import hashlib
import json

import pandas as pd

SELECTION_SCHEDULE_ID = "S-SEL-v1"
PERIODIC_SCHEDULE_ID = "S-PER-v1"
SEED_BASE = 20261000
SELECTION_EPISODES = 61
VALIDATION_MONTHS = (10, 11)
PERIODIC_DATES = {
    33: ("2018-10-13", "2018-10-16", "2018-10-21", "2018-10-25",
         "2018-11-03", "2018-11-05", "2018-11-11", "2018-11-13"),
    69: ("2018-10-03", "2018-10-11", "2018-10-17", "2018-10-25",
         "2018-11-03", "2018-11-05", "2018-11-11", "2018-11-13"),
}


def topology_of(env):
    return str((env.config.get("topology_case") or {}).get("case_name", "TP1")).upper()


def require_tp1(env, purpose):
    if topology_of(env) != "TP1":
        raise ValueError(f"{purpose} must use TP1; {topology_of(env)} is a held-out topology reserved "
                         "for zero-shot final assessment")


def validation_pool(env):
    """Every validation episode, checked to lie entirely inside October-November."""
    if env.mode != "val":
        raise ValueError(f"validation schedules require mode='val' (got {env.mode!r})")
    require_tp1(env, "validation scheduling")
    df, col = env.time_series["df"], env.time_series["datetime_col"]
    val = set(int(i) for i in env.splits["val_indices"])
    length = int(env.episode_length)
    pool = []
    for k, start in enumerate(sorted(int(s) for s in env.episode_start_indices)):
        stamps = pd.to_datetime(df.loc[start:start + length - 1, col])
        first, last = stamps.iloc[0], stamps.iloc[-1]
        ok = (len(stamps) == length and all(i in val for i in range(start, start + length))
              and first.month in VALIDATION_MONTHS and last.month in VALIDATION_MONTHS
              and first.date() == last.date() and (first.hour, first.minute) == (0, 0))
        if not ok:
            raise ValueError(f"validation episode at {start} ({first} - {last}) is not a full Oct-Nov day")
        pool.append(dict(episode_id="val_" + first.strftime("%Y-%m-%d"), start_index=start,
                         start_time=str(first), end_time=str(last), seed=SEED_BASE + k))
    return pool


def _schedule(schedule_id, env, episodes, rule):
    body = dict(schedule_id=schedule_id, feeder=f"IEEE{env.num_buses}", topology="TP1", split="val",
                episode_length=int(env.episode_length), rule=rule, episodes=episodes)
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def selection_schedule(env):
    pool = validation_pool(env)
    if len(pool) != SELECTION_EPISODES:
        raise ValueError(f"expected {SELECTION_EPISODES} validation days, found {len(pool)}")
    return _schedule(SELECTION_SCHEDULE_ID, env, pool, "all validation days (2018-10-01 .. 2018-11-30)")


def periodic_schedule(env):
    dates = PERIODIC_DATES.get(int(env.num_buses))
    if dates is None:
        raise ValueError(f"no periodic validation schedule for IEEE{env.num_buses}")
    by_id = {e["episode_id"]: e for e in validation_pool(env)}
    missing = [d for d in dates if "val_" + d not in by_id]
    if missing:
        raise ValueError(f"periodic validation days not in the validation pool: {missing}")
    return _schedule(PERIODIC_SCHEDULE_ID, env, [by_id["val_" + d] for d in dates],
                     "4 Oct + 4 Nov days at 12.5/37.5/62.5/87.5 % daily-PV-energy rank per month")


def schedule_summary(schedule):
    """Compact manifest form (episode list kept; it is the evidence of what was evaluated)."""
    return {k: schedule[k] for k in ("schedule_id", "sha256", "feeder", "topology", "split",
                                     "episode_length", "rule", "episodes")}
