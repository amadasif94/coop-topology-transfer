"""Completeness of F-MPC models and forecasts WITHOUT importing PyTorch, AutoGluon or the RL configuration. Maintained.

    python -m evaluation.forecast_mpc.status --feeder ieee33 --stage oct --output-root results/soc_feasible_affine_v2
                                             [--chunks 6] [--kind models|forecasts|both]

Python standard library only (safe on the RIT login node, where importing torch can fail). Reads file names and JSON
records; writes nothing. Per target (load_node_1..N, pv_node_1..N):
  complete         <target>.json record present
  saved_no_record  predictor folder with a finished AutoGluon save (predictor.pkl, learner.pkl, metadata.json,
                   version.txt, non-empty models/) but no record: `train ... --verify-saved` checks it read-only and
                   `train ... --skip-completed` validates and completes it (models only)
  incomplete       anything else without a record: an unfinished fit or a task that is still running
  missing          nothing yet
With --chunks N the array task (0..N-1) of every non-complete target is printed, to resubmit only those tasks.
Exit code 0 only if every target of the requested kind(s) is complete.
"""
import argparse
import json
import time
from pathlib import Path

STAGES = ('oct', 'nov', 'dec')              # evaluation.forecast_mpc.common.STAGES
NUM_BUSES = dict(ieee33=33, ieee69=69)      # evaluation.forecast_mpc.common.NUM_BUSES
LABEL = 'forecast_mpc'


def targets(feeder):
    n = NUM_BUSES[feeder]
    return [f'load_node_{i}' for i in range(1, n + 1)] + [f'pv_node_{i}' for i in range(1, n + 1)]


def chunk_index(position, total, chunks):
    """Array task of target `position` under numpy.array_split(targets, chunks) (common.select_targets)."""
    size, extra = divmod(total, chunks)
    boundary = extra * (size + 1)
    return position // (size + 1) if position < boundary else extra + (position - boundary) // size


SAVED_ARTIFACTS = ('predictor.pkl', 'learner.pkl', 'metadata.json', 'version.txt')   # train.SAVED_ARTIFACTS


def saved_complete(folder):
    """A finished AutoGluon save: the four files and a non-empty models/ folder (train.validate_saved checks more)."""
    models = folder / 'models'
    return (all((folder / a).is_file() for a in SAVED_ARTIFACTS) and models.is_dir()
            and any(p.is_file() for p in models.rglob('*')))


def classify(directory, target, kind):
    record = directory / f'{target}.json'
    if record.is_file():
        return 'complete'
    artifact = directory / (target if kind == 'models' else f'{target}.npz')
    if kind == 'models' and artifact.is_dir() and saved_complete(artifact):
        return 'saved_no_record'
    return 'incomplete' if artifact.exists() else 'missing'


def report(root, feeder, stage, kind, chunks=None):
    directory = Path(root) / feeder / LABEL / kind / stage
    names = targets(feeder)
    states = {t: classify(directory, t, kind) for t in names}
    counts = {s: sum(v == s for v in states.values()) for s in ('complete', 'saved_no_record', 'incomplete', 'missing')}
    ok = counts['complete'] == len(names)
    print(f'FMPC_STATUS {kind} {feeder} {stage}: {"COMPLETE" if ok else "INCOMPLETE"} {counts["complete"]}/{len(names)} '
          f'| saved_no_record {counts["saved_no_record"]} | incomplete {counts["incomplete"]} | missing '
          f'{counts["missing"]} | {directory}', flush=True)
    now = time.time()
    for t, s in states.items():
        if s in ('saved_no_record', 'incomplete'):
            art = directory / (t if kind == 'models' else f'{t}.npz')
            files = [p for p in art.rglob('*') if p.is_file()] if art.is_dir() else ([art] if art.exists() else [])
            age = (now - max(p.stat().st_mtime for p in files)) / 60 if files else None
            board = (directory / f'{t}.leaderboard.csv').is_file() if kind == 'models' else None
            print(f'  {s:15s} {t}' + (f' (task {chunk_index(names.index(t), len(names), chunks)})' if chunks else '')
                  + (f' last write {age:.0f} min ago' if age is not None else '')
                  + (f' leaderboard {"yes" if board else "no"}' if board is not None else ''), flush=True)
    if chunks:
        tasks = sorted({chunk_index(i, len(names), chunks) for i, t in enumerate(names) if states[t] != 'complete'})
        print(f'  array tasks with work left (of {chunks}): {tasks}', flush=True)
    records = [json.loads((directory / f'{t}.json').read_text(encoding='utf-8')) for t, s in states.items()
               if s == 'complete']
    if records:
        def values(key):
            return sorted({str(r.get(key)) for r in records})
        print(f'  records: stage {values("stage")} cutoff {values("cutoff_time")} rows {values("train_rows")}'
              + (f' recovered {sum(bool(r.get("recovered")) for r in records)}' if kind == 'models' else '')
              + (f' dropped {values("dropped_features")}' if kind == 'models' else ''), flush=True)
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=sorted(NUM_BUSES))
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--output-root', required=True, help='e.g. results/soc_feasible_affine_v2')
    p.add_argument('--kind', choices=('models', 'forecasts', 'both'), default='both')
    p.add_argument('--chunks', type=int, help='array size used for training/generation (to name tasks to resubmit)')
    a = p.parse_args(argv)
    kinds = ('models', 'forecasts') if a.kind == 'both' else (a.kind,)
    ok = all([report(a.output_root, a.feeder, a.stage, k, a.chunks) for k in kinds])
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
