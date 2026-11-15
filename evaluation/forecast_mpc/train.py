"""Train the F-MPC forecasting models: one AutoGluon TabularPredictor per bus and quantity. Maintained entry point.

    python -u -m evaluation.forecast_mpc.train --feeder ieee33 --stage oct [--chunk K --chunks N | --targets ...]
    python -u -m evaluation.forecast_mpc.train --feeder ieee33 --stage nov [--chunk K --chunks N]
    python -u -m evaluation.forecast_mpc.train --feeder ieee33 --stage dec [--chunk K --chunks N]
    python -u -m evaluation.forecast_mpc.train --feeder ieee33 --stage oct --status        (completeness only)

Monthly expanding-window walk-forward: stage 'oct' fits on 2018-01-01 .. 09-30, 'nov' on 2018-01-01 .. 10-31, 'dec' on
2018-01-01 .. 11-30, each a separate set of models. No row of the stage's month or later is used: every training target
row and every row its features read lies before the stage cutoff (00:00 on the 1st of the month), which the record
states and the code checks; AutoGluon's internal holdout is drawn from those rows too.
Each model predicts the target one 15-minute row ahead from its 672-row history window (evaluation.forecast_mpc.
features); multi-step forecasts are made recursively by evaluation.forecast_mpc.generate. Settings are frozen
(common.AUTOGLUON_SETTINGS: medium_quality, 300 s per model, MAE); only AutoGluon's internal holdout drawn from the
training rows chooses among its models. After fitting, only the best model is kept (delete_models, save_space).

Outputs (never overwritten): <root>/<feeder>/forecast_mpc/models/<stage>/<target>/ (predictor) and <target>.json
(training record: rows, time range, cutoff, series sha256, AutoGluon version, settings, predictor sha256) and
<target>.leaderboard.csv. An existing target is refused. With --skip-completed (always used by the Slurm arrays), a
target with a record is skipped, and a saved predictor WITHOUT a record (fitted, but the run stopped before the record
was written) is loaded read-only, validated and completed (recover_target); a folder without the complete AutoGluon
save (predictor.pkl, learner.pkl, metadata.json, version.txt, models/), saved by another AutoGluon version, that cannot
be loaded or fails validation, or written to within the last 15 minutes is refused and never deleted.
--verify-saved runs the same validation read-only (no record, no cleanup, no fit) so it can be checked first.

Feature schema: the input schema is FEATURES; AutoGluon may RETAIN fewer columns because it removes useless ones
(single-valued or duplicate). For PV, y_roll_min_96 and y_roll_min_672 are always 0 in the training rows (PV is zero
every night, so the minimum over any 24 h or 1-week window is 0) and are removed. The record keeps input_features,
retained_features and dropped_features (with the reason checked on the training frame); any other removal is refused.
"""
import argparse
from importlib import metadata
import time

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, MAINTAINED_PROTOCOL, maintained_output_root, write_json
from evaluation.forecast_mpc.common import (AUTOGLUON_SETTINGS, INFORMATION_RULE, MODEL_SCHEMA, STAGES, cutoff_time,
                                           dir_sha256, file_sha256, load_series, models_dir, print_status,
                                           select_targets, stage_status, target_kind_bus, training_rows)
from evaluation.forecast_mpc.features import FEATURES, LABEL, WINDOW, feature_frame, training_window


def training_frame(series, target, stage):
    """(DataFrame [y + FEATURES], target rows) of one target and stage; every row it reads is before the cutoff."""
    rows = training_rows(series, stage)
    y = series.target(target)
    frame = feature_frame(training_window(y, rows), series.timestamps[rows], series.price, rows)
    frame.insert(0, LABEL, y[rows])
    cutoff = cutoff_time(series, stage)
    if series.timestamps[rows.max()] >= cutoff or rows.min() < WINDOW:
        raise AssertionError('training rows outside [row 672, cutoff)')
    return frame, rows


def autogluon_version():
    for name in ('autogluon.tabular', 'autogluon'):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    import autogluon.tabular as agt
    return getattr(agt, '__version__', 'unknown')


def _model_best(predictor):
    best = getattr(predictor, 'model_best', None)
    return best() if callable(best) else best if best is not None else predictor.get_model_best()


def feature_schema(predictor, frame):
    """Input schema (the columns given to fit: FEATURES) versus the schema AutoGluon RETAINED after its own
    preprocessing. AutoGluon removes useless input columns (a single value, or an exact copy of another column) and
    feature_metadata_in lists only the retained ones; it still accepts the full input frame at predict time.
    Accepted only if (a) the input schema is exactly FEATURES, (b) the retained columns are a nonempty subset of it, and
    (c) every removed column is constant, or an exact duplicate of a retained column, in this stage's training frame, so
    no informative feature is lost. Anything else raises ValueError (incompatible model)."""
    original = getattr(predictor, 'original_features', None)
    original = list(original) if original is not None else None
    if original is not None and set(original) != set(FEATURES):
        raise ValueError(f'predictor input schema differs from the feature builder: {sorted(set(original) ^ set(FEATURES))}')
    retained = list(predictor.feature_metadata_in.get_features())
    unknown = sorted(set(retained) - set(FEATURES))
    if not retained or unknown:
        raise ValueError(f'predictor retained no features or unknown features {unknown}')
    x = frame[list(FEATURES)]
    dropped = {}
    for col in FEATURES:
        if col in retained:
            continue
        if x[col].nunique(dropna=False) == 1:
            dropped[col] = f'constant in the training frame (value {x[col].iloc[0]!r})'
            continue
        twin = next((r for r in retained if x[col].equals(x[r])), None)
        if twin is None:
            raise ValueError(f'AutoGluon dropped the informative feature {col!r} (neither constant nor a duplicate)')
        dropped[col] = f'exact duplicate of {twin} in the training frame'
    return dict(input_features=list(FEATURES), input_schema_source='predictor.original_features' if original is not None
                else 'feature builder (predictor exposes no original_features)', retained_features=retained,
                dropped_features=dropped)


def _cleanup(predictor):
    cleanup = []
    for name, call in (('delete_models', lambda: predictor.delete_models(models_to_keep='best', dry_run=False)),
                       ('save_space', lambda: predictor.save_space())):
        try:
            call()
            cleanup.append(f'{name}: ok')
        except Exception as exc:  # noqa: BLE001 - recorded; the fitted best model is unaffected
            cleanup.append(f'{name}: failed {type(exc).__name__}: {exc}')
    return cleanup


def _write_record(series, target, stage, directory, frame, rows, predictor, schema, time_limit, **extra):
    kind, bus = target_kind_bus(target)
    y = frame[LABEL].to_numpy(dtype=float)
    record = dict(
        schema=MODEL_SCHEMA, feeder=series.feeder, stage=stage, target=target, kind=kind, bus=bus, label=LABEL,
        features=list(FEATURES), **schema, train_rows=int(len(rows)), train_first_row=int(rows.min()),
        train_last_row=int(rows.max()), train_first_time=str(series.timestamps[rows.min()]),
        train_last_time=str(series.timestamps[rows.max()]), history_first_time=str(series.timestamps[rows.min() - WINDOW]),
        cutoff_time=str(cutoff_time(series, stage)),
        cutoff_note='no row at or after cutoff_time is a training target or feature input',
        series_path=str(series.path), series_sha256=series.sha256, autogluon_version=autogluon_version(),
        settings=dict(AUTOGLUON_SETTINGS, time_limit_s=AUTOGLUON_SETTINGS['time_limit_s'] if time_limit is None
                      else time_limit, time_limit_is_default=time_limit is None),
        model_best=_model_best(predictor), information_rule=INFORMATION_RULE,
        target_train_min=float(y.min()), target_train_max=float(y.max()), target_train_mean=float(y.mean()),
        clip_lower=0.0, clip_upper=float(y.max()) if kind == 'pv' else None,
        leaderboard_sha256=file_sha256(directory / f'{target}.leaderboard.csv'),
        model_dir_sha256=dir_sha256(directory / target), **extra)
    write_json(directory / f'{target}.json', record)
    return record


def _write_leaderboard(predictor, path):
    with path.open('x', encoding='utf-8', newline='') as f:
        predictor.leaderboard(silent=True).to_csv(f, index=False)


RECOVERY_QUIET_SECONDS = 900      # a predictor folder written to within 15 minutes may still be training: not recovered


SAVED_ARTIFACTS = ('predictor.pkl', 'learner.pkl', 'metadata.json', 'version.txt')   # files of a finished AutoGluon save
PREDICTION_CHECK_ROWS = 192


def validate_saved(series, target, stage, directory):
    """Read-only validation of a saved predictor that has no training record. Nothing is written or changed.

    Checks, in order: the saved artifacts of a finished AutoGluon fit (predictor.pkl, learner.pkl, metadata.json,
    version.txt and a non-empty models/ folder); no write in the last 15 minutes; the AutoGluon version in version.txt
    equals the installed one; a read-only load; label 'y', regression, MAE metric; the rows it was fitted on equal this
    stage's training rows (when AutoGluon exposes the count); the feature-schema rule (feature_schema: input schema =
    FEATURES, dropped columns only constant or duplicate in this stage's training frame); the best model appears in the
    saved leaderboard; finite predictions of the right length on training rows spread over the whole training period.
    Returns the validated pieces, or raises RuntimeError with every problem found."""
    model_path = directory / target
    missing = [a for a in SAVED_ARTIFACTS if not (model_path / a).is_file()]
    models = model_path / 'models'
    if not models.is_dir() or not any(p.is_file() for p in models.rglob('*')):
        missing.append('models/ (non-empty)')
    if missing:
        raise RuntimeError(f'{model_path}: not a finished AutoGluon save (missing {missing}): an incomplete fit; move '
                           'the folder away by hand and re-run this target')
    newest = max(p.stat().st_mtime for p in model_path.rglob('*') if p.is_file())
    if time.time() - newest < RECOVERY_QUIET_SECONDS:
        raise RuntimeError(f'{model_path} was modified less than {RECOVERY_QUIET_SECONDS // 60} minutes ago and may '
                           'still be training; not recovered')
    saved_version = (model_path / 'version.txt').read_text(encoding='utf-8').strip()
    installed = autogluon_version()
    if saved_version != installed:
        raise RuntimeError(f'{model_path}: saved with AutoGluon {saved_version}, installed {installed}; incompatible')
    from autogluon.tabular import TabularPredictor
    try:
        predictor = TabularPredictor.load(str(model_path))
    except Exception as exc:  # noqa: BLE001 - reported as an unusable model
        raise RuntimeError(f'{model_path}: cannot be loaded ({type(exc).__name__}: {exc}); corrupted or incompatible; '
                           'move it away by hand and re-run this target') from exc
    frame, rows = training_frame(series, target, stage)
    problems = []
    if getattr(predictor, 'label', LABEL) != LABEL:
        problems.append(f'label {predictor.label!r}')
    if getattr(predictor, 'problem_type', 'regression') != 'regression':
        problems.append(f'problem_type {predictor.problem_type!r}')
    metric = getattr(getattr(predictor, 'eval_metric', None), 'name', AUTOGLUON_SETTINGS['eval_metric'])
    if metric != AUTOGLUON_SETTINGS['eval_metric']:
        problems.append(f'eval_metric {metric!r}')
    fitted_rows = getattr(getattr(predictor, '_learner', None), '_pre_X_rows', None)
    fitted_rows = int(fitted_rows) if isinstance(fitted_rows, (int, np.integer)) else None
    if fitted_rows is not None and fitted_rows != len(rows):
        problems.append(f'fitted on {fitted_rows} rows, this stage has {len(rows)}')
    schema, best = None, None
    try:
        schema = feature_schema(predictor, frame)
        best = _model_best(predictor)
        board = directory / f'{target}.leaderboard.csv'
        if board.is_file() and best not in set(pd.read_csv(board)['model'].astype(str)):
            problems.append(f'best model {best!r} is not in the saved leaderboard {board.name}')
        sample = frame[list(FEATURES)].iloc[np.linspace(0, len(frame) - 1, PREDICTION_CHECK_ROWS).astype(int)]
        pred = np.asarray(predictor.predict(sample), dtype=float).reshape(-1)
        if pred.shape != (PREDICTION_CHECK_ROWS,) or not np.isfinite(pred).all():
            problems.append(f'prediction check returned {pred.shape} values or nonfinite values')
    except (ValueError, KeyError) as exc:
        problems.append(f'{type(exc).__name__}: {exc}')
    if problems:
        raise RuntimeError(f'{model_path}: not recovered (incompatible): {"; ".join(problems)}')
    return dict(predictor=predictor, frame=frame, rows=rows, schema=schema, fitted_rows=fitted_rows, best=best,
                saved_version=saved_version)


def recover_target(series, target, stage, directory, time_limit):
    """Complete the record of a predictor that was fitted and saved but has no training record (e.g. the run stopped
    at the old feature-schema check): validate_saved, then the leaderboard (if absent), the idempotent cleanup and the
    record. The predictor is never refitted; anything that fails validation is refused and nothing is deleted."""
    v = validate_saved(series, target, stage, directory)
    predictor = v['predictor']
    board = directory / f'{target}.leaderboard.csv'
    if not board.exists():
        _write_leaderboard(predictor, board)
    record = _write_record(series, target, stage, directory, v['frame'], v['rows'], predictor, v['schema'], time_limit,
                           fit_seconds=None, cleanup=_cleanup(predictor), recovered=True,
                           recovery_note='predictor fitted and saved earlier without a record; loaded and validated '
                                         'read-only (validate_saved), then completed (cleanup is idempotent)',
                           fitted_rows_check='verified' if v['fitted_rows'] is not None else 'not exposed by AutoGluon',
                           prediction_check_rows=PREDICTION_CHECK_ROWS)
    print('FMPC_TRAIN_TARGET_RECOVERED', series.feeder, stage, target, f'rows={len(v["rows"])}',
          f'dropped={sorted(v["schema"]["dropped_features"])}', flush=True)
    return record


def verify_saved(series, names, stage, directory):
    """--verify-saved: run validate_saved on every selected target that has a saved predictor but no record. Writes
    nothing. Returns True if every such predictor passes."""
    ok, found = True, 0
    for t in names:
        if (directory / f'{t}.json').exists() or not (directory / t).exists():
            continue
        found += 1
        try:
            v = validate_saved(series, t, stage, directory)
        except RuntimeError as exc:
            ok = False
            print('FMPC_SAVED_REFUSED', series.feeder, stage, t, str(exc), flush=True)
            continue
        print('FMPC_SAVED_OK', series.feeder, stage, t, f'rows={len(v["rows"])}', f'fitted_rows={v["fitted_rows"]}',
              f'best={v["best"]}', f'autogluon={v["saved_version"]}', f'retained={len(v["schema"]["retained_features"])}',
              f'dropped={v["schema"]["dropped_features"]}', flush=True)
    print('FMPC_VERIFY_SAVED_DONE', series.feeder, stage, f'saved_without_record={found}', 'all_ok' if ok else 'REFUSALS',
          flush=True)
    return ok


def train_target(series, target, stage, directory, time_limit, skip_completed=False):
    record_path, model_path = directory / f'{target}.json', directory / target
    if record_path.exists():
        if skip_completed:
            print('FMPC_TRAIN_TARGET_SKIPPED (record exists)', target, flush=True)
            return None
        raise FileExistsError(f'{record_path} exists; models are never overwritten (use --skip-completed to skip)')
    if model_path.exists():
        if skip_completed:
            return recover_target(series, target, stage, directory, time_limit)
        raise FileExistsError(f'{model_path} exists without a training record; --skip-completed validates and '
                              'completes a saved predictor, an incomplete one must be moved away by hand')
    from autogluon.tabular import TabularPredictor          # only needed here; the rest of the package runs without it
    frame, rows = training_frame(series, target, stage)
    started = time.perf_counter()
    predictor = TabularPredictor(label=LABEL, eval_metric=AUTOGLUON_SETTINGS['eval_metric'], path=str(model_path),
                                 verbosity=2)
    predictor.fit(train_data=frame, time_limit=AUTOGLUON_SETTINGS['time_limit_s'] if time_limit is None else time_limit,
                  presets=AUTOGLUON_SETTINGS['presets'])
    fit_seconds = time.perf_counter() - started
    _write_leaderboard(predictor, directory / f'{target}.leaderboard.csv')
    cleanup = _cleanup(predictor)
    schema = feature_schema(predictor, frame)
    record = _write_record(series, target, stage, directory, frame, rows, predictor, schema, time_limit,
                           fit_seconds=fit_seconds, cleanup=cleanup, recovered=False)
    print('FMPC_TRAIN_TARGET_DONE', series.feeder, stage, target, f'rows={len(rows)}', f'best={record["model_best"]}',
          f'dropped={sorted(schema["dropped_features"])}', f'{fit_seconds:.0f}s', flush=True)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--targets', nargs='+', help='explicit subset (default: all, or the --chunk of --chunks)')
    p.add_argument('--chunk', type=int, help='Slurm array index K (0-based)')
    p.add_argument('--chunks', type=int, help='number of array tasks N')
    p.add_argument('--time-limit', type=int, help=f'seconds per model (default {AUTOGLUON_SETTINGS["time_limit_s"]}; '
                                                  'a non-default value is recorded)')
    p.add_argument('--skip-completed', action='store_true', help='skip targets that already have a training record')
    p.add_argument('--status', action='store_true', help='report completeness of the stage and exit (writes nothing)')
    p.add_argument('--verify-saved', action='store_true',
                   help='validate saved predictors that have no record, read-only, and exit (writes nothing, no fit)')
    p.add_argument('--output-root', help='default results/<protocol>')
    p.add_argument('--time-series', help='processed series (default: the environment configuration)')
    a = p.parse_args(argv)
    root = maintained_output_root(MAINTAINED_PROTOCOL, a.output_root)
    names = select_targets(a.feeder, a.targets, a.chunk, a.chunks)
    directory = models_dir(root, a.feeder, a.stage)
    if a.status:
        ok = print_status(stage_status(directory, a.feeder, names, kind='model'), 'FMPC_TRAIN_STATUS')
        raise SystemExit(0 if ok else 1)
    series = load_series(a.feeder, a.time_series)
    if a.verify_saved:
        raise SystemExit(0 if verify_saved(series, names, a.stage, directory) else 1)
    directory.mkdir(parents=True, exist_ok=True)
    rows = training_rows(series, a.stage)
    print('FMPC_TRAIN_START', a.feeder, a.stage, f'targets={len(names)}', f'rows={len(rows)}',
          f'{series.timestamps[rows.min()]} .. {series.timestamps[rows.max()]}', f'cutoff={cutoff_time(series, a.stage)}',
          f'series_sha256={series.sha256[:16]}', flush=True)
    for t in names:
        train_target(series, t, a.stage, directory, a.time_limit, a.skip_completed)
    print('FMPC_TRAIN_DONE', a.feeder, a.stage, f'targets={len(names)}', directory, flush=True)
    return directory


if __name__ == '__main__':
    main()
