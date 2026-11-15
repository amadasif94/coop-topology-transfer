"""Isolated December F-MPC topology tasks. Run from the repository root."""
import argparse
import csv
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / 'results/soc_feasible_affine_v2/ieee69/forecast_mpc/load_x130'
ROOT = REPO / 'results/fmpc_ieee69_topology_tasks'
TPS = tuple(f'TP{i}' for i in range(1, 11))
TABLES = ('step_metrics', 'episode_summary', 'aggregate_summary', 'terminal_soc_summary',
          'fmpc_days', 'fmpc_solves')
TIMINGS = {'solve_time_s', 'max_solve_time_s', 'solve_time_sec'}
for component in ('policy', 'env_step', 'doe', 'backup_search', 'pf', 'diagnostic_pf', 'env_other'):
    for prefix in ('', 'mean_', 'mean_mean_'):
        TIMINGS.add(f'{prefix}time_{component}_ms')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def rows(folder, name):
    with (folder / (name + '.csv')).open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def validate_tables(folder, tp, starts):
    """Cover every scheduled day, including explicitly failed/incomplete episodes."""
    attempts = {}
    for r in rows(folder, 'episode_summary'):
        start = int(r['start_index'])
        n = int(r['attempted_evaluation_steps'])
        complete = int(r['completed_episodes'])
        failed = int(r['failed_episodes'])
        require(int(r['intended_episodes']) == 1 and complete in (0, 1) and failed in (0, 1)
                and int(r['incomplete_episodes']) == 1 - complete and 0 <= n <= 96
                and (not complete or (n == 96 and not failed)), f'inconsistent episode {start}')
        attempts[start] = n
    require(set(attempts) == set(starts), 'episode schedule mismatch')
    for name in TABLES:
        seen = set()
        for r in rows(folder, name):
            require(r['topology_case'] == tp, f'{name}: unexpected topology')
            key = ()
            if name != 'aggregate_summary':
                start = int(r['start_index'])
                key = (start,)
                if name in ('step_metrics', 'fmpc_solves'):
                    step = int(r['step'])
                    key += (step,)
                    if name == 'fmpc_solves':
                        require(int(r['horizon_steps']) == 96 - step, f'wrong horizon at {key}')
                        day = starts.index(start) + 1
                        expected = f'2018-12-{day:02} {step // 4:02}:{step % 4 * 15:02}'
                        require(r['date_time'].replace('T', ' ')[:16] == expected, f'wrong solve time at {key}')
                elif name == 'fmpc_days':
                    require(int(r['solves']) == attempts[start], f'inconsistent solve count {key}')
            require(key not in seen, f'{name}: duplicate {key}')
            seen.add(key)
        expected = {(s,) for s in starts}
        if name == 'aggregate_summary':
            expected = {()}
        elif name in ('step_metrics', 'fmpc_solves'):
            expected = {(s, i) for s in starts for i in range(attempts[s])}
            if name == 'step_metrics':
                expected.update((s, -1) for s in starts if not attempts[s])
        elif name == 'fmpc_days':
            expected = {(s,) for s in starts if attempts[s]}
        require(seen == expected, f'{name}: missing or unexpected records')


def equivalent(a, b):
    if a == b:
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str) and a.startswith('[') and b.startswith('['):
        return equivalent(json.loads(a), json.loads(b))
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8)
    except (ValueError, TypeError):
        return False


def compare_day(folder, reference, start, source_exceptions=()):
    # A 31-day aggregate has no one-day counterpart.
    episode, = rows(folder, 'episode_summary')
    require(int(episode['completed_episodes']) == 1, 'smoke needs one completed 96-step day')
    report = {}
    for name in TABLES:
        if name == 'aggregate_summary':
            continue
        measured = {r.get('step', ''): r for r in rows(folder, name)}
        count = len(measured)
        digest = hashlib.sha256()
        for r in rows(reference, name):
            if r['topology_case'] != 'TP1' or int(r['start_index']) != start:
                continue
            other = measured.pop(r.get('step', ''), None)
            require(other is not None, f'{name}: unexpected reference row')
            digest.update(json.dumps(r, sort_keys=True).encode())
            for col in (r.keys() | other.keys()) - TIMINGS:
                require(equivalent(r.get(col, ''), other.get(col, '')),
                        f'{name} step={r.get("step")} column={col}: {r.get(col)!r} vs {other.get(col)!r}')
        require(count and not measured, f'{name}: missing reference rows')
        report[name] = dict(matched_rows=count, reference_rows_sha256=digest.hexdigest())
    return dict(reference=str(reference), start_index=start, tables=report, source_exceptions=list(source_exceptions),
                absolute_tolerance=1e-8, relative_tolerance=1e-9, excluded_timings=sorted(TIMINGS))


def reference():
    manifest = read(REFERENCE / 'run_manifest.json')
    record = manifest['forecast_mpc_record']
    require(manifest['status'] == 'finished' and manifest['seed'] == 42 and
            record['protocol'] == 'soc_feasible' and record['split'] == 'test' and
            record['load_scaling_factor'] == 1.3 and record['days'] == 'all' and
            [e['topology'] for e in manifest['environments']].count('TP1') == 1 and
            sorted(e['topology'] for e in manifest['environments']) == sorted(TPS),
            'expected completed IEEE69 1.3x all-topology December reference')
    return manifest


def forecast_identity(record):
    directory = Path(record['forecast_stages']['dec']['directory'])
    # ForecastStore verifies the actual NPZ bytes against these records whenever it loads.
    return {f'{kind}_node_{i}': sha(directory / f'{kind}_node_{i}.json')
            for kind in ('load', 'pv') for i in range(1, 70)}


def source_exceptions(reference, current):
    """Reviewed generator-only change on the saved-forecast evaluation path."""
    generator = 'evaluation/forecast_mpc/generate.py'
    old = 'ffd4af7cfe0df51969b04b2e2ee8344101d8a797cd1e3dfc8401cda352c1e61d'
    new = '0003bcd06e99876bd6d55560c504f804f41473d5673b706c52add8ae9109f799'
    require(reference.keys() == current.keys(),
            f'evaluation source inventory paths differ: {sorted(reference.keys() ^ current.keys())}')
    changed = sorted(path for path in reference if current[path] != reference[path])
    if not changed:
        return []
    require(changed == [generator] and (reference[generator], current[generator]) == (old, new),
            f'unapproved evaluation source difference: {changed}')
    return [dict(path=generator, reference_sha256=old, current_sha256=new,
                 reason='Optional issue_steps in offline generator; evaluation reads verified saved forecasts.')]


def check_manifest(manifest, ref, tp, pct, smoke):
    env, = manifest['environments']
    base, = [e for e in ref['environments'] if e['topology'] == tp]
    r, original = manifest['forecast_mpc_record'], ref['forecast_mpc_record']
    require(manifest['status'] == 'finished' and env['feeder'] == 'IEEE69' and env['topology'] == tp
            and manifest['seed'] == ref['seed'] and r['load_scaling_factor'] == pct / 100
            and env['intended_episodes'] == (1 if smoke else 31), 'incorrect shard identity')
    source_exceptions(original['source_inventory'], r['source_inventory'])
    for field in ('settings', 'protocol', 'solver', 'experiment', 'split',
                  'allow_non_optimal', 'price_availability', 'pv_scaling_factor', 'dispatch', 'horizon'):
        require(r[field] == original[field], f'reference mismatch: {field}')
    require(r['forecast_stores'][tp] == original['forecast_stores'][tp], 'forecast provenance mismatch')
    require(env['data_scaling'] == dict(base['data_scaling'], load_scaling_factor=pct / 100),
            'environment scaling differs beyond authorized load multiplier')
    for field in ('battery_constraints', 'active_current_ratings_a', 'available_episode_start_indices',
                  'expected_steps_per_episode', 'interface_identity', 'doe', 'backup', 'reward', 'source_files',
                  'backup_mode', 'voltage_tolerance_pu', 'current_relative_tolerance'):
        require(env[field] == base[field], f'{tp}: reference mismatch: {field}')
    for entry in (base, env):
        require(len(entry['available_episode_start_indices']) == 31 and entry['expected_steps_per_episode'] == 96,
                'incorrect reference schedule')
    def inputs(entry):
        return {k: (v['exists'], v.get('sha256')) for k, v in entry['input_files'].items()}
    require(inputs(env) == inputs(base), 'topology or input data fingerprint mismatch')
    return env['available_episode_start_indices'][:env['intended_episodes']]


@contextmanager
def locked(parent, name):
    # Production runs on Linux. The kernel releases this lock even on Slurm timeout.
    import fcntl
    require(parent.resolve().parent == ROOT.resolve() and not parent.is_symlink(), 'unexpected output parent')
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ('.' + name + '.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checked(parent, name, identity):
    result = parent / name
    require(not result.is_symlink(), f'unexpected output link {result}')
    if not result.exists():
        return None
    receipt = read(result / 'completion.json')
    require(receipt['identity'] == identity, f'stale publication: {result}')
    data = result / 'tables'
    require(inventory(data) == receipt['files'], f'published files changed: {result}')
    return data


def publish(parent, name, attempt, data, identity):
    # Preserve absolute snapshot paths in the original manifest. Hard links use no second
    # copy of the CSV payload. Rename the complete directory on the same filesystem.
    final = parent / name
    require(not final.exists() and not final.is_symlink(), f'refusing to replace {final}')
    staged = attempt / 'publication'
    staged.mkdir()
    files = inventory(data)
    for relative in files:
        target = staged / 'tables' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(data / relative, target)
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EXDEV):
                raise
            shutil.copyfile(data / relative, target)
            require(sha(target) == files[relative], f'copy verification failed: {relative}')
    write(staged / 'completion.json', dict(identity=identity, files=files))
    staged.rename(final)


def identity(pct, tp, smoke, forecasts):
    return dict(schema=1, scale_percent=pct, topology=tp, smoke=smoke,
                reference_sha256=sha(REFERENCE / 'run_manifest.json'), forecasts=forecasts,
                workflow_sha256=sha(Path(__file__)))


def task(pct, tp, smoke, ref, forecasts):
    from . import run
    original = ref['forecast_mpc_record']
    current = run.code_inventory()
    exceptions = source_exceptions(original['source_inventory'], current)
    parent = ROOT / ('smoke' if smoke else f'load_x{pct}')
    ident = identity(pct, tp, smoke, forecasts)
    with locked(parent, tp):
        existing = checked(parent, tp, ident)
        if existing:
            if smoke:
                starts = check_manifest(read(existing / 'run_manifest.json'), ref, tp, pct, smoke)
                report = compare_day(existing, REFERENCE, starts[0], exceptions)
                require(report == read(existing / 'reference_comparison.json'), 'reference comparison changed')
            print('Verified existing publication', parent / tp)
            return
        attempt = Path(tempfile.mkdtemp(prefix='.' + tp + '-', dir=parent))
        settings = original['settings']
        args = ['--feeder', 'ieee69', '--split', 'test', '--topologies', tp, '--scales', str(pct / 100),
                '--protocol', original['protocol'], '--output-root', str(attempt),
                '--forecast-dirs', 'dec=' + original['forecast_stages']['dec']['directory'],
                '--price-availability', original['price_availability']]
        for flag, key in (('--time-limit', 'time_limit_s'), ('--mip-gap', 'mip_rel_gap'),
                          ('--branch-margin', 'branch_margin'), ('--voltage-margin', 'voltage_margin_pu')):
            args.extend([flag, str(settings[key])])
        if original['allow_non_optimal']:
            args.append('--allow-non-optimal')
        if smoke:
            args.extend(['--max-days', '1'])
        data, = run.main(args)
        manifest = read(data / 'run_manifest.json')
        require(manifest['forecast_mpc_record']['source_inventory'] == current, 'sources changed after preflight')
        starts = check_manifest(manifest, ref, tp, pct, smoke)
        validate_tables(data, tp, starts)
        if smoke:
            write(data / 'reference_comparison.json', compare_day(data, REFERENCE, starts[0], exceptions))
        publish(parent, tp, attempt, data, ident)
        print('Published', parent / tp)


def aggregate(pct, ref, forecasts):
    parent = ROOT / f'load_x{pct}'
    sources = []
    shard_hashes = {}
    for tp in TPS:
        data = checked(parent, tp, identity(pct, tp, False, forecasts))
        require(data is not None, f'missing completed {tp}')
        starts = check_manifest(read(data / 'run_manifest.json'), ref, tp, pct, False)
        validate_tables(data, tp, starts)
        sources.append(data)
        shard_hashes[tp] = sha(parent / tp / 'completion.json')
    ident = dict(shards=shard_hashes, scale_percent=pct)
    with locked(parent, 'aggregate'):
        if checked(parent, 'aggregate', ident):
            print('Verified existing aggregate', parent / 'aggregate')
            return
        attempt = Path(tempfile.mkdtemp(prefix='.aggregate-', dir=parent))
        out = attempt / 'tables'
        out.mkdir()
        for name in TABLES:
            columns = {}
            for src in sources:
                columns.update(dict.fromkeys(next(rows(src, name)).keys()))
            with (out / (name + '.csv')).open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(columns))
                writer.writeheader()
                for src in sources:
                    writer.writerows(rows(src, name))
        write(out / 'aggregation_manifest.json', ident)
        publish(parent, 'aggregate', attempt, out, ident)
        print('Published', parent / 'aggregate' / 'tables')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    stages.add_parser('smoke')
    for stage in ('task', 'aggregate'):
        sub = stages.add_parser(stage)
        sub.add_argument('percent', type=int, choices=(100, 110, 120, 130))
        if stage == 'task':
            sub.add_argument('topology', choices=TPS)
    args = parser.parse_args()
    os.chdir(REPO)
    require(not ROOT.resolve().is_relative_to((REPO / 'results/soc_feasible_affine_v2').resolve()),
            'isolated root resolves inside protected results')
    ref = reference()
    forecasts = forecast_identity(ref['forecast_mpc_record'])
    if args.stage == 'smoke':
        task(130, 'TP1', True, ref, forecasts)
        return
    require(checked(ROOT / 'smoke', 'TP1', identity(130, 'TP1', True, forecasts)),
            'run the 1.3x smoke/reference comparison first')
    if args.stage == 'task':
        task(args.percent, args.topology, False, ref, forecasts)
    else:
        aggregate(args.percent, ref, forecasts)


if __name__ == '__main__':
    main()
