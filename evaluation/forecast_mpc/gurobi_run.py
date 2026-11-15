"""Gurobi F-MPC: one December topology per task, IEEE33 and IEEE69, with persistent failure diagnostics. Maintained.

    python -u -m evaluation.forecast_mpc.gurobi_run task   --feeder ieee69 --scale 1.3 --topology TP3 [--max-days 1]
    python -u -m evaluation.forecast_mpc.gurobi_run status --feeder ieee69 --scale 1.3 [--max-days 1]

The controller, forecasts, environment, AC power flow, limiter, network backup, accounting and table writers are those
of evaluation.forecast_mpc.run (unchanged; HiGHS F-MPC and PF-MPC keep working). This runner only
  * routes evaluation.oracle.solver's milp() to evaluation.forecast_mpc.gurobi_backend (same arrays, Gurobi the ONLY
    backend of the run: MIPGap 1e-4, TimeLimit 60 s per LP/MILP call, feasible time-limited incumbents accepted,
    Seed 42, Threads = $SLURM_CPUS_PER_TASK or 1),
  * adds solver_backend / solver_native (native Gurobi status, objective, bound, gap, runtime per call) to
    fmpc_solves.csv and a 'solver_backend' block to run_manifest.json,
  * runs every task in its own attempt folder, validates provenance and tables, and only then publishes atomically.
Gurobi-only execution is PROVEN, not labelled: every SciPy/HiGHS entry point raises while the run is active (tripwire),
every decision checks that solver.milp is the Gurobi backend and that it executed >= 1 Gurobi call, and before
publication every solve row's invocation records (session id, sequential call ids, parameters, LP/MILP sequence) are
reconciled with the backend's own call counter (verify_backend_records); any gap fails the task.

Layout (never overwritten; default root results/fmpc_gurobi, refused inside results/soc_feasible* or the HiGHS tasks root):
  <root>/<feeder>/load_x<pct>[_smoke<N>]/.attempts/<attempt id>/attempt.json      status, stage, Slurm ids, heartbeat
                                                               failure_traceback.txt  (on failure / interruption)
                                                               output/...             run_manifest.json + CSV tables
  <root>/<feeder>/load_x<pct>[_smoke<N>]/<TP>/completion.json + tables/            published (hard links)
  <root>/<feeder>/load_x<pct>[_smoke<N>]/group_identity.json                        shared identity of all topologies
Attempt status: started -> completed | verified_existing | failed | interrupted. 'started' with no end is reported by
`status` as interrupted_or_unknown (job killed, node lost, or still running). Nothing is resumed automatically.
Provenance is validated against the explicit Gurobi configuration and freshly computed input, topology, forecast,
battery and schedule identities; no HiGHS reference manifest is read.
"""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import sys
import time
import importlib
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / 'results' / 'fmpc_gurobi'
# Results root of the original F-MPC experiments (= SLURM Script/forecast_mpc.slurm default OUTPUT_ROOT); read only.
FORECAST_RESULTS_ROOT = REPO / 'results' / 'soc_feasible_affine_v2'
SCHEMA = 'graph-marl-dess-fmpc-gurobi/1'
ROOT_MARKER = '.fmpc_gurobi_root.json'
ROOT_LOCK = '.fmpc_gurobi_root.lock'
GROUP_LOCK = '.group_identity.lock'
# Every SciPy entry point to HiGHS optimisation; each raises (and is counted) while a Gurobi F-MPC run is active.
HIGHS_ENTRY_POINTS = (('scipy.optimize', 'milp'), ('scipy.optimize', 'linprog'), ('scipy.optimize._milp', 'milp'),
                      ('scipy.optimize._linprog', 'linprog'))
HIGHS_TASK_ROOT = 'fmpc_ieee69_topology_tasks'
# The F-MPC Gurobi configuration (the identity every task is validated against).
GUROBI_FMPC_CONFIG = dict(backend='gurobi', mip_rel_gap=1e-4, time_limit_s=60.0, accept_feasible_incumbent=True,
                          seed=42)
INCUMBENT_POLICY = ('a feasible incumbent at the time limit is accepted and labelled time_limit_feasible / '
                    'proven_optimal=False; no incumbent, infeasible, unbounded or numeric trouble -> 0 kW fallback step')
HEARTBEAT_S = 60.0
TEST_DAYS = 31
SOC_LIMITS = dict(soc_min=0.1, soc_max=0.9, soc_init=0.5)


class TerminationRequested(BaseException):
    """SIGTERM (Slurm scancel / time limit). BaseException so no `except Exception` in the run can swallow it."""


class BackendVerificationError(Exception):
    """Gurobi-only execution could not be proven from the backend's own invocation records. Never a fallback: the
    task fails and nothing is published. (Not a RuntimeError, so the controller's 0 kW fallback cannot catch it.)"""


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fsync_dir(path):
    if os.name != 'posix':
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, data):
    """Serialise completely, write a UNIQUE temporary sibling (same directory = same filesystem), fsync it, then
    os.replace() it onto `path`: a reader sees the previous complete file or the new complete file, never a partial
    one. On any error the temporary file is removed and `path` is left untouched."""
    path = Path(path)
    text = json.dumps(data, indent=2, sort_keys=True, default=str)          # fails before anything is written
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp')
    try:
        with temp.open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    fsync_dir(path.parent)


@contextmanager
def shared_lock(path):
    """Blocking exclusive lock on `path` (created if needed) for short read-check-write sections on files shared by
    concurrent tasks (root marker, group identity). Linux: flock (released by the kernel if the process dies);
    Windows (local tests): msvcrt byte-range lock."""
    with Path(path).open('a+b') as stream:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
            return
        import msvcrt
        stream.seek(0)
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:                              # LK_LOCK gives up after ~10 s; keep waiting
                continue
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def durable_text(path, text, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def slurm_ids():
    keys = dict(job_id='SLURM_JOB_ID', array_job_id='SLURM_ARRAY_JOB_ID', array_task_id='SLURM_ARRAY_TASK_ID',
                cpus_per_task='SLURM_CPUS_PER_TASK', node='SLURMD_NODENAME', submit_dir='SLURM_SUBMIT_DIR')
    return {k: os.environ.get(v) for k, v in keys.items()}


def stream_path(fd):
    try:
        return os.readlink(f'/proc/self/fd/{fd}')
    except OSError:
        return None


def log_paths(log_out, log_err, attempts_dir):
    """Absolute stdout/stderr files of this process (CLI/launcher value, else /proc/self/fd), outside the attempts."""
    out = dict(stdout=log_out or os.environ.get('FMPC_LOG_OUT') or stream_path(1),
               stderr=log_err or os.environ.get('FMPC_LOG_ERR') or stream_path(2))
    for kind, value in out.items():
        if value and not value.startswith(('pipe:', 'socket:', '/dev/')):
            p = Path(value)
            require(p.is_absolute(), f'{kind} log path must be absolute: {value}')
            require(not p.resolve().is_relative_to(Path(attempts_dir).resolve()),
                    f'{kind} log {value} must not be inside the attempt directories')
    return out


def code_files():
    """Files whose content defines a Gurobi task (F-MPC sources + this runner, its backend and table validator)."""
    from evaluation.forecast_mpc.common import code_inventory
    inventory = code_inventory()
    for p in ('evaluation/forecast_mpc/gurobi_backend.py', 'evaluation/forecast_mpc/gurobi_run.py',
              'evaluation/forecast_mpc/ieee69_array.py'):
        inventory[p] = sha(REPO / p)
    return inventory


# ----------------------------------------------------------------------------------------------- output root / lock
def check_root(root):
    """Refuse authoritative/HiGHS result trees and any non-empty folder that is not a Gurobi F-MPC root."""
    root = Path(root).resolve()
    results = (REPO / 'results').resolve()
    protected = [results / HIGHS_TASK_ROOT] + sorted(results.glob('soc_feasible*'))
    for part in root.parts:
        require(not part.lower().startswith('soc_feasible') and part != HIGHS_TASK_ROOT,
                f'output root {root} is inside a protected (HiGHS / authoritative) result tree')
    for p in protected:
        require(not p.resolve().is_relative_to(root), f'output root {root} contains the protected tree {p}')
    marker = root / ROOT_MARKER

    def ours(p):                     # the marker, its lock and a concurrent writer's temporary file
        return p.name in (ROOT_MARKER, ROOT_LOCK) or (p.name.startswith(f'.{ROOT_MARKER}.') and p.name.endswith('.tmp'))

    def foreign():
        return root.exists() and not marker.is_file() and any(not ours(p) for p in root.iterdir())

    def refuse():
        raise ValueError(f'refusing to reuse {root}: not empty and not a Gurobi F-MPC root (no {ROOT_MARKER}); '
                         'prior HiGHS output directories are never reused')
    if foreign():                                         # checked before anything is written into the folder
        refuse()
    root.mkdir(parents=True, exist_ok=True)
    with shared_lock(root / ROOT_LOCK):                   # concurrent array tasks: one creates, the others read
        if foreign():
            refuse()
        if not marker.is_file():
            atomic_json(marker, dict(schema=SCHEMA, created_at_utc=utc()))
        require(read_json(marker).get('schema') == SCHEMA, f'{marker}: unexpected schema')
    return root


@contextmanager
def locked(parent, name):
    """Exclusive non-blocking lock per (task folder, topology); released by the kernel when the process dies."""
    try:
        import fcntl
    except ImportError:                                  # Windows (local tests only): no inter-process lock
        fcntl = None
    if fcntl is None:
        yield
        return
    with (Path(parent) / f'.{name}.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'another attempt holds the lock of {parent}/{name}') from exc
        yield


# ----------------------------------------------------------------------------------------------- attempt record
class Attempt:
    def __init__(self, directory, data):
        self.directory, self.path = Path(directory), Path(directory) / 'attempt.json'
        self.data = data
        self._last_beat = 0.0
        self.write()

    def write(self):
        atomic_json(self.path, self.data)

    def update(self, **fields):
        self.data.update(fields)
        self.write()

    def stage(self, name):
        self.update(stage=name, last_progress_utc=utc())
        print('FMPC_GUROBI_STAGE', self.data['attempt_id'], name, flush=True)

    def heartbeat(self, **progress):
        self.data['progress'] = dict(self.data.get('progress') or {}, **progress)
        now = time.monotonic()
        if now - self._last_beat >= HEARTBEAT_S:
            self._last_beat = now
            self.update(last_progress_utc=utc())


class Context:
    """Run-time state shared by the patched manifest/controller and the failure handler."""

    def __init__(self, attempt):
        self.attempt, self.manifest, self.backend, self.substage, self.solves = attempt, None, None, None, 0
        self.highs_calls, self.verification = 0, None


def report_failure(ctx, status, exc, tb, stage):
    """Persist the failure; any secondary (diagnostic) error is printed and never hides the original one."""
    print(f'FMPC_GUROBI_{status.upper()} stage={stage} {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
    print(tb, file=sys.stderr, flush=True)
    attempt = ctx.attempt
    failure = dict(status=status, stage=stage, substage=ctx.substage, exception_type=type(exc).__name__,
                   message=str(exc), timestamp_utc=utc(), traceback_file=None)
    try:
        path = attempt.directory / 'failure_traceback.txt'
        durable_text(path, f'{failure["timestamp_utc"]} stage={stage} substage={ctx.substage}\n{tb}')
        failure['traceback_file'] = str(path)
    except BaseException as secondary:  # noqa: BLE001
        print(f'FMPC_GUROBI_DIAGNOSTIC_WRITE_FAILED traceback: {type(secondary).__name__}: {secondary}',
              file=sys.stderr, flush=True)
    try:
        attempt.update(status=status, failure=failure, finished_at_utc=utc(), last_progress_utc=utc())
    except BaseException as secondary:  # noqa: BLE001
        print(f'FMPC_GUROBI_DIAGNOSTIC_WRITE_FAILED attempt.json: {type(secondary).__name__}: {secondary}',
              file=sys.stderr, flush=True)
    manifest = ctx.manifest
    if manifest is not None:
        try:
            manifest.data.update(status=status, failure=failure)
            manifest._write()
        except BaseException as secondary:  # noqa: BLE001
            print(f'FMPC_GUROBI_DIAGNOSTIC_WRITE_FAILED run_manifest.json: {type(secondary).__name__}: {secondary}',
                  file=sys.stderr, flush=True)
    return 143 if isinstance(exc, TerminationRequested) else 130 if isinstance(exc, KeyboardInterrupt) else 1


# ----------------------------------------------------------------------------------------------- expected identity
def file_identity(paths):
    from utility.provenance import file_record
    out = {}
    for key, value in paths.items():
        if isinstance(value, (str, Path)) and Path(value).suffix:
            record = file_record(value)
            out[key] = [bool(record['exists']), record.get('sha256')]
    return out


def forecast_identity(directory, feeder):
    """sha256 of every forecast record (each record pins its npz sha256, verified by ForecastStore on load)."""
    from evaluation.forecast_mpc.common import FORECAST_SCHEMA, targets
    directory = Path(directory)
    require(directory.is_dir(), f'forecast directory {directory} does not exist')
    out = {}
    for t in targets(feeder):
        record, npz = directory / f'{t}.json', directory / f'{t}.npz'
        missing = [p.name for p in (record, npz) if not p.is_file()]
        require(not missing, f'missing forecast {t} in {directory}: {missing}')
        content = read_json(record)
        require((content.get('schema'), content.get('feeder'), content.get('target'), content.get('stage'))
                == (FORECAST_SCHEMA, feeder, t, 'dec'), f'{record}: not the December {feeder} forecast of {t}')
        require(content.get('npz_sha256') == sha(npz), f'{npz}: does not match its record (sha256)')
        out[t] = sha(record)
    return out


def expected_identity(feeder, tp, scale, max_days, dec_dir, scratch):
    """Identity computed BEFORE the run from the configuration itself (no reference manifest)."""
    from configs.experiments import B4_BATTERIES, MAINTAINED_PROTOCOL, protocol_id, topology_config
    from evaluation.forecast_mpc.common import load_series, stage_day_starts
    from evaluation.reference_runs import split_protocol
    from utility.evaluation_protocol import protocol_config
    experiment = protocol_id(MAINTAINED_PROTOCOL)
    cfg = protocol_config(topology_config(feeder, tp, Path(scratch) / 'network_inputs', experiment=experiment),
                          split_protocol('test', [tp]))
    inputs = file_identity(cfg['paths'])
    series = load_series(feeder)
    starts = [int(s) for s in stage_day_starts(series, 'dec')]
    require(len(starts) == TEST_DAYS, f'expected {TEST_DAYS} December days, found {len(starts)}')
    require(inputs.get('processed_time_series', [None, None])[1] == series.sha256,
            'configured processed time series differs from the forecasting series')
    from evaluation.oracle.solver import OracleSettings
    settings = OracleSettings(time_limit_s=GUROBI_FMPC_CONFIG['time_limit_s'],
                              mip_rel_gap=GUROBI_FMPC_CONFIG['mip_rel_gap']).as_dict()
    return dict(schema=SCHEMA, feeder=feeder, topology=tp, scale=float(scale), max_days=max_days,
                experiment=experiment, input_files=inputs, input_files_sha256=json_sha(inputs),
                series_sha256=series.sha256, december_start_indices=starts,
                intended_start_indices=starts[:max_days] if max_days else starts,
                forecast_directory=str(Path(dec_dir).resolve()), forecasts=forecast_identity(dec_dir, feeder),
                battery=dict(B4_BATTERIES[feeder], **SOC_LIMITS), settings=settings)


def gurobi_identity(backend):
    return dict(GUROBI_FMPC_CONFIG, gurobi_version=backend.version, threads=backend.threads,
                incumbent_policy=INCUMBENT_POLICY, objective_sense='minimize')


# ----------------------------------------------------------------------------------------------- validation
SHARED_ENV_FIELDS = ('battery_constraints', 'available_episode_start_indices', 'expected_steps_per_episode', 'doe',
                     'backup_mode', 'backup', 'reward', 'voltage_tolerance_pu', 'current_relative_tolerance',
                     'data_scaling', 'interface_identity')


def validate_manifest(manifest, expected, gurobi, code):
    """Check run_manifest.json of the finished evaluation against the pre-run identity."""
    from evaluation.forecast_mpc.common import NUM_BUSES
    feeder, tp = expected['feeder'], expected['topology']
    require(manifest.get('status') == 'finished' and manifest.get('seed') == 42 and manifest.get('mode') == 'test',
            'run manifest not finished / wrong seed / not the test split')
    envs = manifest.get('environments') or []
    require(len(envs) == 1, f'expected one environment, found {len(envs)}')
    env = envs[0]
    require(env['feeder'] == feeder.upper() and str(env['topology']).upper() == tp, 'wrong feeder or topology')
    n_days = expected['max_days'] or TEST_DAYS
    require(env['intended_episodes'] == n_days, f'intended episodes {env["intended_episodes"]}, expected {n_days}')
    require(env['available_episode_start_indices'] == expected['december_start_indices'],
            'December start indices differ from the forecasting series')
    require(env['expected_steps_per_episode'] == 96, 'episode length is not 96')
    actual_inputs = {k: [bool(v['exists']), v.get('sha256')] for k, v in env['input_files'].items()}
    if actual_inputs != expected['input_files']:
        diff = sorted(k for k in actual_inputs.keys() | expected['input_files'].keys()
                      if actual_inputs.get(k) != expected['input_files'].get(k))
        raise ValueError(f'input data / topology fingerprint mismatch: {diff}')
    scaling = env.get('data_scaling') or {}
    require(float(scaling.get('load_scaling_factor', 1.0)) == expected['scale']
            and float(scaling.get('pv_scaling_factor', 1.0)) == 1.0
            and float(scaling.get('price_scaling_factor', 1.0)) == 1.0, f'unexpected data scaling {scaling}')
    require(not env['doe']['enabled'], 'DOE must be off')
    batteries = env['battery_constraints']
    require(batteries, 'no battery constraints recorded')
    for b in batteries:
        for key, value in expected['battery'].items():
            require(abs(float(b[key]) - float(value)) <= 1e-9, f'battery {key}={b[key]}, expected {value}')
    r = manifest['forecast_mpc_record']
    require(r['settings'] == expected['settings'], f'solver settings {r["settings"]} != {expected["settings"]}')
    require(r['allow_non_optimal'] is True and r['price_availability'] == 'dst_guard' and r['split'] == 'test'
            and float(r['load_scaling_factor']) == expected['scale'] and float(r['pv_scaling_factor']) == 1.0,
            'F-MPC record differs from the Gurobi F-MPC configuration')
    require(r.get('solver_backend') == gurobi, f'Gurobi configuration identity mismatch: {r.get("solver_backend")}')
    require(r['source_inventory'] == {k: v for k, v in code.items() if k in r['source_inventory']}
            and set(r['source_inventory']) <= set(code), 'source inventory changed during the run')
    store = r['forecast_stores'][tp]
    directories = store['directory'] if isinstance(store['directory'], list) else [store['directory']]
    require(store['stage'] == 'dec' and store['series_sha256'] == expected['series_sha256']
            and store['targets'] == 2 * NUM_BUSES[feeder] and len(directories) == 1
            and Path(directories[0]).resolve() == Path(expected['forecast_directory']),
            f'forecast store provenance mismatch: {store}')
    return env


def validate_outputs(data, expected, gurobi, code, ctx):
    from evaluation.forecast_mpc.ieee69_array import TABLES, rows, validate_tables
    missing = [n for n in ('run_manifest.json',) + tuple(f'{t}.csv' for t in TABLES) if not (data / n).is_file()]
    require(not missing, f'missing outputs: {missing}')
    env = validate_manifest(read_json(data / 'run_manifest.json'), expected, gurobi, code)
    validate_tables(data, expected['topology'], expected['intended_start_indices'])
    ctx.verification = verify_backend_records(list(rows(data, 'fmpc_solves')), ctx, gurobi)
    require(forecast_identity(expected['forecast_directory'], expected['feeder']) == expected['forecasts'],
            'forecast files changed during the run')
    require(code_files() == code, 'source files changed during the run')
    return env


def _flag(value):
    return {'True': True, 'False': False, '': None}.get(str(value), value)


def verify_backend_records(solve_rows, ctx, gurobi):
    """Prove from the backend's OWN invocation records (not from labels) that every LP/MILP call of this run was a
    Gurobi optimize() of this process's backend with the configured parameters, and that SciPy/HiGHS was never
    entered. Every F-MPC decision step solves (forecast errors are hard errors, never a no-solve decision), so a solve
    row without a Gurobi record is a failure. Raises BackendVerificationError; returns a summary for the receipt."""
    backend = ctx.backend

    def fail(message):
        raise BackendVerificationError(message)
    if ctx.highs_calls:
        fail(f'SciPy/HiGHS optimisation was entered {ctx.highs_calls} time(s) during the Gurobi run')
    if not solve_rows or len(solve_rows) != ctx.solves:
        fail(f'{len(solve_rows)} solve rows, {ctx.solves} controller decisions')
    ids, stages = [], dict(LP=0, MILP=0)
    for r in solve_rows:
        where = f'solve start_index={r.get("start_index")} step={r.get("step")}'
        try:
            calls = json.loads(r.get('solver_native') or '')
        except ValueError:
            fail(f'{where}: no readable Gurobi invocation record')
        if not isinstance(calls, list) or not calls:
            fail(f'{where}: no Gurobi invocation record (an optimisation without Gurobi)')
        for call in calls:
            identity = (call.get('backend'), call.get('session'), call.get('gurobi_version'), call.get('threads'),
                        call.get('seed'), call.get('time_limit_s'), call.get('mip_rel_gap'))
            wanted = ('gurobi', backend.session, gurobi['gurobi_version'], gurobi['threads'], gurobi['seed'],
                      gurobi['time_limit_s'], gurobi['mip_rel_gap'])
            if identity != wanted:
                fail(f'{where}: invocation record {identity} is not this run\'s Gurobi backend {wanted}')
            if call.get('stage') not in stages or not isinstance(call.get('call_id'), int):
                fail(f'{where}: malformed invocation record {call}')
            stages[call['stage']] += 1
            ids.append(call['call_id'])
        sequence = [c['stage'] for c in calls]
        ok, milp = _flag(r.get('solve_ok')), _flag(r.get('milp_used'))
        allowed = ([['LP', 'MILP']] if milp is True else [['LP']] if ok is True
                   else [['LP'], ['LP', 'MILP']])            # a failed solve stops after the LP or after the MILP
        if sequence not in allowed:
            fail(f'{where}: Gurobi calls {sequence} do not match the solve (solve_ok={ok}, milp_used={milp})')
        if ok is True and float(r.get('solver_status_code')) != float(calls[-1]['mapped_status']):
            fail(f'{where}: reported status {r.get("solver_status_code")} is not the Gurobi call\'s mapped status')
    if sorted(ids) != list(range(1, backend.calls_completed + 1)):
        fail(f'Gurobi call ids {len(ids)} recorded vs {backend.calls_completed} executed by the backend (missing, '
             'duplicated or foreign records)')
    return dict(verified='gurobi_invocation_records', session=backend.session, solves=len(solve_rows),
                gurobi_calls=len(ids), lp_calls=stages['LP'], milp_calls=stages['MILP'], highs_calls=ctx.highs_calls)


def check_group(parent, env, expected, gurobi, code):
    """Every topology of one (feeder, load factor[, smoke]) folder shares inputs, battery, safety and solver identity.
    The first topology creates group_identity.json (atomic temp + os.replace) and later ones compare with it; the
    read-check-write runs under an exclusive lock, so no task reads a partial file or overwrites another's."""
    shared = dict(schema=SCHEMA, feeder=expected['feeder'], scale=expected['scale'], max_days=expected['max_days'],
                  gurobi=gurobi, settings=expected['settings'], code_sha256=json_sha(code),
                  series_sha256=expected['series_sha256'], forecasts_sha256=json_sha(expected['forecasts']),
                  environment={k: env.get(k) for k in SHARED_ENV_FIELDS})
    shared = json.loads(json.dumps(shared, sort_keys=True, default=str))
    path = Path(parent) / 'group_identity.json'
    with shared_lock(Path(parent) / GROUP_LOCK):
        if not path.exists():
            atomic_json(path, shared)
            return shared
        existing = read_json(path)
    diff = sorted(k for k in existing.keys() | shared.keys()
                  if json.dumps(existing.get(k), sort_keys=True, default=str)
                  != json.dumps(shared.get(k), sort_keys=True, default=str))
    require(not diff, f'{path}: this topology differs from the other topologies in {diff}')
    return shared


# ----------------------------------------------------------------------------------------------- publication
def task_identity(expected, gurobi, code):
    return dict(expected=expected, gurobi=gurobi, code_sha256=json_sha(code))


def verify_publication(final, identity):
    from evaluation.forecast_mpc.ieee69_array import inventory
    require(not final.is_symlink(), f'unexpected link {final}')
    receipt = read_json(final / 'completion.json')
    require(receipt.get('identity_sha256') == json_sha(identity),
            f'{final} was published with another identity; refusing to overwrite it (use another --output-root)')
    require(inventory(final / 'tables') == receipt['files'], f'published files changed: {final}')
    return receipt


def publish(parent, tp, attempt, data, identity):
    from evaluation.forecast_mpc.ieee69_array import inventory
    final = Path(parent) / tp
    require(not final.exists() and not final.is_symlink(), f'refusing to replace {final}')
    staged = attempt.directory / 'publication'
    (staged / 'tables').mkdir(parents=True)
    files = inventory(data)
    for relative in files:
        target = staged / 'tables' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(data / relative, target)
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EXDEV, errno.EINVAL):
                raise
            shutil.copyfile(data / relative, target)
        require(sha(target) == files[relative], f'publication copy differs: {relative}')
    receipt = dict(schema=SCHEMA, identity=identity, identity_sha256=json_sha(identity), files=files,
                   attempt_id=attempt.data['attempt_id'], published_at_utc=utc(),
                   gurobi_verification=attempt.data.get('gurobi_verification'))
    durable_text(staged / 'completion.json', json.dumps(receipt, indent=2, sort_keys=True, default=str),
                 exclusive=True)
    require(not final.exists(), f'refusing to replace {final}')
    os.rename(staged, final)
    fsync_dir(parent)
    return final


# ----------------------------------------------------------------------------------------------- evaluation hooks
@contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def manifest_class(base, ctx, gurobi, attempt_ref):
    class GurobiRunManifest(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            ctx.manifest = self

        def _gurobi_fields(self):
            record = self.data.get('forecast_mpc_record')
            if record is not None:
                record.update(solver=f'gurobipy (Gurobi {gurobi["gurobi_version"]}) on the arrays of '
                                     'evaluation.oracle.solver.solve, via evaluation.forecast_mpc.gurobi_backend',
                              solver_backend=gurobi, solver_native_columns='fmpc_solves.csv: solver_backend, '
                                                                          'solver_native (per LP/MILP call)')
            self.data['gurobi_attempt'] = attempt_ref

        def add_environment(self, *args, **kwargs):
            self._gurobi_fields()
            return super().add_environment(*args, **kwargs)

        def finish(self):
            self._gurobi_fields()
            return super().finish()
    return GurobiRunManifest


def controller_class(base, ctx):
    class GurobiFMPCController(base):
        def select_action(self, obs, noise_std=0.0):
            from evaluation.oracle import solver
            ctx.substage = 'forecast window + optimisation (LP/MILP)'
            if solver.milp != ctx.backend.milp:                    # checked at EVERY decision, before solving
                raise BackendVerificationError('evaluation.oracle.solver.milp is not the Gurobi backend')
            ctx.backend.last_calls = []
            before, rows_before = ctx.backend.calls_completed, len(self.solves)
            request = super().select_action(obs, noise_std)
            calls = list(ctx.backend.last_calls)
            if len(self.solves) != rows_before + 1:
                raise BackendVerificationError('the controller did not log exactly one solve for this decision')
            if not calls or ctx.backend.calls_completed - before != len(calls):
                raise BackendVerificationError(
                    f'decision step {self.solves[-1].get("step")}: {len(calls)} Gurobi call(s) recorded, '
                    f'{ctx.backend.calls_completed - before} executed; the optimisation did not run on Gurobi')
            row = self.solves[-1]
            row['solver_backend'] = 'gurobi'
            row['gurobi_calls'] = len(calls)
            row['solver_native'] = json.dumps(calls, default=str)
            ctx.solves += 1
            ctx.substage = 'simulation (AC power flow, limiter, network backup)'
            ctx.attempt.heartbeat(topology=self.topology, start_index=row['start_index'], step=row['step'],
                                  solves=ctx.solves)
            return request
    return GurobiFMPCController


def evaluate(ctx, argv, gurobi, attempt_ref):
    from evaluation.forecast_mpc import run
    tools = run.feeder_tools

    def feeder_tools(feeder):
        a, b, c, manifest = tools(feeder)
        return a, b, c, manifest_class(manifest, ctx, gurobi, attempt_ref)

    def write_tables(*args, **kwargs):
        ctx.substage = 'output generation (tables)'
        return original_write(*args, **kwargs)
    original_write = run.write_tables
    with patched(run, 'feeder_tools', feeder_tools), patched(run, 'write_tables', write_tables), \
            patched(run, 'ForecastMPCController', controller_class(run.ForecastMPCController, ctx)), \
            highs_tripwire(ctx), ctx.backend.active():
        data, = run.main(argv)
    return Path(data)


@contextmanager
def highs_tripwire(ctx):
    """While the Gurobi run is active every SciPy/HiGHS optimisation entry point counts the call and raises
    BackendVerificationError, so no code path can silently optimise with HiGHS. evaluation.oracle.solver's own
    reference is separately rebound to Gurobi by GurobiBackend.active(); the tripwire covers every other reference."""
    def tripwire(name):
        def entered(*args, **kwargs):
            ctx.highs_calls += 1
            raise BackendVerificationError(f'{name} (SciPy/HiGHS) called during a Gurobi F-MPC run')
        return entered
    with ExitStack() as stack:
        for module_name, attr in HIGHS_ENTRY_POINTS:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, attr):
                stack.enter_context(patched(module, attr, tripwire(f'{module_name}.{attr}')))
        yield


# ----------------------------------------------------------------------------------------------- task / status
TOPOLOGIES = dict(ieee33=[f'TP{i}' for i in range(1, 8)], ieee69=[f'TP{i}' for i in range(1, 11)])  # = configs


def task_folder(root, feeder, scale, max_days):
    """Same label as experiments.load_stress.stress_env.scale_label (kept import-free for `status` on a login node)."""
    return Path(root) / feeder / (f'load_x{int(round(float(scale) * 100))}' + (f'_smoke{max_days}' if max_days else ''))


def default_forecast_dir(feeder):
    """December forecasts of the ORIGINAL F-MPC experiments (absolute; read only, verified by forecast_identity).
    They were generated and evaluated under SLURM Script/forecast_mpc.slurm's OUTPUT_ROOT, default
    results/soc_feasible_affine_v2, passed to run.py as --output-root. configs.experiments.maintained_output_root()
    without an explicit root returns results/soc_feasible (the earlier zero-preserving tree), so it is NOT used here."""
    from evaluation.forecast_mpc.common import forecasts_dir
    return Path(forecasts_dir(FORECAST_RESULTS_ROOT, feeder, 'dec'))


def run_task(args):
    """Returns the process exit code. Every catchable failure leaves attempt.json + failure_traceback.txt."""
    from evaluation.forecast_mpc.gurobi_backend import GurobiBackend, default_threads
    root = check_root(args.output_root)
    parent = task_folder(root, args.feeder, args.scale, args.max_days)
    attempts = parent / '.attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    ids = slurm_ids()
    attempt_id = '-'.join([args.topology, datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                           ids['job_id'] or 'local', ids['array_task_id'] or 'x', uuid.uuid4().hex[:8]])
    directory = attempts / attempt_id
    directory.mkdir()
    attempt = Attempt(directory, dict(
        schema=SCHEMA, attempt_id=attempt_id, status='started', stage='init', feeder=args.feeder,
        topology=args.topology, scale=args.scale, max_days=args.max_days, started_at_utc=utc(),
        last_progress_utc=utc(), slurm=ids, host=socket.gethostname(), pid=os.getpid(),
        python=platform.python_version(), argv=sys.argv, gurobi_config=GUROBI_FMPC_CONFIG,
        output_root=str(root)))
    ctx = Context(attempt)
    print('FMPC_GUROBI_ATTEMPT', directory, flush=True)
    try:
        attempt.update(logs=log_paths(args.log_out, args.log_err, attempts))
        with locked(parent, args.topology):
            attempt.stage('inputs_and_forecasts')
            dec_dir = Path(args.forecast_dir) if args.forecast_dir else default_forecast_dir(args.feeder)
            code = code_files()
            expected = expected_identity(args.feeder, args.topology, args.scale, args.max_days, dec_dir,
                                         directory / 'expected')
            attempt.stage('gurobi_environment')
            ctx.backend = GurobiBackend(default_threads(), GUROBI_FMPC_CONFIG['seed'])
            gurobi = gurobi_identity(ctx.backend)
            identity = task_identity(expected, gurobi, code)
            attempt.update(gurobi=gurobi, identity_sha256=json_sha(identity))
            final = parent / args.topology
            if final.exists() or final.is_symlink():
                receipt = verify_publication(final, identity)
                attempt.update(status='verified_existing', stage='completion', published=str(final),
                               published_by_attempt=receipt['attempt_id'], finished_at_utc=utc())
                print('FMPC_GUROBI_VERIFIED_EXISTING', final, flush=True)
                return 0
            attempt.stage('evaluation')
            argv = ['--feeder', args.feeder, '--split', 'test', '--topologies', args.topology,
                    '--scales', str(args.scale), '--output-root', str(directory / 'output'),
                    '--forecast-dirs', f'dec={dec_dir}', '--price-availability', 'dst_guard',
                    '--time-limit', repr(GUROBI_FMPC_CONFIG['time_limit_s']),
                    '--mip-gap', repr(GUROBI_FMPC_CONFIG['mip_rel_gap']), '--allow-non-optimal']
            if args.max_days:
                argv += ['--max-days', str(args.max_days)]
            data = evaluate(ctx, argv, gurobi, dict(attempt_id=attempt_id, attempt_json=str(attempt.path)))
            attempt.update(output=str(data))
            ctx.substage = None
            attempt.stage('validation')
            env = validate_outputs(data, expected, gurobi, code, ctx)
            attempt.update(gurobi_verification=ctx.verification)
            check_group(parent, env, expected, gurobi, code)
            attempt.stage('publication')
            final = publish(parent, args.topology, attempt, data, identity)
            attempt.stage('completion')
            attempt.update(status='completed', published=str(final), finished_at_utc=utc(), solves=ctx.solves)
            print('FMPC_GUROBI_PUBLISHED', final, flush=True)
            return 0
    except BaseException as exc:  # noqa: BLE001 - includes SystemExit from run.main and TerminationRequested
        tb = traceback.format_exc()
        status = 'interrupted' if isinstance(exc, (TerminationRequested, KeyboardInterrupt)) else 'failed'
        return report_failure(ctx, status, exc, tb, attempt.data.get('stage'))


def status_report(args):
    """Published / failed / interrupted / interrupted_or_unknown per topology. Exit 0 only if all are published.
    Standard library only (runs on a login node)."""
    from evaluation.forecast_mpc.ieee69_array import inventory
    parent = task_folder(Path(args.output_root).resolve(), args.feeder, args.scale, args.max_days)
    report, complete = {}, True
    for tp in args.topologies or TOPOLOGIES[args.feeder]:
        entry = dict(published=None, attempts=[])
        final = parent / tp
        if final.exists():
            try:
                receipt = read_json(final / 'completion.json')
                ok = inventory(final / 'tables') == receipt['files']
                entry['published'] = dict(ok=ok, attempt_id=receipt.get('attempt_id'),
                                          identity_sha256=receipt.get('identity_sha256'))
            except Exception as exc:  # noqa: BLE001
                entry['published'] = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
        for path in sorted((parent / '.attempts').glob(f'{tp}-*')):
            try:
                a = read_json(path / 'attempt.json')
                state = a['status']
                if state == 'started':
                    state = 'interrupted_or_unknown'          # incomplete: no completion, failure or interruption record
                entry['attempts'].append(dict(attempt_id=path.name, status=state, recorded_status=a['status'],
                                              stage=a.get('stage'), last_progress_utc=a.get('last_progress_utc'),
                                              progress=a.get('progress'), slurm=a.get('slurm'),
                                              failure=a.get('failure'), logs=a.get('logs'),
                                              traceback=(path / 'failure_traceback.txt').is_file()))
            except Exception as exc:  # noqa: BLE001
                entry['attempts'].append(dict(attempt_id=path.name, status='interrupted_or_unknown',
                                              error=f'unreadable attempt.json: {type(exc).__name__}: {exc}'))
        entry['state'] = ('complete' if entry['published'] and entry['published'].get('ok') else
                          'published_but_modified' if entry['published'] else
                          entry['attempts'][-1]['status'] if entry['attempts'] else 'not_started')
        complete &= entry['state'] == 'complete'
        report[tp] = entry
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for tp, e in report.items():
            last = e['attempts'][-1] if e['attempts'] else {}
            print(f'{tp:5s} {e["state"]:24s} attempts={len(e["attempts"])} last_stage={last.get("stage")} '
                  f'last_progress={last.get("last_progress_utc")} failure={(last.get("failure") or {}).get("message")}')
    return 0 if complete else 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('task', 'status'):
        s = sub.add_parser(name)
        s.add_argument('--feeder', required=True, choices=('ieee33', 'ieee69'))
        s.add_argument('--scale', required=True, type=float, choices=(1.0, 1.1, 1.2, 1.3))
        s.add_argument('--max-days', type=int, help='smoke task: first N December days (separate _smoke<N> folder)')
        s.add_argument('--output-root', default=str(DEFAULT_ROOT))
        if name == 'task':
            s.add_argument('--topology', required=True, type=str.upper)
            s.add_argument('--forecast-dir', help='December forecast folder (default: the maintained dec stage folder)')
            s.add_argument('--log-out', help='absolute stdout log path (recorded; default /proc/self/fd/1)')
            s.add_argument('--log-err', help='absolute stderr log path (recorded; default /proc/self/fd/2)')
        else:
            s.add_argument('--topologies', nargs='+', type=str.upper)
            s.add_argument('--json', action='store_true')
    args = p.parse_args(argv)
    if args.max_days is not None and not 1 <= args.max_days < TEST_DAYS:
        p.error(f'--max-days must be between 1 and {TEST_DAYS - 1} (omit it for all {TEST_DAYS} days)')
    if args.command == 'status':
        return status_report(args)
    from configs.experiments import TOPOLOGIES as CONFIGURED
    require(CONFIGURED == TOPOLOGIES, 'configs.experiments.TOPOLOGIES changed; update gurobi_run.TOPOLOGIES')
    if args.topology not in TOPOLOGIES[args.feeder]:
        p.error(f'--topology must be one of {TOPOLOGIES[args.feeder]}')
    previous = None
    if hasattr(signal, 'SIGTERM'):
        def on_term(signum, frame):
            raise TerminationRequested(f'signal {signum} (SIGTERM) received')
        try:
            previous = signal.signal(signal.SIGTERM, on_term)
        except ValueError:                                # not the main thread (tests): no handler
            previous = None
    try:
        return run_task(args)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
