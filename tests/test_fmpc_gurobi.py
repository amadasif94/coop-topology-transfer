"""Gurobi F-MPC backend and runner. gurobipy is replaced by tests.fake_gurobipy (HiGHS underneath), so no licence or
installation is needed. The optional class RealGurobi runs only if a real gurobipy with a working licence is present.

    python -m unittest tests.test_fmpc_gurobi -v
"""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from tests import fake_gurobipy as fake

TMP = None


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix='fmpc-gurobi-tests-'))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


@contextlib.contextmanager
def gurobipy_module(value):
    """Replace ONLY sys.modules['gurobipy'] (patch.dict(sys.modules) would also unload torch etc. on exit)."""
    missing = object()
    old = sys.modules.get('gurobipy', missing)
    sys.modules['gurobipy'] = value
    try:
        yield
    finally:
        if old is missing:
            sys.modules.pop('gurobipy', None)
        else:
            sys.modules['gurobipy'] = old


@contextlib.contextmanager
def fake_gurobi(**state):
    fake.reset()
    fake.STATE.__dict__.update(state)
    with gurobipy_module(fake.module()):
        yield fake.STATE


def small_problem():
    """rows: equality, >=, <=, two-sided, free; variables: [0,inf), (-inf,inf), [0,1] integer, bounded."""
    from scipy import sparse
    A = sparse.csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, -1, 0, 1], [1, 1, 1, 1]], float))
    lo = np.array([4.0, 1.0, -np.inf, -2.0, -np.inf])
    hi = np.array([4.0, np.inf, 6.0, 3.0, np.inf])
    c = np.array([1.0, 2.0, -1.0, 0.5])
    lb, ub = np.array([0.0, -np.inf, 0.0, 0.0]), np.array([np.inf, np.inf, 1.0, 2.0])
    return c, A, lo, hi, lb, ub


class Adapter(unittest.TestCase):
    def test_rows_are_split_exactly(self):
        from evaluation.forecast_mpc.gurobi_backend import split_rows
        _, A, lo, hi, _, _ = small_problem()
        blocks = {sense: (m.toarray(), rhs) for m, sense, rhs in split_rows(A, lo, hi)}
        self.assertEqual(blocks['='][0].tolist(), [[1, 1, 0, 0]])
        np.testing.assert_array_equal(blocks['>'][1], [1.0, -2.0])                  # rows 1 and 3 (two-sided)
        np.testing.assert_array_equal(blocks['<'][1], [6.0, 3.0])                   # rows 2 and 3; free row 4 dropped
        self.assertEqual(sum(len(b[1]) for b in blocks.values()), 5)

    def test_lp_and_milp_equal_highs_on_the_same_arrays(self):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend
        c, A, lo, hi, lb, ub = small_problem()
        with fake_gurobi() as state:
            backend = GurobiBackend(threads=3, seed=42)
            for integ in (np.zeros(4), np.array([0, 0, 1, 0])):
                ours = backend.milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi),
                                    options=dict(time_limit=60.0, mip_rel_gap=1e-4, disp=False))
                ref = milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi))
                self.assertEqual(ours.status, 0)
                self.assertAlmostEqual(ours.fun, ref.fun, places=9)
                np.testing.assert_allclose(ours.x, ref.x, atol=1e-9)
            self.assertEqual([call['stage'] for call in backend.last_calls], ['LP', 'MILP'])
            self.assertIn(('TimeLimit', 60.0), state.model_params)
            self.assertIn(('MIPGap', 1e-4), state.model_params)
            self.assertEqual(state.disposed, 2)

    def test_status_mapping_and_incumbent_policy(self):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend
        c, A, lo, hi, lb, ub = small_problem()
        args = dict(integrality=np.zeros(4), bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi),
                    options=dict(time_limit=60.0, mip_rel_gap=1e-4))
        cases = [(9, True, 1, True), (13, True, 1, True), (9, False, 1, False), (3, False, 2, False),
                 (4, False, 2, False), (5, False, 3, False), (12, False, 4, False), (11, True, 4, False)]
        for status, solution, code, keep in cases:
            with self.subTest(status=status, solution=solution), fake_gurobi(force_status=status,
                                                                              force_solution=solution):
                backend = GurobiBackend(1, 42)
                res = backend.milp(c, **args)
                self.assertEqual(res.status, code)
                self.assertEqual(res.x is not None, keep)
                self.assertEqual(backend.last_calls[-1]['status'], status)

    def test_api_errors_are_backend_errors_not_runtime_fallbacks(self):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend, GurobiBackendError
        self.assertFalse(issubclass(GurobiBackendError, RuntimeError))       # the controller catches RuntimeError only
        with gurobipy_module(None), self.assertRaisesRegex(GurobiBackendError, 'importable'):
            GurobiBackend(1, 42)
        with fake_gurobi(fail_start=True), self.assertRaisesRegex(GurobiBackendError, 'license'):
            GurobiBackend(1, 42)
        c, A, lo, hi, lb, ub = small_problem()
        with fake_gurobi(fail_optimize=True):
            backend = GurobiBackend(1, 42)
            with self.assertRaisesRegex(GurobiBackendError, 'Out of memory'):
                backend.milp(c, integrality=np.zeros(4), bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi))

    def test_threads_from_slurm_and_fixed_seed(self):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend, default_threads
        from evaluation.forecast_mpc.gurobi_run import GUROBI_FMPC_CONFIG
        self.assertEqual(GUROBI_FMPC_CONFIG['seed'], 42)
        with mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '6'}):
            self.assertEqual(default_threads(), 6)
            with fake_gurobi() as state:
                backend = GurobiBackend(default_threads(), GUROBI_FMPC_CONFIG['seed'])
                self.assertIn(('Threads', 6), state.env_params)
                self.assertIn(('Seed', 42), state.env_params)
                self.assertEqual((backend.threads, backend.seed, backend.version), (6, 42, '12.0.3'))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_threads(), 1)
        with mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '0'}), self.assertRaises(ValueError):
            default_threads()

    def test_solver_milp_is_restored(self):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend
        from evaluation.oracle import solver
        original = solver.milp
        with fake_gurobi():
            backend = GurobiBackend(1, 42)
            with self.assertRaises(KeyError), backend.active():
                self.assertEqual(solver.milp, backend.milp)
                raise KeyError('x')
        self.assertIs(solver.milp, original)


# --------------------------------------------------------------------------------------------- runner mechanisms
def expected(feeder='ieee69', tp='TP1', forecast='f' * 64):
    return dict(feeder=feeder, topology=tp, scale=1.3, max_days=1, forecasts={'load_node_1': forecast},
                forecast_directory='/x', settings={}, series_sha256='s', intended_start_indices=[0],
                december_start_indices=[0])


def fake_outputs(directory, tables=('step_metrics', 'episode_summary')):
    directory.mkdir(parents=True)
    (directory / 'run_manifest.json').write_text('{}')
    for t in tables:
        (directory / f'{t}.csv').write_text('topology_case\nTP1\n')
    return directory


class Runner(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=TMP)) / 'root'
        self.err = io.StringIO()

    def run_task(self, feeder='ieee69', tp='TP1', evaluate=None, validate=True, ident=None, extra=()):
        from evaluation.forecast_mpc import gurobi_run
        def default_evaluate(ctx, argv, gurobi, ref):
            ctx.solves = 5
            return fake_outputs(Path(argv[argv.index('--output-root') + 1]) / 'data')
        patches = [mock.patch.object(gurobi_run, 'expected_identity', return_value=ident or expected(feeder, tp)),
                   mock.patch.object(gurobi_run, 'code_files', return_value={'a.py': 'h'}),
                   mock.patch.object(gurobi_run, 'default_forecast_dir', return_value=Path('/x')),
                   mock.patch.object(gurobi_run, 'evaluate', side_effect=evaluate or default_evaluate)]
        if validate:
            patches.append(mock.patch.object(gurobi_run, 'validate_outputs',
                                             return_value={k: 1 for k in gurobi_run.SHARED_ENV_FIELDS}))
        with contextlib.ExitStack() as stack, fake_gurobi():
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stderr(self.err))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return gurobi_run.main(['task', '--feeder', feeder, '--scale', '1.3', '--topology', tp, '--max-days', '1',
                                    '--output-root', str(self.root), '--log-out', str(TMP / 'o.out'),
                                    '--log-err', str(TMP / 'o.err'), *extra])

    def parent(self, feeder='ieee69'):
        return self.root / feeder / 'load_x130_smoke1'

    def attempts(self, feeder='ieee69'):
        return sorted((self.parent(feeder) / '.attempts').iterdir())

    def test_injected_exception_traceback_failed_manifest_preserved_attempt(self):
        from utility.run_manifest import RunManifest
        for feeder in ('ieee33', 'ieee69'):                                    # same mechanism for both feeders
            with self.subTest(feeder=feeder):
                seen = {}

                def evaluate(ctx, argv, gurobi, ref):
                    seen['status'] = json.loads(ctx.attempt.path.read_text())['status']
                    ctx.manifest = RunManifest(ctx.attempt.directory / 'output', 'forecast_mpc', 42, 'test')
                    raise ValueError('boom in simulation')
                code = self.run_task(feeder, evaluate=evaluate)
                self.assertEqual((code, seen['status']), (1, 'started'))
                attempt, = self.attempts(feeder)
                record = json.loads((attempt / 'attempt.json').read_text())
                self.assertEqual((record['status'], record['stage']), ('failed', 'evaluation'))
                self.assertEqual(record['failure']['exception_type'], 'ValueError')
                self.assertIn('boom in simulation', record['failure']['message'])
                self.assertTrue(record['failure']['timestamp_utc'])
                tb = (attempt / 'failure_traceback.txt').read_text()
                self.assertIn('Traceback', tb)
                self.assertIn('boom in simulation', tb)
                manifest = json.loads((attempt / 'output' / 'run_manifest.json').read_text())
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(manifest['failure']['stage'], 'evaluation')
                self.assertIn('boom in simulation', self.err.getvalue())
                self.assertFalse((self.parent(feeder) / 'TP1').exists())
                self.assertEqual(record['logs'], dict(stdout=str(TMP / 'o.out'), stderr=str(TMP / 'o.err')))

    def test_secondary_diagnostic_failure_does_not_hide_the_error(self):
        from evaluation.forecast_mpc import gurobi_run
        real = gurobi_run.durable_text

        def broken(path, text, exclusive=False):
            if Path(path).name == 'failure_traceback.txt':
                raise OSError('disk quota exceeded (injected)')
            return real(path, text, exclusive)
        with mock.patch.object(gurobi_run, 'durable_text', broken):
            code = self.run_task(evaluate=mock.Mock(side_effect=ValueError('original error')))
        self.assertEqual(code, 1)
        self.assertIn('original error', self.err.getvalue())
        self.assertIn('DIAGNOSTIC_WRITE_FAILED', self.err.getvalue())
        record = json.loads((self.attempts()[0] / 'attempt.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertIn('original error', record['failure']['message'])

    def test_missing_csv_never_produces_a_completion_receipt(self):
        code = self.run_task(validate=False)                        # real validate_outputs; fake data lacks tables
        self.assertEqual(code, 1)
        record = json.loads((self.attempts()[0] / 'attempt.json').read_text())
        self.assertEqual(record['stage'], 'validation')
        self.assertIn('missing outputs', record['failure']['message'])
        self.assertEqual(list(self.root.rglob('completion.json')), [])

    def test_sigterm_is_recorded_as_interrupted(self):
        def evaluate(ctx, argv, gurobi, ref):
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)                           # what the kernel delivers on scancel / time limit
        code = self.run_task(evaluate=evaluate)
        self.assertEqual(code, 143)
        record = json.loads((self.attempts()[0] / 'attempt.json').read_text())
        self.assertEqual(record['status'], 'interrupted')
        self.assertTrue((self.attempts()[0] / 'failure_traceback.txt').is_file())

    def test_status_reports_incomplete_and_failed_attempts(self):
        from evaluation.forecast_mpc import gurobi_run
        self.run_task(evaluate=mock.Mock(side_effect=ValueError('x')))
        killed = self.parent() / '.attempts' / 'TP2-killed'
        killed.mkdir()
        (killed / 'attempt.json').write_text(json.dumps(dict(status='started', stage='evaluation',
                                                             last_progress_utc='2026-11-11T00:00:00')))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = gurobi_run.main(['status', '--feeder', 'ieee69', '--scale', '1.3', '--max-days', '1',
                                    '--output-root', str(self.root), '--json', '--topologies', 'TP1', 'TP2', 'TP3'])
        report = json.loads(out.getvalue())
        self.assertEqual(code, 3)
        self.assertEqual(report['TP1']['state'], 'failed')
        self.assertEqual(report['TP2']['state'], 'interrupted_or_unknown')
        self.assertEqual(report['TP3']['state'], 'not_started')

    def test_publication_verified_never_overwritten_and_needs_no_highs_reference(self):
        from evaluation.forecast_mpc import gurobi_run, ieee69_array
        with mock.patch.object(ieee69_array, 'REFERENCE', TMP / 'no_highs_reference_here'):
            self.assertEqual(self.run_task(), 0)
            final = self.parent() / 'TP1'
            receipt = json.loads((final / 'completion.json').read_text())
            files = ieee69_array.inventory(final)
            self.assertEqual(receipt['identity']['gurobi']['seed'], 42)
            self.assertEqual(self.run_task(), 0)                                   # same identity: verified only
            states = sorted(json.loads((a / 'attempt.json').read_text())['status'] for a in self.attempts())
            self.assertEqual(states, ['completed', 'verified_existing'])
            self.assertEqual(self.run_task(ident=expected(forecast='0' * 64)), 1)  # other forecasts: refused
            self.assertIn('another identity', self.err.getvalue())
            self.assertEqual(ieee69_array.inventory(final), files)
            with self.assertRaisesRegex(ValueError, 'refusing to replace'):
                gurobi_run.publish(self.parent(), 'TP1', mock.Mock(), final, {})
            status = io.StringIO()
            with contextlib.redirect_stdout(status):
                gurobi_run.main(['status', '--feeder', 'ieee69', '--scale', '1.3', '--max-days', '1',
                                 '--output-root', str(self.root), '--json', '--topologies', 'TP1'])
            self.assertEqual(json.loads(status.getvalue())['TP1']['state'], 'complete')

    def test_slurm_threads_and_seed_recorded(self):
        with mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '8', 'SLURM_JOB_ID': '777',
                                          'SLURM_ARRAY_TASK_ID': '3'}):
            self.assertEqual(self.run_task(tp='TP3'), 0)
        record = json.loads((self.attempts()[0] / 'attempt.json').read_text())
        self.assertEqual((record['gurobi']['threads'], record['gurobi']['seed']), (8, 42))
        self.assertEqual((record['gurobi']['mip_rel_gap'], record['gurobi']['time_limit_s']), (1e-4, 60.0))
        self.assertTrue(record['gurobi']['accept_feasible_incumbent'])
        self.assertEqual((record['slurm']['job_id'], record['slurm']['array_task_id']), ('777', '3'))
        self.assertTrue(record['attempt_id'].startswith('TP3-') and '-777-3-' in record['attempt_id'])

    def test_group_identity_rejects_a_differing_topology(self):
        from evaluation.forecast_mpc.gurobi_run import check_group
        parent = Path(tempfile.mkdtemp(dir=TMP))
        env = dict(battery_constraints=[dict(capacity_kwh=850.0)], backup_mode='directional')
        check_group(parent, env, expected(), dict(seed=42), {'a': 'h'})
        check_group(parent, env, expected(tp='TP2'), dict(seed=42), {'a': 'h'})
        with self.assertRaisesRegex(ValueError, 'environment'):
            check_group(parent, dict(env, battery_constraints=[dict(capacity_kwh=700.0)]), expected(tp='TP3'),
                        dict(seed=42), {'a': 'h'})
        with self.assertRaisesRegex(ValueError, 'gurobi'):
            check_group(parent, env, expected(tp='TP4'), dict(seed=7), {'a': 'h'})

    def test_output_root_refuses_protected_and_foreign_folders(self):
        from evaluation.forecast_mpc.gurobi_run import REPO, check_root
        for bad in (REPO / 'results' / 'soc_feasible_affine_v2' / 'x', REPO / 'results' / 'fmpc_ieee69_topology_tasks',
                    REPO / 'results'):
            with self.subTest(root=bad), self.assertRaises(ValueError):
                check_root(bad)
        foreign = Path(tempfile.mkdtemp(dir=TMP))
        (foreign / 'ieee69').mkdir()                                  # e.g. a prior HiGHS output root
        with self.assertRaisesRegex(ValueError, 'refusing to reuse'):
            check_root(foreign)


class ManifestValidation(unittest.TestCase):
    """validate_manifest against the explicit Gurobi configuration and the pre-run identity (no HiGHS reference)."""

    def setUp(self):
        from evaluation.forecast_mpc.gurobi_run import GUROBI_FMPC_CONFIG, SOC_LIMITS
        self.gurobi = dict(GUROBI_FMPC_CONFIG, gurobi_version='12.0.3', threads=4)
        self.expected = dict(feeder='ieee69', topology='TP2', scale=1.3, max_days=None,
                             december_start_indices=list(range(0, 31 * 96, 96)),
                             input_files=dict(line_data=[True, 'aa'], processed_time_series=[True, 'ss']),
                             battery=dict(capacity_kwh=850.0, max_charge_kw=170.0, max_discharge_kw=170.0, **SOC_LIMITS),
                             settings=dict(time_limit_s=60.0, mip_rel_gap=1e-4), series_sha256='ss',
                             forecast_directory=str(Path('/f').resolve()))
        battery = dict(self.expected['battery'], charge_efficiency=0.95)
        self.manifest = dict(status='finished', seed=42, mode='test', environments=[dict(
            feeder='IEEE69', topology='TP2', intended_episodes=31, expected_steps_per_episode=96,
            available_episode_start_indices=list(range(0, 31 * 96, 96)),
            input_files=dict(line_data=dict(exists=True, sha256='aa'), processed_time_series=dict(exists=True,
                                                                                                    sha256='ss')),
            data_scaling=dict(load_scaling_factor=1.3), doe=dict(enabled=False), battery_constraints=[battery] * 9)],
            forecast_mpc_record=dict(settings=dict(time_limit_s=60.0, mip_rel_gap=1e-4), allow_non_optimal=True,
                                     price_availability='dst_guard', split='test', load_scaling_factor=1.3,
                                     pv_scaling_factor=1.0, solver_backend=self.gurobi, source_inventory={'a.py': 'h'},
                                     forecast_stores=dict(TP2=dict(stage='dec', series_sha256='ss', targets=138,
                                                                   directory=['/f']))))      # merged-store form

    def check(self, mutate=None):
        import copy
        from evaluation.forecast_mpc.gurobi_run import validate_manifest
        manifest = copy.deepcopy(self.manifest)
        if mutate:
            mutate(manifest)
        return validate_manifest(manifest, self.expected, self.gurobi, {'a.py': 'h', 'b.py': 'g'})

    def test_good_manifest_passes(self):
        self.assertEqual(self.check()['topology'], 'TP2')

    def test_wrong_identities_are_rejected(self):
        env = lambda m: m['environments'][0]
        rec = lambda m: m['forecast_mpc_record']
        cases = dict(
            input_hash=lambda m: env(m)['input_files']['processed_time_series'].update(sha256='zz'),
            topology_hash=lambda m: env(m)['input_files']['line_data'].update(sha256='bb'),
            topology_name=lambda m: env(m).update(topology='TP3'),
            forecast_hash=lambda m: rec(m)['forecast_stores']['TP2'].update(series_sha256='other'),
            forecast_dir=lambda m: rec(m)['forecast_stores']['TP2'].update(directory='/elsewhere'),
            battery=lambda m: env(m)['battery_constraints'].__setitem__(0, dict(env(m)['battery_constraints'][0],
                                                                               capacity_kwh=700.0)),
            soc_limit=lambda m: env(m)['battery_constraints'].__setitem__(0, dict(env(m)['battery_constraints'][0],
                                                                                 soc_min=0.0)),
            start_indices=lambda m: env(m)['available_episode_start_indices'].__setitem__(0, 5),
            gurobi_seed=lambda m: rec(m).update(solver_backend=dict(self.gurobi, seed=7)),
            gurobi_gap=lambda m: rec(m).update(solver_backend=dict(self.gurobi, mip_rel_gap=1e-6)),
            gurobi_missing=lambda m: rec(m).pop('solver_backend'),
            highs_settings=lambda m: rec(m).update(settings=dict(time_limit_s=600.0, mip_rel_gap=1e-6)),
            strict_incumbents=lambda m: rec(m).update(allow_non_optimal=False),
            doe=lambda m: env(m).update(doe=dict(enabled=True)),
            scaling=lambda m: env(m).update(data_scaling=dict(load_scaling_factor=1.2)),
            sources=lambda m: rec(m).update(source_inventory={'a.py': 'changed'}),
            unfinished=lambda m: m.update(status='started'))
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.check(mutate)


def call(i, stage='LP', status=0, **over):
    from evaluation.forecast_mpc.gurobi_run import GUROBI_FMPC_CONFIG
    record = dict(backend='gurobi', session='S', gurobi_version='12.0.3', threads=4, seed=42, stage=stage,
                  time_limit_s=GUROBI_FMPC_CONFIG['time_limit_s'], mip_rel_gap=GUROBI_FMPC_CONFIG['mip_rel_gap'],
                  call_id=i, status=2, mapped_status=status)
    record.update(over)
    return record


def solve_row(step, calls, ok=True, milp=False, status=0):
    return dict(start_index='0', step=str(step), solver_backend='gurobi', solve_ok=str(ok),
                milp_used='' if milp is None else str(milp), solver_status_code='' if not ok else f'{status}.0',
                solver_native=json.dumps(calls))


class BackendVerification(unittest.TestCase):
    """Regression for audit defect 1: Gurobi-only execution must be proven from invocation records, not labels."""

    def setUp(self):
        from types import SimpleNamespace
        from evaluation.forecast_mpc.gurobi_run import GUROBI_FMPC_CONFIG
        self.gurobi = dict(GUROBI_FMPC_CONFIG, gurobi_version='12.0.3', threads=4)
        self.rows = [solve_row(0, [call(1)]),                                         # LP only
                     solve_row(1, [call(2), call(3, 'MILP', 1)], milp=True, status=1),  # LP overlap -> MILP incumbent
                     solve_row(2, [call(4, status=2)], ok=False, milp=None),          # LP infeasible -> fallback
                     solve_row(3, [call(5), call(6, 'MILP', 4)], ok=False, milp=None)]  # MILP failed -> fallback
        self.ctx = SimpleNamespace(backend=SimpleNamespace(session='S', calls_completed=6), solves=4, highs_calls=0)

    def verify(self, rows=None, **ctx):
        from evaluation.forecast_mpc.gurobi_run import verify_backend_records
        if 'calls_completed' in ctx:
            self.ctx.backend.calls_completed = ctx.pop('calls_completed')
        self.ctx.__dict__.update(ctx)
        return verify_backend_records(self.rows if rows is None else rows, self.ctx, self.gurobi)

    def test_valid_records_including_fallback_solves(self):
        summary = self.verify()
        self.assertEqual((summary['gurobi_calls'], summary['lp_calls'], summary['milp_calls']), (6, 4, 2))

    def test_label_without_invocation_record_is_rejected(self):
        from evaluation.forecast_mpc.gurobi_run import BackendVerificationError
        rows = [dict(r) for r in self.rows]
        rows[0]['solver_native'] = '[]'                     # labelled 'gurobi' but no Gurobi optimize() happened
        bad = {
            'empty record': (rows, dict(calls_completed=5)),
            'missing column': ([dict(self.rows[0], solver_native='')] + self.rows[1:], {}),
            'HiGHS entered': (None, dict(highs_calls=1)),
            'backend ran more calls than recorded': (None, dict(calls_completed=7)),
            'fewer rows than decisions': (None, dict(solves=5)),
            'foreign session': ([solve_row(0, [call(1, session='other')])] + self.rows[1:], {}),
            'wrong time limit': ([solve_row(0, [call(1, time_limit_s=600.0)])] + self.rows[1:], {}),
            'wrong gap': ([solve_row(0, [call(1, mip_rel_gap=1e-6)])] + self.rows[1:], {}),
            'wrong seed': ([solve_row(0, [call(1, seed=7)])] + self.rows[1:], {}),
            'MILP used but no MILP call': ([self.rows[0], solve_row(1, [call(2), call(3)], milp=True)] + self.rows[2:],
                                           {}),
            'duplicate call id': ([self.rows[0], solve_row(1, [call(1), call(3, 'MILP', 1)], milp=True, status=1)]
                                  + self.rows[2:], {}),
            'status not from Gurobi': ([solve_row(0, [call(1)], status=1)] + self.rows[1:], {}),
        }
        for name, (rows_, ctx) in bad.items():
            with self.subTest(name), self.assertRaises(BackendVerificationError):
                self.setUp()
                self.verify(rows_, **ctx)

    def test_decision_guard_fails_when_gurobi_did_not_run(self):
        from types import SimpleNamespace
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend
        from evaluation.forecast_mpc.gurobi_run import BackendVerificationError, controller_class

        class NoSolveController:                            # logs a solve but never calls Gurobi
            topology = 'TP1'

            def __init__(self):
                self.solves = []

            def select_action(self, obs, noise_std=0.0):
                self.solves.append(dict(start_index=0, step=0))
                return np.zeros(2)
        with fake_gurobi():
            backend = GurobiBackend(1, 42)
            ctx = SimpleNamespace(backend=backend, substage=None, solves=0,
                                  attempt=SimpleNamespace(heartbeat=lambda **k: None))
            cls = controller_class(NoSolveController, ctx)
            with self.assertRaisesRegex(BackendVerificationError, 'not the Gurobi backend'):
                cls().select_action(None)                   # solver.milp still HiGHS
            with backend.active(), self.assertRaisesRegex(BackendVerificationError, 'did not run on Gurobi'):
                cls().select_action(None)

    def test_highs_tripwire_fails_the_task_and_is_restored(self):
        import scipy.optimize
        from evaluation.forecast_mpc import gurobi_run, run
        original = scipy.optimize.milp

        def main_calling_highs(argv):                       # a code path that would optimise with SciPy/HiGHS
            scipy.optimize.milp(np.ones(1), bounds=Bounds(0, 1))
        root = Path(tempfile.mkdtemp(dir=TMP)) / 'root'
        with mock.patch.object(gurobi_run, 'expected_identity', return_value=expected()), \
                mock.patch.object(gurobi_run, 'code_files', return_value={'a.py': 'h'}), \
                mock.patch.object(gurobi_run, 'default_forecast_dir', return_value=Path('/x')), \
                mock.patch.object(run, 'main', side_effect=main_calling_highs), fake_gurobi(), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = gurobi_run.main(['task', '--feeder', 'ieee69', '--scale', '1.3', '--topology', 'TP1',
                                    '--max-days', '1', '--output-root', str(root)])
        self.assertEqual(code, 1)
        self.assertIs(scipy.optimize.milp, original)
        attempt, = (root / 'ieee69' / 'load_x130_smoke1' / '.attempts').iterdir()
        record = json.loads((attempt / 'attempt.json').read_text())
        self.assertEqual(record['failure']['exception_type'], 'BackendVerificationError')
        self.assertIn('SciPy/HiGHS', record['failure']['message'])
        self.assertFalse((root / 'ieee69' / 'load_x130_smoke1' / 'TP1').exists())


class SharedFiles(unittest.TestCase):
    """Regression for audit defect 2: shared marker / identity files are written atomically under a lock."""

    def test_failed_write_leaves_previous_file_and_no_temporary(self):
        from evaluation.forecast_mpc.gurobi_run import atomic_json
        folder = Path(tempfile.mkdtemp(dir=TMP))
        path = folder / 'group_identity.json'
        atomic_json(path, dict(a=1))
        with mock.patch('evaluation.forecast_mpc.gurobi_run.os.replace', side_effect=OSError('injected')), \
                self.assertRaises(OSError):
            atomic_json(path, dict(a=2))
        with self.assertRaises(TypeError):                              # serialisation fails before any write
            atomic_json(path, {('tuple', 'key'): 1})
        self.assertEqual(json.loads(path.read_text()), dict(a=1))
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ['group_identity.json'])

    def test_file_is_complete_whenever_visible(self):
        from evaluation.forecast_mpc import gurobi_run
        folder = Path(tempfile.mkdtemp(dir=TMP))
        path = folder / 'marker.json'
        real = os.replace
        seen = []

        def checking_replace(src, dst):
            seen.append((Path(dst).exists(), json.loads(Path(src).read_text()), Path(src).parent == Path(dst).parent))
            return real(src, dst)
        with mock.patch.object(gurobi_run.os, 'replace', checking_replace):
            gurobi_run.atomic_json(path, dict(schema='x', big=list(range(10000))))
        self.assertEqual(seen, [(False, dict(schema='x', big=list(range(10000))), True)])

    def test_concurrent_tasks_share_root_and_group_identity(self):
        import threading
        from evaluation.forecast_mpc.gurobi_run import ROOT_MARKER, check_group, check_root
        root = Path(tempfile.mkdtemp(dir=TMP)) / 'fresh_root'
        env = dict(battery_constraints=[dict(capacity_kwh=850.0)], backup_mode='directional')
        errors, barrier = [], threading.Barrier(12)

        def task(i):
            try:
                barrier.wait()
                check_root(root)
                parent = root / 'ieee69' / 'load_x130'
                parent.mkdir(parents=True, exist_ok=True)
                check_group(parent, env, expected(tp=f'TP{i % 10 + 1}'), dict(seed=42), {'a': 'h'})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=task, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(json.loads((root / ROOT_MARKER).read_text())['schema'], 'graph-marl-dess-fmpc-gurobi/1')
        self.assertEqual(json.loads((root / 'ieee69' / 'load_x130' / 'group_identity.json').read_text())['gurobi'],
                         dict(seed=42))
        self.assertEqual([p.name for p in root.rglob('*.tmp')], [])
        with self.assertRaisesRegex(ValueError, 'differs'):            # a mismatch is a clear error, not a parse error
            check_group(root / 'ieee69' / 'load_x130', env, expected(), dict(seed=7), {'a': 'h'})

    def test_root_with_only_lock_or_temporary_files_is_not_foreign(self):
        from evaluation.forecast_mpc.gurobi_run import ROOT_LOCK, ROOT_MARKER, check_root
        root = Path(tempfile.mkdtemp(dir=TMP))
        (root / ROOT_LOCK).write_bytes(b'')
        (root / f'.{ROOT_MARKER}.123.abc.tmp').write_text('{')                   # another task mid-write
        self.assertEqual(check_root(root), root.resolve())


def fake_forecast_dir(feeder, folder):
    """A December forecast folder with valid records (contents are placeholders; only identity checks run on it)."""
    import hashlib
    from evaluation.forecast_mpc.common import FORECAST_SCHEMA, targets
    folder.mkdir(parents=True)
    for t in targets(feeder):
        (folder / f'{t}.npz').write_bytes(t.encode())
        (folder / f'{t}.json').write_text(json.dumps(dict(schema=FORECAST_SCHEMA, feeder=feeder, target=t, stage='dec',
                                                          npz_sha256=hashlib.sha256(t.encode()).hexdigest())))
    return folder


class ForecastDirectory(unittest.TestCase):
    """Regression for RIT job 118048: the runner looked in results/soc_feasible/<feeder>/forecast_mpc/forecasts/dec."""

    def test_default_is_the_original_fmpc_forecast_folder(self):
        from evaluation.forecast_mpc.gurobi_run import REPO, default_forecast_dir
        for feeder in ('ieee33', 'ieee69'):
            d = default_forecast_dir(feeder)
            self.assertEqual(d, REPO / 'results' / 'soc_feasible_affine_v2' / feeder / 'forecast_mpc' / 'forecasts'
                             / 'dec')
            self.assertTrue(d.is_absolute())
            self.assertNotIn('soc_feasible', d.parts)

    def test_runner_passes_it_for_every_topology_and_load_factor(self):
        from evaluation.forecast_mpc import gurobi_run
        seen = []

        def identity(feeder, tp, scale, max_days, dec_dir, scratch):
            seen.append(('identity', feeder, Path(dec_dir)))
            return expected(feeder, tp)

        def stop(ctx, argv, gurobi, ref):
            seen.append(('run.py', argv[1], Path(argv[argv.index('--forecast-dirs') + 1].partition('=')[2])))
            raise ValueError('stop after the forecast selection')
        root = Path(tempfile.mkdtemp(dir=TMP)) / 'root'
        with mock.patch.object(gurobi_run, 'expected_identity', identity), \
                mock.patch.object(gurobi_run, 'code_files', return_value={'a.py': 'h'}), \
                mock.patch.object(gurobi_run, 'evaluate', stop), fake_gurobi(), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            for feeder, tps in gurobi_run.TOPOLOGIES.items():
                for tp in tps:
                    for scale in ('1.0', '1.1', '1.2', '1.3'):
                        self.assertEqual(gurobi_run.main(['task', '--feeder', feeder, '--scale', scale, '--topology',
                                                          tp, '--output-root', str(root)]), 1)
        self.assertEqual(len(seen), 2 * 4 * (7 + 10))
        for _, feeder, d in seen:
            self.assertEqual(d, gurobi_run.default_forecast_dir(feeder))

    def test_missing_or_invalid_forecasts_fail_explicitly(self):
        from evaluation.forecast_mpc.gurobi_run import forecast_identity
        for feeder in ('ieee33', 'ieee69'):
            good = fake_forecast_dir(feeder, Path(tempfile.mkdtemp(dir=TMP)) / 'dec')
            self.assertEqual(len(forecast_identity(good, feeder)), 2 * (33 if feeder == 'ieee33' else 69))
            cases = dict(no_directory=lambda d: shutil.rmtree(d),
                         no_record=lambda d: (d / 'load_node_1.json').unlink(),
                         no_npz=lambda d: (d / 'pv_node_5.npz').unlink(),
                         changed_npz=lambda d: (d / 'load_node_2.npz').write_bytes(b'changed'),
                         other_stage=lambda d: (d / 'load_node_3.json').write_text(json.dumps(dict(
                             json.loads((d / 'load_node_3.json').read_text()), stage='nov'))),
                         other_feeder=lambda d: (d / 'load_node_4.json').write_text(json.dumps(dict(
                             json.loads((d / 'load_node_4.json').read_text()), feeder='other'))))
            for name, damage in cases.items():
                d = fake_forecast_dir(feeder, Path(tempfile.mkdtemp(dir=TMP)) / 'dec')
                damage(d)
                with self.subTest(feeder=feeder, case=name), self.assertRaises(ValueError):
                    forecast_identity(d, feeder)

    def test_missing_forecasts_fail_the_task_at_inputs_and_forecasts(self):
        from evaluation.forecast_mpc import gurobi_run
        root = Path(tempfile.mkdtemp(dir=TMP)) / 'root'
        missing = Path(tempfile.mkdtemp(dir=TMP)) / 'no_forecasts_here'

        def identity(feeder, tp, scale, max_days, dec_dir, scratch):
            gurobi_run.forecast_identity(dec_dir, feeder)            # the real check, without the heavy topology build
            return expected(feeder, tp)
        err = io.StringIO()
        with mock.patch.object(gurobi_run, 'expected_identity', identity), \
                mock.patch.object(gurobi_run, 'code_files', return_value={'a.py': 'h'}), fake_gurobi(), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = gurobi_run.main(['task', '--feeder', 'ieee33', '--scale', '1.3', '--topology', 'TP1',
                                    '--max-days', '1', '--output-root', str(root), '--forecast-dir', str(missing)])
        self.assertEqual(code, 1)
        attempt, = (root / 'ieee33' / 'load_x130_smoke1' / '.attempts').iterdir()
        record = json.loads((attempt / 'attempt.json').read_text())
        self.assertEqual((record['status'], record['stage']), ('failed', 'inputs_and_forecasts'))
        self.assertIn('does not exist', record['failure']['message'])
        self.assertTrue((attempt / 'failure_traceback.txt').is_file())
        self.assertIn('FMPC_GUROBI_FAILED stage=inputs_and_forecasts', err.getvalue())


class EndToEndSynthetic(unittest.TestCase):
    """One synthetic December day of IEEE33 TP1 through the real F-MPC run with the (fake) Gurobi backend."""

    def test_one_day_smoke_publishes_with_gurobi_only(self):
        from configs.ieee33_config import IEEE33_CONFIG
        from evaluation.forecast_mpc import gurobi_run
        from evaluation.forecast_mpc.common import FORECAST_SCHEMA, day_starts, load_series, save_forecast, targets
        from tests.synthetic_data import actual_forecast_arrays, write_synthetic_series
        work = Path(tempfile.mkdtemp(dir=TMP))
        series_path = write_synthetic_series('ieee33', work / 'synthetic_ieee33.csv')
        with mock.patch.dict(IEEE33_CONFIG['paths'], {'processed_time_series': series_path}):
            s = load_series('ieee33')
            starts = day_starts(s, 'test')[:1]
            dec = work / 'dec'
            dec.mkdir()
            arrays = dict(load=actual_forecast_arrays(s, starts, 'load'), pv=actual_forecast_arrays(s, starts, 'pv'))
            for t in targets('ieee33'):
                kind, bus = t.split('_node_')
                save_forecast(dec, t, arrays[kind][..., int(bus) - 1], starts, s.timestamps[starts],
                              dict(schema=FORECAST_SCHEMA, feeder='ieee33', stage='dec', target=t,
                                   series_sha256=s.sha256))
            root = work / 'root'
            with fake_gurobi(), mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '2'}), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = gurobi_run.main(['task', '--feeder', 'ieee33', '--scale', '1.3', '--topology', 'TP1',
                                        '--max-days', '1', '--output-root', str(root), '--forecast-dir', str(dec)])
                from evaluation.forecast_mpc import gurobi_benchmark
                bench = work / 'bench'
                gurobi_benchmark.main(['capture', '--feeder', 'ieee33', '--topology', 'TP1', '--scale', '1.3',
                                       '--steps', '0', '1', '--forecast-dir', str(dec), '--out', str(bench)])
                gurobi_benchmark.main(['compare', '--out', str(bench)])
        report = json.loads((bench / 'benchmark.json').read_text())
        self.assertEqual([i['step'] for i in report['instances']], [0, 1])
        for inst in report['instances']:
            for what in ('pipeline', 'LP', 'MILP'):
                comparison = inst[f'{what}_comparison']                      # fake Gurobi = HiGHS underneath
                self.assertLess(comparison['objective_rel_diff'], 1e-6)
                self.assertLess(comparison['first_step_max_abs_kw_diff'], 1e-4)
            calls = [c for r in inst['runs'] for c in r['calls']]
            self.assertTrue(all(c['max_row_violation'] < 1e-6 for c in calls))
            self.assertEqual({c['stage'] for r in inst['runs'] if r['what'] == 'MILP' for c in r['calls']}, {'MILP'})
        attempt, = (root / 'ieee33' / 'load_x130_smoke1' / '.attempts').iterdir()
        record = json.loads((attempt / 'attempt.json').read_text())
        tb = attempt / 'failure_traceback.txt'
        self.assertEqual(code, 0, tb.read_text() if tb.is_file() else record)
        self.assertEqual(record['status'], 'completed')
        final = root / 'ieee33' / 'load_x130_smoke1' / 'TP1'
        manifest = json.loads((final / 'tables' / 'run_manifest.json').read_text())
        backend = manifest['forecast_mpc_record']['solver_backend']
        self.assertEqual((backend['backend'], backend['threads'], backend['seed']), ('gurobi', 2, 42))
        import pandas as pd
        solves = pd.read_csv(final / 'tables' / 'fmpc_solves.csv')
        self.assertEqual(len(solves), 96)
        self.assertEqual(set(solves.solver_backend), {'gurobi'})
        native = json.loads(solves.solver_native.iloc[0])
        self.assertEqual(native[0]['stage'], 'LP')
        self.assertEqual(native[0]['time_limit_s'], 60.0)
        total = sum(len(json.loads(n)) for n in solves.solver_native)
        verification = record['gurobi_verification']
        self.assertEqual((verification['solves'], verification['gurobi_calls'], verification['highs_calls']),
                         (96, total, 0))
        self.assertEqual(int(solves.gurobi_calls.sum()), total)
        receipt = json.loads((final / 'completion.json').read_text())
        self.assertEqual(receipt['gurobi_verification'], verification)
        self.assertTrue((root / 'ieee33' / 'load_x130_smoke1' / 'group_identity.json').is_file())


@unittest.skipUnless(os.environ.get('FMPC_GUROBI_INTEGRATION') == '1',
                     'optional: set FMPC_GUROBI_INTEGRATION=1 with a real gurobipy and licence')
class RealGurobi(unittest.TestCase):
    def test_real_gurobi_matches_highs_on_small_problem(self):
        from evaluation.forecast_mpc.gurobi_backend import GurobiBackend
        c, A, lo, hi, lb, ub = small_problem()
        backend = GurobiBackend(1, 42)
        for integ in (np.zeros(4), np.array([0, 0, 1, 0])):
            ours = backend.milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi),
                                options=dict(time_limit=60.0, mip_rel_gap=1e-4))
            ref = milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi))
            self.assertEqual(ours.status, 0)
            self.assertAlmostEqual(ours.fun, ref.fun, places=6)
        print('Gurobi', backend.version, backend.last_calls)


if __name__ == '__main__':
    unittest.main()
