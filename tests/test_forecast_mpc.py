"""F-MPC forecasting pipeline and controller on SYNTHETIC data (no research data, no AutoGluon training).

    python -m unittest tests.test_forecast_mpc -v

AutoGluon is replaced by tests.synthetic_data.StubPredictor (deterministic rules on the feature columns); the series is
a generated 2018 year in the environment's file format. Solver calls here are unit-test calls on that synthetic day.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests.synthetic_data import StubPredictor, actual_forecast_arrays, fake_autogluon, write_synthetic_series

TMP = None
SERIES = None


def setUpModule():
    global TMP, SERIES
    TMP = Path(tempfile.mkdtemp(prefix='fmpc-tests-'))
    SERIES = write_synthetic_series('ieee33', TMP / 'series' / 'synthetic_ieee33.csv')


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def series():
    from evaluation.forecast_mpc.common import load_series
    return load_series('ieee33', SERIES)


def make_env(scale=1.0):
    from configs.experiments import feeder_config
    from environments.marl_env import make_marl_env
    return make_marl_env(feeder_config('ieee33', SERIES, 'soc_feasible'), 'val', 42, 0.0, scale)


def reset(env, day=0):
    from utility.evaluation_reporting import reset_for_evaluation
    reset_for_evaluation(env, int(env.episode_start_indices[day]), 42)
    return int(env.episode_start_indices[day])


def actual_store(s, starts):
    from evaluation.forecast_mpc.common import ForecastStore
    return ForecastStore.from_arrays('ieee33', actual_forecast_arrays(s, starts, 'load'),
                                     actual_forecast_arrays(s, starts, 'pv'), starts, s.timestamps[starts],
                                     stage='val', series_sha256=s.sha256)


class Features(unittest.TestCase):
    def test_window_indexing_and_feature_values(self):
        from evaluation.forecast_mpc.features import FEATURES, feature_frame, training_window
        y = np.arange(2000, dtype=float)
        price = np.arange(2000, dtype=float) * 10
        stamps = pd.date_range('2018-01-08 00:00', periods=2000, freq='15min')
        rows = np.array([700, 1500])
        f = feature_frame(training_window(y, rows), stamps[rows], price, rows)
        self.assertEqual(list(f.columns), list(FEATURES))
        self.assertNotIn('month', f.columns)
        self.assertNotIn('dayofyear', f.columns)
        for k in (1, 2, 96, 672):
            np.testing.assert_array_equal(f[f'y_lag_{k}'], rows - k)
        np.testing.assert_allclose(f['y_roll_mean_4'], rows - 2.5)
        np.testing.assert_allclose(f['y_roll_std_4'], np.std([0, 1, 2, 3], ddof=1))
        np.testing.assert_array_equal(f['y_roll_min_96'], rows - 96)
        np.testing.assert_array_equal(f['y_roll_max_96'], rows - 1)
        np.testing.assert_array_equal(f['lag_96_minus_lag_672'], 576)
        np.testing.assert_array_equal(f['price'], price[rows])
        np.testing.assert_array_equal(f['price_lag_1'], price[rows - 1])
        np.testing.assert_array_equal(f['price_lag_96'], price[rows - 96])
        t = stamps[rows[0]]
        self.assertEqual((f.hour[0], f.minute[0], f.dow[0]), (t.hour, t.minute, t.dayofweek))

    def test_history_features_never_read_the_target_row_or_later(self):
        from evaluation.forecast_mpc.features import training_window, history_features
        y = np.random.default_rng(0).random(3000)
        rows = np.array([1000])
        a = history_features(training_window(y, rows))
        y2 = y.copy()
        y2[1000:] += 100.0                                   # the target row and everything after it
        b = history_features(training_window(y2, rows))
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        y2[999] += 1.0                                       # control: the previous row is used
        self.assertNotEqual(history_features(training_window(y2, rows))['y_lag_1'][0], a['y_lag_1'][0])


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = series()

    def test_monthly_expanding_windows_end_before_each_cutoff(self):
        from evaluation.forecast_mpc.common import STAGES, cutoff_time, training_rows
        expected = dict(oct=('2018-10-01', 273), nov=('2018-11-01', 304), dec=('2018-12-01', 334))
        self.assertEqual(STAGES, ('oct', 'nov', 'dec'))
        for stage, (cutoff, days) in expected.items():
            r = training_rows(self.s, stage)
            self.assertEqual(len(r), days * 96 - 672)                  # expanding window from 2018-01-01
            self.assertEqual(str(cutoff_time(self.s, stage)), f'{cutoff} 00:00:00')
            self.assertEqual(self.s.timestamps[r.max()], pd.Timestamp(cutoff) - pd.Timedelta('15min'))
            self.assertEqual(r.min(), 672)
        nov = training_rows(self.s, 'nov')
        self.assertTrue(set(self.s.splits['val'][:31 * 96]) <= set(nov))          # October is in the nov fit
        self.assertFalse(set(self.s.splits['val'][31 * 96:]) & set(nov))          # November is not
        self.assertFalse(set(self.s.splits['test']) & set(training_rows(self.s, 'dec')))

    def test_later_rows_never_change_a_stage_training_frame(self):
        from evaluation.forecast_mpc.train import training_frame
        for stage, month in (('oct', '2018-10-01'), ('nov', '2018-11-01'), ('dec', '2018-12-01')):
            frame, r = training_frame(self.s, 'pv_node_3', stage)
            self.assertTrue((self.s.timestamps[r] < pd.Timestamp(month)).all())
            np.testing.assert_array_equal(frame['y'], self.s.pv[r, 2])
            changed = series()
            later = np.asarray(changed.timestamps >= pd.Timestamp(month))
            changed.pv[later] += 50.0                                  # the forecast month and everything after it
            changed.price[later] += 7.0
            pd.testing.assert_frame_equal(frame, training_frame(changed, 'pv_node_3', stage)[0])

    def test_stage_days_are_the_stage_month(self):
        from evaluation.forecast_mpc.common import stage_day_starts
        for stage, month, days in (('oct', '2018-10', 31), ('nov', '2018-11', 30), ('dec', '2018-12', 31)):
            starts = stage_day_starts(self.s, stage)
            self.assertEqual(len(starts), days)
            self.assertTrue(all(str(self.s.timestamps[x]).startswith(month) for x in starts))

    def test_train_cli_with_stub_records_and_refuses_overwrite(self):
        from evaluation.forecast_mpc import train
        from evaluation.forecast_mpc.common import read_json
        root = TMP / 'train_root'
        argv = ['--feeder', 'ieee33', '--stage', 'nov', '--targets', 'load_node_2', 'pv_node_3',
                '--output-root', str(root), '--time-series', str(SERIES)]
        with fake_autogluon():
            StubPredictor.RULE = 'lag96'
            out = train.main(argv)
            self.assertEqual(out, root / 'ieee33' / 'forecast_mpc' / 'models' / 'nov')
            rec = read_json(out / 'pv_node_3.json')
            self.assertEqual((rec['stage'], rec['kind'], rec['bus'], rec['train_rows']), ('nov', 'pv', 3, 304 * 96 - 672))
            self.assertEqual(rec['train_last_time'], '2018-10-31 23:45:00')
            self.assertEqual(rec['cutoff_time'], '2018-11-01 00:00:00')
            self.assertEqual(rec['schema'], 'graph-marl-dess-forecast-model/2')
            self.assertEqual(rec['series_sha256'], self.s.sha256)
            self.assertAlmostEqual(rec['clip_upper'], float(self.s.pv[672:304 * 96, 2].max()))
            self.assertIsNone(read_json(out / 'load_node_2.json')['clip_upper'])
            self.assertEqual(rec['settings']['presets'], 'medium_quality')
            self.assertEqual(StubPredictor.fits[-1]['rows'], 304 * 96 - 672)     # exactly the rows given to fit()
            with self.assertRaises(FileExistsError):
                train.main(argv)
            train.main(argv + ['--skip-completed'])                 # skips, writes nothing new
            with self.assertRaises(SystemExit) as status:
                train.main(['--feeder', 'ieee33', '--stage', 'nov', '--status', '--output-root', str(root)])
            self.assertEqual(status.exception.code, 1)              # 64 of 66 targets missing
            with self.assertRaises(SystemExit):                     # the old two-stage names no longer exist
                train.main(['--feeder', 'ieee33', '--stage', 'val', '--status', '--output-root', str(root)])

    def test_chunks_cover_every_target_once(self):
        from evaluation.forecast_mpc.common import select_targets, targets
        for feeder, n in (('ieee33', 6), ('ieee69', 12)):
            chunks = [select_targets(feeder, chunk=k, chunks=n) for k in range(n)]
            self.assertEqual(sum(chunks, []), targets(feeder))
            self.assertEqual(len(targets(feeder)), dict(ieee33=66, ieee69=138)[feeder])


def age(path, seconds=3600):
    """Make every file below `path` look `seconds` old (the recovery refuses folders written in the last 15 min)."""
    import os
    import time
    old = time.time() - seconds
    for p in [path] + [q for q in Path(path).rglob('*')]:
        os.utime(p, (old, old))


class SchemaAndRecovery(unittest.TestCase):
    """The RIT failure (AutoGluon drops PV's always-zero y_roll_min_96 / y_roll_min_672) and its recovery."""

    @classmethod
    def setUpClass(cls):
        cls.s = series()

    def tearDown(self):
        StubPredictor.RULE, StubPredictor.DROP_USELESS, StubPredictor.EXTRA_DROP = 'lag96', True, ()

    def argv(self, root, *targets, stage='oct', extra=()):
        return ['--feeder', 'ieee33', '--stage', stage, '--targets', *targets, '--output-root', str(root),
                '--time-series', str(SERIES), *extra]

    def test_constant_pv_features_dropped_by_autogluon_are_recorded_and_accepted(self):
        from evaluation.forecast_mpc import generate, train
        from evaluation.forecast_mpc.common import read_json
        from evaluation.forecast_mpc.features import FEATURES
        root = TMP / 'schema_root'
        with fake_autogluon():
            out = train.main(self.argv(root, 'pv_node_5', 'load_node_5'))
            self.assertEqual(StubPredictor.fits[-2]['useless'], [])                                           # load_node_5
            self.assertEqual(sorted(StubPredictor.fits[-1]['useless']), ['y_roll_min_672', 'y_roll_min_96'])  # as on RIT
            pv, load = read_json(out / 'pv_node_5.json'), read_json(out / 'load_node_5.json')
            self.assertEqual(pv['input_features'], list(FEATURES))
            self.assertEqual(pv['input_schema_source'], 'predictor.original_features')
            self.assertEqual(sorted(pv['dropped_features']), ['y_roll_min_672', 'y_roll_min_96'])
            self.assertTrue(all(v.startswith('constant') for v in pv['dropped_features'].values()))
            self.assertEqual(sorted(pv['retained_features']), sorted(set(FEATURES) - {'y_roll_min_96', 'y_roll_min_672'}))
            self.assertEqual((load['dropped_features'], len(load['retained_features'])), ({}, len(FEATURES)))
            self.assertFalse(pv['recovered'])
            fdir = generate.main(self.argv(root, 'pv_node_5', extra=('--max-days', '1')))      # full frame in, retained used
            self.assertEqual(read_json(fdir / 'pv_node_5.json')['days'], 1)

    def test_dropping_an_informative_feature_is_refused(self):
        from evaluation.forecast_mpc import train
        StubPredictor.EXTRA_DROP = ('y_lag_1',)
        with fake_autogluon(), self.assertRaisesRegex(ValueError, "informative feature 'y_lag_1'"):
            train.main(self.argv(TMP / 'refuse_root', 'load_node_6'))

    def test_saved_predictor_without_record_is_validated_and_completed(self):
        from evaluation.forecast_mpc import generate, status, train
        from evaluation.forecast_mpc.common import read_json
        root = TMP / 'recover_root'
        directory = root / 'ieee33' / 'forecast_mpc' / 'models' / 'oct'
        with fake_autogluon():
            with mock.patch.object(train, 'write_json', side_effect=RuntimeError('simulated stop before the record')):
                with self.assertRaisesRegex(RuntimeError, 'simulated stop'):
                    train.main(self.argv(root, 'pv_node_7'))
            self.assertTrue((directory / 'pv_node_7' / 'predictor.pkl').is_file())
            self.assertTrue((directory / 'pv_node_7.leaderboard.csv').is_file())
            self.assertFalse((directory / 'pv_node_7.json').exists())          # the state left on RIT
            self.assertEqual(status.classify(directory, 'pv_node_7', 'models'), 'saved_no_record')
            with self.assertRaisesRegex(FileExistsError, 'without a training record'):
                train.main(self.argv(root, 'pv_node_7'))                       # never silently reused
            with self.assertRaisesRegex(RuntimeError, 'may still be training'):
                train.main(self.argv(root, 'pv_node_7', extra=('--skip-completed',)))
            age(directory / 'pv_node_7')
            before = sorted(p.relative_to(root) for p in root.rglob('*'))
            with self.assertRaises(SystemExit) as verified:                    # read-only validation first
                train.main(self.argv(root, 'pv_node_7', extra=('--verify-saved',)))
            self.assertEqual(verified.exception.code, 0)
            self.assertEqual(sorted(p.relative_to(root) for p in root.rglob('*')), before)   # wrote nothing
            fits = len(StubPredictor.fits)
            train.main(self.argv(root, 'pv_node_7', extra=('--skip-completed',)))
            self.assertEqual(len(StubPredictor.fits), fits)                    # recovered, not refitted
            rec = read_json(directory / 'pv_node_7.json')
            self.assertTrue(rec['recovered'])
            self.assertEqual((rec['fitted_rows_check'], rec['train_rows'], rec['fit_seconds']), ('verified', 25536, None))
            self.assertEqual(sorted(rec['dropped_features']), ['y_roll_min_672', 'y_roll_min_96'])
            self.assertEqual(status.classify(directory, 'pv_node_7', 'models'), 'complete')
            fdir = generate.main(self.argv(root, 'pv_node_7', extra=('--max-days', '1')))
            self.assertTrue((fdir / 'pv_node_7.npz').is_file())

    def test_unusable_saved_folders_are_refused_and_kept(self):
        from evaluation.forecast_mpc import train
        from evaluation.forecast_mpc.train import training_frame
        root = TMP / 'refuse_saved_root'
        directory = root / 'ieee33' / 'forecast_mpc' / 'models' / 'oct'
        (directory / 'load_node_8' / 'models').mkdir(parents=True)              # fit stopped before the final save
        (directory / 'load_node_8' / 'models' / 'partial.bin').write_bytes(b'x')
        (directory / 'load_node_8' / 'learner.pkl').write_text('x')
        frame, _ = training_frame(self.s, 'load_node_10', 'oct')
        StubPredictor(path=directory / 'load_node_9').fit(frame)                # corrupted predictor file
        (directory / 'load_node_9' / 'predictor.pkl').write_text('not a predictor')
        StubPredictor(path=directory / 'load_node_10').fit(frame.iloc[:5000])  # fitted on other rows
        StubPredictor(path=directory / 'load_node_11').fit(training_frame(self.s, 'load_node_11', 'oct')[0])
        (directory / 'load_node_11' / 'version.txt').write_text('0.8.2')        # saved by another AutoGluon version
        StubPredictor(path=directory / 'load_node_12').fit(training_frame(self.s, 'load_node_12', 'oct')[0])
        pd.DataFrame(dict(model=['OtherModel'])).to_csv(directory / 'load_node_12.leaderboard.csv', index=False)
        for t in ('load_node_8', 'load_node_9', 'load_node_10', 'load_node_11', 'load_node_12'):
            age(directory / t)
        cases = dict(load_node_8='not a finished AutoGluon save', load_node_9='cannot be loaded',
                     load_node_10='fitted on 5000 rows', load_node_11='saved with AutoGluon 0.8.2',
                     load_node_12='not in the saved leaderboard')
        with fake_autogluon():
            for t, message in cases.items():
                with self.assertRaisesRegex(RuntimeError, message):
                    train.main(self.argv(root, t, extra=('--skip-completed',)))
                self.assertTrue((directory / t).is_dir() and not (directory / f'{t}.json').exists())
            with self.assertRaises(SystemExit) as verified:                    # the read-only check reports them too
                train.main(self.argv(root, *cases, extra=('--verify-saved',)))
            self.assertEqual(verified.exception.code, 1)

    def test_status_needs_no_torch_and_names_tasks(self):
        import subprocess
        import sys
        from evaluation.forecast_mpc.common import select_targets, targets
        from evaluation.forecast_mpc.status import chunk_index
        for feeder, n in (('ieee33', 6), ('ieee69', 12)):
            names = targets(feeder)
            for k in range(n):
                for t in select_targets(feeder, chunk=k, chunks=n):
                    self.assertEqual(chunk_index(names.index(t), len(names), n), k)
        root = TMP / 'status_root'
        d = root / 'ieee33' / 'forecast_mpc' / 'models' / 'oct'
        (d / 'pv_node_1' / 'models' / 'best').mkdir(parents=True)              # the layout found on RIT
        (d / 'pv_node_1' / 'models' / 'best' / 'model.pkl').write_text('m')
        for name in ('predictor.pkl', 'learner.pkl', 'metadata.json', 'version.txt'):
            (d / 'pv_node_1' / name).write_text('x')
        (d / 'pv_node_12').mkdir(parents=True)
        (d / 'pv_node_12' / 'predictor.pkl').write_text('{}')                   # without learner.pkl etc.: incomplete
        (d / 'load_node_1.json').write_text(json.dumps(dict(stage='oct', cutoff_time='2018-10-01 00:00:00',
                                                            train_rows=25536)))
        code = ('import sys, importlib.abc\n'
                'class B(importlib.abc.MetaPathFinder):\n'
                '    def find_spec(self, name, path=None, target=None):\n'
                '        if name.split(".")[0] in ("torch", "numpy", "pandas", "autogluon", "configs", "environments"):\n'
                '            raise ImportError("blocked " + name)\n'
                f'sys.meta_path.insert(0, B()); sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n'
                'from evaluation.forecast_mpc import status\n'
                f'status.main(["--feeder", "ieee33", "--stage", "oct", "--output-root", {str(root)!r}, "--chunks", "6"])\n')
        proc = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn('INCOMPLETE 1/66 | saved_no_record 1 | incomplete 1 | missing 63', proc.stdout)
        self.assertIn('saved_no_record pv_node_1 (task 3)', proc.stdout)
        self.assertIn('incomplete      pv_node_12 (task 4)', proc.stdout)
        self.assertNotIn('blocked', proc.stderr)


class RecursiveForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = series()
        from evaluation.forecast_mpc.common import day_starts
        cls.starts = day_starts(cls.s, 'val')[:2]

    def forecast(self, y, rule, lower=0.0, upper=None):
        from evaluation.forecast_mpc.generate import recursive_forecast
        column = dict(lag96='y_lag_96', lag1='y_lag_1')[rule]
        return recursive_forecast(y, self.s.price, self.s.timestamps, self.starts, lambda X: X[column].to_numpy(),
                                  lower, upper)

    def test_indexing_seasonal_and_last_value_rules(self):
        y = self.s.load[:, 5]
        f96, _ = self.forecast(y, 'lag96')
        f1, _ = self.forecast(y, 'lag1')
        for d, s in enumerate(self.starts):
            for k in (0, 1, 50, 95):
                np.testing.assert_allclose(f96[d, k, k:], y[s + np.arange(k, 96) - 96], rtol=1e-6)   # always < s + k
                np.testing.assert_allclose(f1[d, k, k:], y[s + k - 1], rtol=1e-6)                   # last observed
                self.assertTrue(np.isnan(f1[d, k, :k]).all())

    def test_forecast_issued_at_t_ignores_rows_from_t_on(self):
        from evaluation.forecast_mpc.generate import recursive_forecast
        y = self.s.load[:, 7].copy()
        mix = lambda X: X.to_numpy(dtype=float).mean(axis=1)            # depends on every feature
        base, _ = recursive_forecast(y, self.s.price, self.s.timestamps, self.starts, mix)
        d, k = 1, 30
        t = self.starts[d] + k
        y2 = y.copy()
        y2[t:] *= 3.0
        changed, _ = recursive_forecast(y2, self.s.price, self.s.timestamps, self.starts, mix)
        np.testing.assert_array_equal(base[d, k], changed[d, k])
        np.testing.assert_array_equal(base[0], changed[0])               # the earlier day is untouched too
        y3 = y.copy()
        y3[t - 1] += 5.0                                                  # control: row t - 1 is observed and used
        self.assertFalse(np.array_equal(base[d, k], recursive_forecast(y3, self.s.price, self.s.timestamps, self.starts,
                                                                       mix)[0][d, k]))

    def test_clipping_is_counted(self):
        y = self.s.pv[:, 1]
        f, clipped = self.forecast(y, 'lag96', lower=0.0, upper=10.0)
        self.assertLessEqual(np.nanmax(f), 10.0)
        self.assertGreater(clipped['above_upper'], 0)
        self.assertEqual(clipped['predictions'], 2 * sum(96 - k for k in range(96)))

    def test_generate_cli_with_stub(self):
        from evaluation.forecast_mpc import generate, train
        from evaluation.forecast_mpc.common import read_json
        root = TMP / 'generate_root'
        common = ['--feeder', 'ieee33', '--targets', 'load_node_4', '--output-root', str(root), '--time-series', str(SERIES)]
        with fake_autogluon():
            StubPredictor.RULE = 'lag96'
            for stage in ('oct', 'nov'):
                train.main(common + ['--stage', stage])
            out = generate.main(common + ['--stage', 'oct', '--max-days', '2'])
            self.assertEqual(out.name, 'oct_first2')
            rec = read_json(out / 'load_node_4.json')
            self.assertEqual((rec['days'], rec['first_day'], rec['stage'], rec['month'], rec['model_cutoff_time']),
                             (2, '2018-10-01 00:00:00', 'oct', '2018-10', '2018-10-01 00:00:00'))
            with np.load(out / 'load_node_4.npz') as z:
                f = z['forecast']
            s = self.starts[0]
            np.testing.assert_allclose(f[0, 10, 10:], self.s.load[s + np.arange(10, 96) - 96, 3], rtol=1e-6)
            nov = generate.main(common + ['--stage', 'nov', '--max-days', '1'])          # November uses the nov models
            rec = read_json(nov / 'load_node_4.json')
            self.assertEqual((rec['first_day'], rec['last_day'], rec['model_cutoff_time']),
                             ('2018-11-01 00:00:00', '2018-11-01 00:00:00', '2018-11-01 00:00:00'))
            self.assertTrue(rec['model_record'].endswith(str(Path('models') / 'nov' / 'load_node_4.json')))
            with self.assertRaises(FileExistsError):
                generate.main(common + ['--stage', 'oct', '--max-days', '2'])
            model = root / 'ieee33' / 'forecast_mpc' / 'models' / 'oct' / 'load_node_4' / 'predictor.pkl'
            model.write_text(model.read_text().replace('lag96', 'lag1'))       # tampered model is refused
            with self.assertRaisesRegex(ValueError, 'differs from the predictor recorded'):
                generate.main(common + ['--stage', 'oct', '--max-days', '1'])


class Validation(unittest.TestCase):
    def test_horizon_metrics_match_the_baselines_they_reproduce(self):
        from evaluation.forecast_mpc.common import day_starts
        from evaluation.forecast_mpc.generate import recursive_forecast
        from evaluation.forecast_mpc.validate import horizon_rows
        s = series()
        starts = day_starts(s, 'val')[:2]
        y = s.load[:, 9]
        for column, base in (('y_lag_96', 'same_time_yesterday'), ('y_lag_1', 'last_value')):
            f, _ = recursive_forecast(y, s.price, s.timestamps, starts, lambda X, c=column: X[c].to_numpy())
            t = pd.DataFrame(horizon_rows(f, y, starts, 'x', 'load', 10))
            np.testing.assert_allclose(t['mae'], t[f'mae_{base}'], rtol=1e-5)
            self.assertEqual(list(t.steps_ahead), list(range(1, 97)))
            self.assertEqual(list(t.n), [2 * (96 - h) for h in range(96)])

    def test_perfect_forecasts_score_zero_and_pass_physical_checks(self):
        from evaluation.forecast_mpc.common import day_starts
        from evaluation.forecast_mpc.validate import physical_checks, validate
        s = series()
        store = actual_store(s, day_starts(s, 'val')[:2])
        table, summary, _ = validate(store, s)
        self.assertAlmostEqual(summary['load_total']['mae'], 0.0, places=6)
        self.assertGreater(summary['load_total']['mae_last_value'], 0.0)
        self.assertGreater(summary['pv_total_daylight']['n'], 0)
        checks = physical_checks(store, s)
        self.assertEqual(int(checks.nonfinite.sum() + checks.negative.sum()), 0)
        self.assertLess(float(checks.night_max_kw.max()), 1e-9)          # sin(0) rounding in the synthetic PV


class Controller(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation.forecast_mpc.common import day_starts
        cls.s = series()
        cls.starts = day_starts(cls.s, 'val')

    def controller(self, env, store, scale):
        from evaluation.forecast_mpc.controller import ForecastMPCController
        from evaluation.oracle.network import LinDistFlowNetwork
        from evaluation.oracle.solver import OracleSettings
        return ForecastMPCController(env, LinDistFlowNetwork(env), OracleSettings(), store, 'ieee33', 'TP1', scale,
                                     price_mode='full_day')

    def test_inputs_equal_pf_mpc_inputs_when_forecasts_are_the_actuals(self):
        """Same optimisation model: with perfect forecasts the F-MPC input is exactly PF-MPC's (load x lambda, kvar
        rule, unscaled PV, prices, SoC, terminal valuation) and so is the plan."""
        from evaluation.forecast_mpc.controller import forecast_day_data
        from evaluation.oracle.network import LinDistFlowNetwork
        from evaluation.oracle.solver import OracleSettings, day_data, solve_day
        env = make_env(1.2)
        env.set_dispatch_mode('direct_kw')
        start = reset(env, 45)                                     # a November day outside the DST window
        store = actual_store(self.s, np.array([start]))
        for step in (0, 37):
            while env.current_step < step:
                env.step(np.zeros(env.num_agents))
            soc = np.array([b.get_soc() for b in env.batteries])
            ours, substituted = forecast_day_data(env, store, step, soc, 1.2, 'dst_guard')
            self.assertFalse(substituted)
            ref = day_data(env, step, soc)
            for field in ('load_kw', 'load_kvar', 'pv_kw', 'price', 'soc0', 'capacity_kwh', 'p_ch_max_kw'):
                np.testing.assert_allclose(getattr(ours, field), getattr(ref, field), rtol=1e-12, atol=1e-9,
                                           err_msg=field)
            for field in ('day_mean_price', 'day_start_energy_kwh', 'dt_hours', 'export_limit_kw'):
                self.assertAlmostEqual(getattr(ours, field), getattr(ref, field), places=9)
        net = LinDistFlowNetwork(env)
        np.testing.assert_allclose(solve_day(net, ours, OracleSettings(), False).net_power_kw,
                                   solve_day(net, ref, OracleSettings(), False).net_power_kw, atol=1e-9)

    def test_load_scaled_pv_not(self):
        from evaluation.forecast_mpc.controller import forecast_day_data
        env = make_env(1.3)
        start = reset(env, 45)
        store = actual_store(self.s, np.array([start]))
        soc = np.array([b.get_soc() for b in env.batteries])
        d, _ = forecast_day_data(env, store, 0, soc, 1.3, 'full_day')
        np.testing.assert_allclose(d.load_kw, 1.3 * self.s.load[start:start + 96])
        np.testing.assert_allclose(d.pv_kw, self.s.pv[start:start + 96])

    def test_controller_uses_no_realized_profiles_measured_soc_and_one_solve_per_step(self):
        from evaluation.oracle import solver as solver_module
        env = make_env(1.0)
        start = reset(env, 45)
        store = actual_store(self.s, np.array([start]))
        ctl = self.controller(env, store, 1.0)
        real_profiles = env._get_profiles

        def guarded(index):
            if guarded.inside:
                raise AssertionError('controller read realized profiles')
            return real_profiles(index)
        guarded.inside = False
        env._get_profiles = guarded
        with mock.patch.object(solver_module, 'day_data', side_effect=AssertionError('day_data used')):
            for _ in range(3):
                soc = [b.get_soc() for b in env.batteries]
                guarded.inside = True
                request = ctl.select_action(None)
                guarded.inside = False
                self.assertEqual(ctl.solves[-1]['soc0'], soc)
                env.step(request)
        self.assertEqual([r['step'] for r in ctl.solves], [0, 1, 2])
        self.assertEqual(ctl.day['solves'], 3)
        self.assertEqual([r['horizon_steps'] for r in ctl.solves], [96, 95, 94])
        self.assertTrue(all(r['solve_ok'] and r['fallback'] == 'none' for r in ctl.solves))

    def test_realized_future_does_not_change_the_action(self):
        env_a, env_b = make_env(1.0), make_env(1.0)
        start = reset(env_a, 45)
        reset(env_b, 45)
        df = env_b.time_series['df']
        cols = env_b.time_series['load_cols'] + env_b.time_series['renewable_cols']
        df.loc[start + 1:start + 95, cols] = df.loc[start + 1:start + 95, cols] * 1.5     # realized future changed
        store = actual_store(self.s, np.array([start]))
        a = self.controller(env_a, store, 1.0).select_action(None)
        b = self.controller(env_b, store, 1.0).select_action(None)
        np.testing.assert_array_equal(a, b)

    def test_failed_solve_falls_back_to_zero_and_is_logged(self):
        from evaluation.forecast_mpc import controller as module
        from evaluation.oracle.solver import PlanError
        requests = []
        for _ in range(2):                                          # deterministic
            env = make_env(1.0)
            start = reset(env, 45)
            ctl = self.controller(env, actual_store(self.s, np.array([start])), 1.0)
            with mock.patch.object(module, 'solve_day', side_effect=PlanError('forced failure')):
                request = ctl.select_action(None)
            env.step(request)
            requests.append(request)
            row = ctl.solves[-1]
            self.assertEqual((row['solve_ok'], row['fallback'], row['proven_optimal']), (False, 'zero_kw_request', False))
            self.assertIn('forced failure', row['solver_message'])
            self.assertEqual((ctl.day['failed_solves'], ctl.day['fallback_steps']), (1, 1))
        np.testing.assert_array_equal(requests[0], np.zeros(5))
        np.testing.assert_array_equal(requests[0], requests[1])

    def test_missing_forecast_is_an_error_not_a_fallback(self):
        env = make_env(1.0)
        reset(env, 45)
        ctl = self.controller(env, actual_store(self.s, np.array([self.starts[0]])), 1.0)
        with self.assertRaises(KeyError):
            ctl.select_action(None)

    def test_dst_guard_prices(self):
        from evaluation.forecast_mpc.controller import known_day_prices
        env = make_env(1.0)
        start = reset(env, 0)                                       # 2018-10-01: inside the guarded window
        full, sub_full = known_day_prices(env, 'full_day')
        guarded, sub = known_day_prices(env, 'dst_guard')
        self.assertTrue(sub and not sub_full)
        np.testing.assert_array_equal(guarded[:92], full[:92])
        np.testing.assert_array_equal(guarded[92:], self.s.price[start - 4:start])
        start = reset(env, 45)                                      # 2018-11-15: outside
        guarded, sub = known_day_prices(env, 'dst_guard')
        self.assertFalse(sub)
        np.testing.assert_array_equal(guarded, self.s.price[start:start + 96])


def write_stage_dir(s, directory, stage, starts):
    """A complete forecast folder of one monthly stage holding 'perfect' (actual) forecasts of the given days."""
    from evaluation.forecast_mpc.common import FORECAST_SCHEMA, save_forecast, targets
    directory.mkdir(parents=True)
    arrays = dict(load=actual_forecast_arrays(s, starts, 'load'), pv=actual_forecast_arrays(s, starts, 'pv'))
    for t in targets('ieee33'):
        kind, bus = t.split('_node_')
        save_forecast(directory, t, arrays[kind][..., int(bus) - 1], starts, s.timestamps[starts],
                      dict(schema=FORECAST_SCHEMA, feeder='ieee33', stage=stage, target=t, series_sha256=s.sha256))
    return directory


class Stores(unittest.TestCase):
    def test_monthly_stores_check_their_month_and_merge(self):
        from evaluation.forecast_mpc.common import ForecastStore, day_starts
        s = series()
        val = day_starts(s, 'val')
        root = TMP / 'stores'
        oct_dir = write_stage_dir(s, root / 'oct', 'oct', val[[0, 1]])
        nov_dir = write_stage_dir(s, root / 'nov', 'nov', val[[31]])
        wrong = write_stage_dir(s, root / 'wrong', 'nov', val[[2]])                 # an October day as 'nov'
        a, b = ForecastStore(oct_dir, 'ieee33', s.sha256, 'oct'), ForecastStore(nov_dir, 'ieee33', s.sha256, 'nov')
        with self.assertRaisesRegex(ValueError, 'outside 2018-11'):
            ForecastStore(wrong, 'ieee33', s.sha256, 'nov')
        with self.assertRaisesRegex(ValueError, "expected 'oct'"):
            ForecastStore(nov_dir, 'ieee33', s.sha256, 'oct')
        merged = ForecastStore.merge([a, b])
        self.assertEqual(merged.stage, 'oct+nov')
        self.assertEqual(merged.day_start_index.tolist(), val[[0, 1, 31]].tolist())
        np.testing.assert_array_equal(merged.window(val[31], 5)[0], b.window(val[31], 5)[0])
        with self.assertRaisesRegex(ValueError, 'more than one store'):
            ForecastStore.merge([a, a])

    def test_validate_cli_reports_each_month_and_pooled(self):
        from evaluation.forecast_mpc import validate
        from evaluation.forecast_mpc.common import day_starts
        s = series()
        val = day_starts(s, 'val')
        root = TMP / 'validate_root'
        oct_dir = write_stage_dir(s, root / 'f_oct', 'oct', val[[3]])
        nov_dir = write_stage_dir(s, root / 'f_nov', 'nov', val[[40]])
        out = validate.main(['--feeder', 'ieee33', '--stage', 'oct', 'nov', '--forecast-dirs', str(oct_dir), str(nov_dir),
                             '--output-root', str(root), '--time-series', str(SERIES)])
        self.assertTrue(out.name.startswith('oct+nov_'))
        table = pd.read_csv(out / 'horizon_metrics.csv')
        self.assertEqual(set(table.scope), {'oct', 'nov', 'pooled'})
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['scopes']['pooled']['days'], 2)
        self.assertLess(summary['scopes']['oct']['feeder_totals']['load_total']['mae'], 1e-3)  # float32 storage rounding
        self.assertEqual(summary['physical_problems'], dict(nonfinite=0, negative=0, above_training_max=0))
        with self.assertRaises(SystemExit):                      # December is never pooled with the validation months
            validate.main(['--feeder', 'ieee33', '--stage', 'nov', 'dec', '--output-root', str(root)])


class EndToEnd(unittest.TestCase):
    def test_validation_days_of_two_months_use_their_own_stage_forecasts(self):
        from evaluation.forecast_mpc import run
        from evaluation.forecast_mpc.common import day_starts
        s = series()
        val = day_starts(s, 'val')
        root = TMP / 'e2e_root'
        dirs = {stage: write_stage_dir(s, root / f'forecasts_{stage}', stage, val[[pos]])
                for stage, pos in (('oct', 0), ('nov', 31))}                       # 2018-10-01 and 2018-11-01
        argv = ['--feeder', 'ieee33', '--split', 'val', '--days', '0', '31', '--scales', '1.1', '--output-root', str(root),
                '--forecast-dirs', f'oct={dirs["oct"]}', f'nov={dirs["nov"]}']
        from configs.ieee33_config import IEEE33_CONFIG
        # the run uses the CONFIGURED series (the energy accounting reads prices from it); point it at the synthetic one
        with mock.patch.dict(IEEE33_CONFIG['paths'], {'processed_time_series': SERIES}):
            out = run.main(argv)[0]
        self.assertEqual(out, root / 'ieee33' / 'forecast_mpc_validation' / 'load_x110_days0-31')
        manifest = json.loads((out / 'run_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'finished')
        record = manifest['forecast_mpc_record']
        self.assertEqual(record['forecast_stores']['TP1']['stage'], 'oct+nov')
        self.assertEqual(set(record['forecast_stages']), {'oct', 'nov'})
        summary = record['summary']
        self.assertEqual((summary['solves'], summary['max_solves_per_day'], summary['failed_solves']), (192, 96, 0))
        solves = pd.read_csv(out / 'fmpc_solves.csv')
        self.assertEqual(len(solves), 192)
        self.assertEqual(sorted(set(solves.start_index)), sorted(val[[0, 31]].tolist()))
        self.assertIn('backup_activated', solves)
        steps = pd.read_csv(out / 'step_metrics.csv')
        self.assertTrue((steps['marl_dispatch_source'] == 'direct_kw').all())
        np.testing.assert_allclose(steps['load_scaling_factor'], 1.1)
        self.assertTrue((out / 'fmpc_days.csv').is_file())
        with mock.patch.dict(IEEE33_CONFIG['paths'], {'processed_time_series': SERIES}), \
                self.assertRaises(FileExistsError):
            run.main(argv)


if __name__ == '__main__':
    unittest.main()
