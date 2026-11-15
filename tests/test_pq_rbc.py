"""PQ-RBC and the combined summary on SYNTHETIC data (no research data).

    python -m unittest tests.test_pq_rbc -v
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests.synthetic_data import write_synthetic_series

TMP = None
SERIES = None


def setUpModule():
    global TMP, SERIES
    TMP = Path(tempfile.mkdtemp(prefix='pqrbc-tests-'))
    SERIES = write_synthetic_series('ieee33', TMP / 'series' / 'synthetic_ieee33.csv', seed=1)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def make_env(scale=1.0):
    from configs.experiments import feeder_config
    from environments.marl_env import make_marl_env
    return make_marl_env(feeder_config('ieee33', SERIES, 'soc_feasible'), 'val', 42, 0.0, scale)


class Thresholds(unittest.TestCase):
    def test_training_split_quantiles_frozen_once(self):
        from evaluation.rule_based import pq_rbc
        raw = pd.read_csv(SERIES)
        t = pd.to_datetime(raw.date_time, utc=True)
        train = raw.price[(t >= pd.Timestamp('2018-01-01', tz='UTC')) & (t < pd.Timestamp('2018-10-01', tz='UTC'))]
        record = pq_rbc.compute_thresholds('ieee33', SERIES)
        self.assertAlmostEqual(record['q30'], float(np.quantile(train, 0.3)))
        self.assertAlmostEqual(record['q70'], float(np.quantile(train, 0.7)))
        self.assertEqual((record['rows'], record['split_last_time'][:10]), (273 * 96, '2018-09-30'))
        root = TMP / 'freeze_root'
        argv = ['freeze', '--feeder', 'ieee33', '--output-root', str(root), '--time-series', str(SERIES)]
        path = pq_rbc.main(argv)
        self.assertEqual(path, root / 'ieee33' / 'price_quantile_rbc' / 'thresholds.json')
        with self.assertRaisesRegex(SystemExit, 'never recomputed'):
            pq_rbc.main(argv)


class Rule(unittest.TestCase):
    def test_rule_uses_price_soc_and_thresholds_only(self):
        from environments.battery_feasibility import feasible_powers_kw
        from evaluation.rule_based.pq_rbc import PriceQuantileRBC
        from utility.evaluation_reporting import reset_for_evaluation
        env = make_env()
        reset_for_evaluation(env, int(env.episode_start_indices[3]), 42)
        rbc = PriceQuantileRBC(env, 20.0, 40.0, 'TP1', 1.0)
        self.assertEqual(env.dispatch_mode, 'direct_kw')
        price = rbc.current_price()
        self.assertAlmostEqual(price, env._get_profiles(env.current_index)[3])
        charge, discharge = feasible_powers_kw(env)
        for q30, q70, expected in ((price, price + 1, -charge), (price - 2, price, discharge),
                                   (price - 2, price + 2, np.zeros(5))):
            rbc.q30, rbc.q70 = q30, q70
            np.testing.assert_array_equal(rbc.select_action(None), expected)
        env.batteries[0].soc = env.batteries[0].config.soc_max                 # full battery: no charge available
        charge, _ = feasible_powers_kw(env)
        rbc.q30, rbc.q70 = price, price + 1
        self.assertEqual(rbc.select_action(None)[0], 0.0)
        np.testing.assert_array_equal(rbc.select_action(None), -charge)
        df = env.time_series['df']
        df.loc[env.current_index + 1:, env.time_series['load_cols']] *= 2.0     # load is not an input
        np.testing.assert_array_equal(rbc.select_action(None), -charge)


class EndToEnd(unittest.TestCase):
    def test_validation_day_run_dispatches_direct_kw(self):
        from evaluation.rule_based import pq_rbc
        root = TMP / 'run_root'
        from configs.ieee33_config import IEEE33_CONFIG
        pq_rbc.main(['freeze', '--feeder', 'ieee33', '--output-root', str(root), '--time-series', str(SERIES)])
        with mock.patch.dict(IEEE33_CONFIG['paths'], {'processed_time_series': SERIES}):
            out = pq_rbc.main(['run', '--feeder', 'ieee33', '--split', 'val', '--max-days', '1', '--scales', '1.2',
                               '--output-root', str(root)])[0]
        self.assertEqual(out, root / 'ieee33' / 'price_quantile_rbc_validation' / 'load_x120_first1')
        manifest = json.loads((out / 'run_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'finished')
        days = pd.read_csv(out / 'pqrbc_days.csv')
        self.assertEqual(int(days[['charge_steps', 'idle_steps', 'discharge_steps']].sum(axis=1).iloc[0]), 96)
        steps = pd.read_csv(out / 'step_metrics.csv')
        self.assertTrue((steps['marl_dispatch_source'] == 'direct_kw').all())
        self.assertGreater(int(days.charge_steps.iloc[0] + days.discharge_steps.iloc[0]), 0)


class Summary(unittest.TestCase):
    def write_run(self, d, controller, starts):
        d.mkdir(parents=True)
        (d / 'run_manifest.json').write_text(json.dumps(dict(status='finished')))
        pd.DataFrame(dict(controller=controller, load_scaling_factor=1.0, topology_case='TP1', run_id=np.nan,
                          start_index=starts, adjusted_energy_cost=[10.0 + i for i in range(len(starts))],
                          action_mapping='direct_kw', action_mapping_version='direct_kw')).to_csv(
            d / 'episode_summary.csv', index=False)

    def test_new_references_are_single_runs_and_validation_runs_are_excluded(self):
        from configs.experiments import RunLayout
        from evaluation.summarize import summarize
        root = TMP / 'summary_root'
        base = root / 'ieee33'
        self.write_run(base / 'pf_mpc' / 'load_x100', 'pf_mpc', [0, 96])
        self.write_run(base / 'forecast_mpc' / 'load_x100', 'forecast_mpc', [0, 96])
        self.write_run(base / 'price_quantile_rbc' / 'load_x100', 'price_quantile_rbc', [0, 96])
        self.write_run(base / 'forecast_mpc_validation' / 'load_x100_first1', 'forecast_mpc', [5])
        self.write_run(base / 'price_quantile_rbc_validation' / 'load_x100_first1', 'price_quantile_rbc', [5])
        (base / 'forecast_mpc' / 'models' / 'val').mkdir(parents=True)
        (base / 'forecast_mpc' / 'forecasts' / 'val').mkdir(parents=True)
        out = summarize('ieee33', RunLayout(root), optimization_references=True)
        by = pd.read_csv(out / 'by_controller.csv').set_index('controller')
        self.assertEqual(by.loc['pf_mpc', 'result_type'], 'deterministic_optimization_single_run')
        self.assertEqual(by.loc['forecast_mpc', 'result_type'], 'forecast_optimization_single_run')
        self.assertEqual(by.loc['price_quantile_rbc', 'result_type'], 'rule_based_single_run')
        self.assertTrue((by['runs'] == 1).all())
        included = json.loads((out / 'inclusion.json').read_text())['included']
        self.assertFalse(any('validation' in d or 'models' in d or 'forecasts' in d for d in included))
        self.assertEqual(len(included), 3)
        self.write_run(base / 'forecast_mpc' / 'load_x110', 'forecast_mpc', [0, 0])
        with self.assertRaisesRegex(SystemExit, 'single deterministic optimization or rule'):
            summarize('ieee33', RunLayout(root), optimization_references=True)

    def test_output_folders_never_coincide(self):
        from evaluation.forecast_mpc.common import LABEL as FMPC, forecasts_dir, models_dir
        from evaluation.oracle.run import PF_LABELS
        from evaluation.reference_runs import output_dir
        from evaluation.rule_based.pq_rbc import LABEL as PQ, thresholds_path
        root = Path('results/x')
        dirs = [root / 'ieee33' / PF_LABELS['never'] / 'load_x100', output_dir(root, 'ieee33', FMPC, 'test', 1.0),
                output_dir(root, 'ieee33', PQ, 'test', 1.0), output_dir(root, 'ieee33', FMPC, 'val', 1.0, 3),
                output_dir(root, 'ieee33', PQ, 'val', 1.0, 3), models_dir(root, 'ieee33', 'val'),
                forecasts_dir(root, 'ieee33', 'final'), thresholds_path(root, 'ieee33').parent]
        self.assertEqual(len({str(d) for d in dirs}), len(dirs))
        for a in dirs:
            for b in dirs:
                if a != b:
                    self.assertFalse(str(b).startswith(str(a) + '/') and 'load_x' in b.name and 'load_x' in a.name)
        self.assertEqual(len({FMPC, PQ, PF_LABELS['never']}), 3)


if __name__ == '__main__':
    unittest.main()
