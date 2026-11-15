"""Fast unit tests for the IEEE33 load-stress tools (no power flow, no time-series data, no checkpoints).

    python -m unittest experiments.load_stress.test_load_stress -v
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from configs.ieee33_config import IEEE33_CONFIG
from experiments.load_stress import evaluate_load_stress_ieee33 as E
from experiments.load_stress import summarize_load_stress as S
from experiments.load_stress.stress_env import (check_scale, compare_stress_environment, scale_label, scenario,
    stress_config)
from experiments.no_shared_kpi.protocol import ROOT, sha


def env_record(topology='TP1', load=1.0):
    return dict(topology=topology, mode='test', agent_obs_features=['f'], interface_identity={'a': 1},
                normalization_identity={'n': 1}, data_scaling=dict(load_scaling_factor=load, pv_scaling_factor=1.0,
                price_scaling_factor=1.0), doe=dict(enabled=False), backup={'b': 1}, network={'s': 1}, reward={'r': 1},
                battery_constraints=[{'c': 1}], active_current_ratings_a=[255.0], voltage_tolerance_pu=1e-6,
                current_relative_tolerance=1e-6, expected_steps_per_episode=96, available_episode_start_indices=[0, 96],
                input_files=dict(line=dict(exists=True, sha256='x')))


class Scaling(unittest.TestCase):
    def test_scales(self):
        self.assertEqual([scale_label(s) for s in (1.0, 1.1, 1.2, 1.3)], ['load_x100', 'load_x110', 'load_x120', 'load_x130'])
        self.assertEqual((scenario(1.0), scenario(1.2)), ('production', 'stress'))
        for bad in (0.9, 1.15, 1.4):
            with self.assertRaises(ValueError):
                check_scale(bad)

    def test_stress_config_copies(self):
        before = deepcopy(IEEE33_CONFIG)
        cfg = stress_config(IEEE33_CONFIG, 1.3)
        self.assertEqual(cfg['data_scaling']['load_scaling_factor'], 1.3)
        self.assertEqual(cfg['data_scaling']['pv_scaling_factor'], 1.0)
        self.assertEqual(IEEE33_CONFIG, before)
        changed = deepcopy(IEEE33_CONFIG)
        changed['data_scaling']['load_scaling_factor'] = 1.1
        with self.assertRaises(ValueError):
            stress_config(changed, 1.1)

    def test_compare_only_load_scale_differs(self):
        baseline = dict(environments=[env_record()])
        out = compare_stress_environment(env_record(load=1.2), baseline, 1.2)
        self.assertEqual(out['stressed_data_scaling']['load_scaling_factor'], 1.2)
        with self.assertRaises(ValueError):
            compare_stress_environment(env_record(load=1.1), baseline, 1.2)
        pv = env_record(load=1.2)
        pv['data_scaling']['pv_scaling_factor'] = 1.1
        with self.assertRaises(ValueError):
            compare_stress_environment(pv, baseline, 1.2)
        doe = env_record(load=1.2)
        doe['doe'] = dict(enabled=True)
        with self.assertRaisesRegex(ValueError, 'doe'):
            compare_stress_environment(doe, baseline, 1.2)
        files = env_record(load=1.2)
        files['input_files']['line']['sha256'] = 'y'
        with self.assertRaisesRegex(ValueError, 'bytes'):
            compare_stress_environment(files, baseline, 1.2)


class Guards(unittest.TestCase):
    def test_output_root(self):
        with tempfile.TemporaryDirectory() as d:
            E.check_output_root(Path(d) / 'ls', [Path(d) / 'final'])
            for bad in (ROOT / 'results/ablations/ieee33/x', ROOT / 'results/topology_generalization/csv/y', Path(d)):
                with self.assertRaises(E.StressError):
                    E.check_output_root(bad, [Path(d) / 'final'])

    def test_arguments(self):
        common = ['--job-map', 'j.json']
        with self.assertRaises(SystemExit):
            E.parse_args(['--model', 'a0', '--run', '0', '--scale', '1.1', '--smoke', *common])
        with self.assertRaises(SystemExit):
            E.parse_args(['--model', 'zero', '--run', '1', '--scale', '1.0', *common])
        with self.assertRaises(SystemExit):
            E.parse_args(['--model', 'a0', '--run', '0', '--scale', '1.15', *common])
        self.assertTrue(E.parse_args(['--model', 'a1', '--run', '2', '--scale', '1.3', '--smoke', *common]).smoke)

    def test_gates(self):
        with tempfile.TemporaryDirectory() as d:
            optional = E.identity_gate(d, 'a0', 0, 'ck')
            self.assertEqual((optional['performed'], optional['status']), (False, 'NOT_PERFORMED'))
            run = E.run_dir(d, 'a0', 0, 1.0)
            run.mkdir(parents=True)
            (run / 'run_manifest.json').write_text('{}')
            report = dict(status='PASS', evaluation_manifest_sha256=sha(run / 'run_manifest.json'), checkpoint_sha256='ck')
            (run / 'identity_report.json').write_text(json.dumps(report))
            self.assertEqual(E.identity_gate(d, 'a0', 0, 'ck')['performed'], True)
            self.assertEqual(E.identity_gate(d, 'a0', 0, 'ck')['status'], 'PASS')
            (run / 'identity_report.json').write_text(json.dumps(dict(report, status='FAIL')))
            with self.assertRaisesRegex(E.StressError, 'status FAIL'):
                E.identity_gate(d, 'a0', 0, 'ck')
            (run / 'identity_report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(E.StressError, 'different checkpoint'):
                E.identity_gate(d, 'a0', 0, 'other')
            (run / 'run_manifest.json').write_text('{"changed": 1}')
            with self.assertRaisesRegex(E.StressError, 'changed'):
                E.identity_gate(d, 'a0', 0, 'ck')
            self.assertEqual(E.smoke_gate(d, 'a0', 0, 'ck')['status'], 'NOT_PERFORMED')   # never required
            smoke = E.run_dir(d, 'a0', 0, 1.3, smoke=True)
            smoke.mkdir(parents=True)
            (smoke / 'smoke_check.json').write_text(json.dumps(dict(status='FAIL', checkpoint_sha256='ck')))
            self.assertEqual(E.smoke_gate(d, 'a0', 0, 'ck')['status'], 'FAIL')           # recorded, not blocking

    def test_observation_recorder_passes_action_through(self):
        class Agent:
            def select_action(self, obs, noise_std=0.0):
                return np.full(2, 0.25 + noise_std, dtype=np.float32)

        class Env:
            agent_obs_features = ['time', 'load_kw', 'feeder_grid_import_kw']

        sink = []
        rec = E.ObservationRecorder(Agent(), Env(), sink)
        rec.context = dict(topology_case='TP1', episode_id=0, start_index=5, step_counter=0)
        a = rec.select_action(dict(agent_obs=np.array([[0, 0.5, 1.2], [0, 0.7, 1.2]])))
        np.testing.assert_array_equal(a, np.full(2, 0.25, dtype=np.float32))
        self.assertEqual(sink[0]['load_kw_feature_max'], 0.7)
        self.assertEqual(sink[0]['feeder_grid_import_kw_feature_max'], 1.2)
        self.assertEqual(rec.context['step_counter'], 1)


class Summary(unittest.TestCase):
    def write_run(self, root, model, run, scale, cost, rng, identity='PASS'):
        d = E.run_dir(root, model, run, scale)
        d.mkdir(parents=True)
        (d / 'run_manifest.json').write_text(json.dumps(dict(status='finished')))
        starts = [0, 96, 192]
        ep = pd.DataFrame(dict(topology_case='TP1', start_index=starts, total_energy_cost=cost + rng.normal(0, 0.01, 3),
                               adjusted_energy_cost=cost, feasible_rate=1.0))
        ep.to_csv(d / 'episode_summary.csv', index=False)
        steps = pd.DataFrame(dict(topology_case='TP1', start_index=np.repeat(starts, 2), converged=True,
                                  backup_activated=[True, False] * 3))
        steps.to_csv(d / 'step_metrics.csv', index=False)
        if model != 'zero':
            obs = pd.DataFrame(dict(topology_case='TP1', load_kw_feature_max=[0.5 * scale] * 6,
                                    feeder_grid_import_kw_feature_max=[0.4 * scale] * 6))
            obs.to_csv(d / 'obs_diagnostics.csv', index=False)
        if model != 'zero' and scale == 1.0 and identity:
            (d / 'identity_report.json').write_text(json.dumps(dict(status=identity,
                evaluation_manifest_sha256=sha(d / 'run_manifest.json'))))

    def test_summary(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'root'
            for s in (1.0, 1.1, 1.2, 1.3):
                for r in (0, 1, 2):
                    self.write_run(root, 'a0', r, s, 100 * s + r, rng)
                    self.write_run(root, 'a1', r, s, 90 * s + r, rng, identity=('PASS', None, 'FAIL')[r])
                self.write_run(root, 'zero', 0, s, 150 * s, rng)
            S.main(['--root', str(root), '--output', str(Path(d) / 'out'), '--bootstrap', '200'])
            by_model = pd.read_csv(Path(d) / 'out' / 'delta_vs_1x_by_model.csv')
            row = by_model.query("model == 'a0' and scale == 1.3 and metric == 'adjusted_energy_cost'").iloc[0]
            self.assertAlmostEqual(row.mean_delta, 30.0, places=6)
            vs = pd.read_csv(Path(d) / 'out' / 'policy_minus_zero_by_day.csv')
            v = vs.query("model == 'a0' and run == 0 and scale == 1.0")['adjusted_energy_cost']
            self.assertTrue(np.allclose(v, -50.0))
            ood = pd.read_csv(Path(d) / 'out' / 'obs_out_of_distribution.csv')
            self.assertEqual(ood.query('scale == 1.3')['load_kw_frac_steps_above_1x_max'].min(), 1.0)
            inc = json.loads((Path(d) / 'out' / 'inclusion.json').read_text())
            self.assertEqual(len(inc['included']), 24)       # a0 x3, a1 run 0 (PASS) and run 1 (not performed), zero
            checks = {(i['model'], i['run']): i['identity_check'] for i in inc['included']}
            self.assertEqual((checks['a1', 0], checks['a1', 1], checks['zero', 0]), ('PASS', 'NOT_PERFORMED', 'NOT_APPLICABLE'))
            self.assertNotIn(('a1', 2), checks)                  # identity report present but FAIL -> excluded
            long = pd.read_csv(Path(d) / 'out' / 'episode_metrics_long.csv')
            self.assertEqual(set(long.query("model == 'a1' and run == 1").identity_check), {'NOT_PERFORMED'})


if __name__ == '__main__':
    unittest.main()
