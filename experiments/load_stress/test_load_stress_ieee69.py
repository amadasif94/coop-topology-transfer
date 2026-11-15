"""Fast unit tests for the IEEE69 load-stress tools (no power flow, no time-series data, no checkpoints).

    python -m unittest experiments.load_stress.test_load_stress_ieee69 -v
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from configs.ieee69_config import IEEE69_CONFIG
from experiments.load_stress import evaluate_load_stress_ieee69 as E
from experiments.load_stress import summarize_load_stress_ieee69 as S
from experiments.load_stress import validate_load_stress_identity_ieee69 as V
from experiments.load_stress.stress_env import stress_config
from experiments.load_stress.stress_env_ieee69 import compare_stress_environment_ieee69
from experiments.no_shared_kpi.ieee69.protocol import DESS_BUSES
from experiments.no_shared_kpi.protocol import ROOT, sha


def env_record(topology='TP1', load=1.0):
    return dict(feeder='IEEE69', topology=topology, mode='test', agent_obs_features=['f'],
                interface_identity=dict(dess_buses=DESS_BUSES), normalization_identity={'n': 1},
                data_scaling=dict(load_scaling_factor=load, pv_scaling_factor=1.0, price_scaling_factor=1.0),
                doe=dict(enabled=False), backup={'b': 1}, network={'s': 1}, reward={'r': 1}, battery_constraints=[{'c': 1}] * 9,
                active_current_ratings_a=[255.0], voltage_tolerance_pu=1e-6, current_relative_tolerance=1e-6,
                expected_steps_per_episode=96, available_episode_start_indices=[0, 96],
                input_files=dict(line_data=dict(exists=True, sha256='x', path='line.csv')))


def fake_refs(topologies, selection_sha='s1'):
    refs = {}
    for i, tp in enumerate(topologies):
        group = 'a0/tp1_7/run_manifest.json' if int(tp[2:]) <= 7 else ('a0/tp8_9/run_manifest.json' if int(tp[2:]) <= 9
                                                                        else 'a0/tp10/run_manifest.json')
        refs[tp] = dict(path=group, manifest=dict(evaluation_protocol=dict(selection=dict(selection_sha256=selection_sha))),
                        plan=[(e, 96 * e, 123 + e) for e in range(31)])
    return refs


class Environment(unittest.TestCase):
    def test_stress_config_ieee69(self):
        before = deepcopy(IEEE69_CONFIG)
        cfg = stress_config(IEEE69_CONFIG, 1.2)
        self.assertEqual((cfg['data_scaling']['load_scaling_factor'], cfg['data_scaling']['pv_scaling_factor']), (1.2, 1.0))
        self.assertEqual(IEEE69_CONFIG, before)

    def test_compare_uses_ieee69_rules(self):
        baseline = dict(environments=[env_record()])
        self.assertEqual(compare_stress_environment_ieee69(env_record(load=1.3), baseline, 1.3)
                         ['stressed_data_scaling']['load_scaling_factor'], 1.3)
        pv = env_record(load=1.3)
        pv['data_scaling']['pv_scaling_factor'] = 1.1
        with self.assertRaises(ValueError):
            compare_stress_environment_ieee69(pv, baseline, 1.3)
        buses = env_record(load=1.3)
        buses['interface_identity']['dess_buses'] = DESS_BUSES[::-1]
        with self.assertRaisesRegex(ValueError, 'IEEE69'):
            compare_stress_environment_ieee69(buses, baseline, 1.3)
        files = env_record(load=1.3)
        files['input_files']['line_data']['sha256'] = 'y'
        with self.assertRaisesRegex(ValueError, 'input difference'):
            compare_stress_environment_ieee69(files, baseline, 1.3)
        norm = env_record(load=1.3)
        norm['normalization_identity'] = {'n': 2}
        with self.assertRaisesRegex(ValueError, 'normalization'):
            compare_stress_environment_ieee69(norm, baseline, 1.3)


class Arguments(unittest.TestCase):
    def test_arguments(self):
        common = ['--job-map', 'j.json']
        for bad in (['--model', 'a0', '--run', '0', '--scale', '1.0', '--pilot'],
                    ['--model', 'zero', '--run', '0', '--scale', '1.1'],
                    ['--model', 'a0', '--run', '0', '--scale', '1.1', '--reference-manifest', 'x.json'],
                    ['--model', 'a1', '--run', '3', '--scale', '1.1']):
            with self.assertRaises(SystemExit):
                E.parse_args([*bad, *common])
        self.assertEqual(E.parse_args(['--model', 'a2', '--run', '1', '--scale', '1.2', *common]).output_root,
                         Path('results/load_stress/ieee69'))

    def test_output_root(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / 'results/load_stress/ieee69'
            E.check_output_root_ieee69(base, [Path(d) / 'a0'], base=base)
            E.check_output_root_ieee69(base / 'sub', [], base=base)
            with self.assertRaises(E.StressError):
                E.check_output_root_ieee69(Path(d) / 'elsewhere', [], base=base)
            for bad in (ROOT / 'results/load_stress/ieee33', ROOT / 'results/ablations/ieee69'):
                with self.assertRaises(E.StressError):
                    E.check_output_root_ieee69(bad, [])
        self.assertEqual(E.run_dir('r', 'a1', 2, 1.3, pilot=True), Path('r/a1_no_shared_kpi/run_2/load_x130_pilot'))

    def test_episode_count(self):
        self.assertEqual(len(E.MODELS) * 3 * 3 * len(E.TOPOLOGIES) * E.DAYS, 8370)


class OptionalChecks(unittest.TestCase):
    def test_identity_is_optional_but_failures_block(self):
        with tempfile.TemporaryDirectory() as d:
            st = E.identity_status(d, 'a0', 0, 'ck')
            self.assertEqual((st['performed'], st['status']), (False, 'NOT_PERFORMED'))
            run = E.run_dir(d, 'a0', 0, 1.0)
            run.mkdir(parents=True)
            (run / 'run_manifest.json').write_text('{}')
            report = dict(evaluation_manifest_sha256=sha(run / 'run_manifest.json'), checkpoint_sha256='ck')
            for status in ('PASS', 'PASS_PARTIAL'):
                (run / 'identity_report.json').write_text(json.dumps(dict(report, status=status)))
                self.assertEqual(E.identity_status(d, 'a0', 0, 'ck')['status'], status)
            (run / 'identity_report.json').write_text(json.dumps(dict(report, status='FAIL')))
            with self.assertRaisesRegex(E.StressError, 'FAIL'):
                E.identity_status(d, 'a0', 0, 'ck')
            (run / 'identity_report.json').write_text(json.dumps(dict(report, status='PASS')))
            with self.assertRaisesRegex(E.StressError, 'different checkpoint'):
                E.identity_status(d, 'a0', 0, 'other')
            (run / 'run_manifest.json').write_text('{"changed": true}')
            with self.assertRaisesRegex(E.StressError, 'changed'):
                E.identity_status(d, 'a0', 0, 'ck')

    def test_pilot_never_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(E.pilot_status(d, 'a1', 0, 'ck'), [dict(status='NOT_PERFORMED')])
            pilot = E.run_dir(d, 'a1', 0, 1.3, pilot=True)
            pilot.mkdir(parents=True)
            (pilot / 'pilot_check.json').write_text(json.dumps(dict(status='FAIL', checkpoint_sha256='ck')))
            st = E.pilot_status(d, 'a1', 0, 'ck')
            self.assertEqual((st[0]['status'], st[0]['same_checkpoint'], st[0]['scale']), ('FAIL', True, 1.3))

    def test_identity_overall(self):
        all_pass = {tp: 'PASS' for tp in E.TOPOLOGIES}
        self.assertEqual(V.overall(all_pass), 'PASS')
        self.assertEqual(V.overall(dict(all_pass, TP8='NOT_OBSERVABLE')), 'PASS_PARTIAL')
        self.assertEqual(V.overall(dict(all_pass, TP8='FAIL')), 'FAIL')
        self.assertEqual(V.overall({tp: 'NOT_OBSERVABLE' for tp in E.TOPOLOGIES}), 'FAIL')


class References(unittest.TestCase):
    def test_all_ten_topologies_from_groups(self):
        entry = dict(baseline_evaluation_manifests=[dict(path='g1'), dict(path='g2'), dict(path='g3')])
        with mock.patch.object(E, 'references', return_value=fake_refs(E.TOPOLOGIES)):
            refs, sel, groups = E.a0_references(entry, 0)
        self.assertEqual(list(refs), E.TOPOLOGIES)
        self.assertEqual(sel, dict(selection_sha256='s1'))
        self.assertEqual(sorted(len(v) for v in groups.values()), [1, 2, 7])
        no89 = [tp for tp in E.TOPOLOGIES if tp not in ('TP8', 'TP9')]
        with mock.patch.object(E, 'references', return_value=fake_refs(no89)):
            with self.assertRaisesRegex(E.StressError, r"TP8', 'TP9"):
                E.a0_references(entry, 0)
        mixed = fake_refs(E.TOPOLOGIES)
        mixed['TP10']['manifest'] = dict(evaluation_protocol=dict(selection=dict(selection_sha256='s2')))
        with mock.patch.object(E, 'references', return_value=mixed):
            with self.assertRaisesRegex(E.StressError, 'different frozen selections'):
                E.a0_references(entry, 0)
        with self.assertRaisesRegex(E.StressError, 'no IEEE69 A0'):
            E.a0_references(dict(), 0)

    def test_ablation_reference(self):
        plans = {tp: [dict(episode_id=e, start_index=96 * e, seed=123 + e) for e in range(31)] for tp in E.TOPOLOGIES}
        protocol = dict(selection=dict(selection_sha256='s', checkpoint_sha256='c'))
        ref = dict(status='finished', evaluation_protocol=dict(data_split='test', final_test=True, selection=dict(selection_sha256='s')),
                   checkpoint_file=dict(sha256='c'), ablation='no_shared_kpi', run_id=1,
                   episode_plans={tp: plans[tp] for tp in E.TOPOLOGIES if tp != 'TP9'})
        checks = E.verify_ablation_reference(ref, protocol, 'no_shared_kpi', 1, plans)
        self.assertEqual((checks['TP1'], checks['TP9']), ('matches', 'not_in_reference'))
        bad = deepcopy(ref)
        bad['episode_plans']['TP3'][0]['seed'] = 999
        with self.assertRaisesRegex(E.StressError, 'TP3'):
            E.verify_ablation_reference(bad, protocol, 'no_shared_kpi', 1, plans)
        for key, value in (('checkpoint_file', dict(sha256='other')), ('run_id', 2), ('ablation', 'no_centralized_critic')):
            with self.assertRaises(E.StressError):
                E.verify_ablation_reference(dict(ref, **{key: value}), protocol, 'no_shared_kpi', 1, plans)


class Summary(unittest.TestCase):
    def write_run(self, root, model, run, scale, cost, identity='PASS'):
        d = E.run_dir(root, model, run, scale)
        d.mkdir(parents=True)
        (d / 'run_manifest.json').write_text(json.dumps(dict(status='finished')))
        starts = [0, 96]
        ep = pd.DataFrame([dict(topology_case=tp, start_index=s, total_energy_cost=cost, adjusted_energy_cost=cost,
                                mean_voltage_deviation=0.01 * scale, feasible_rate=1.0) for tp in ('TP1', 'TP10') for s in starts])
        ep.to_csv(d / 'episode_summary.csv', index=False)
        steps = pd.DataFrame([dict(topology_case=tp, start_index=s, converged=True, correction_iterations=k,
                                   major_infeasible_action=k == 1) for tp in ('TP1', 'TP10') for s in starts for k in (0, 1)])
        steps.to_csv(d / 'step_metrics.csv', index=False)
        pd.DataFrame(dict(topology_case='TP1', load_kw_feature_max=[0.5 * scale] * 4,
                          feeder_grid_import_kw_feature_max=[0.4 * scale] * 4)).to_csv(d / 'obs_diagnostics.csv', index=False)
        if scale == 1.0 and identity:
            (d / 'identity_report.json').write_text(json.dumps(dict(status=identity,
                evaluation_manifest_sha256=sha(d / 'run_manifest.json'))))

    def test_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'root'
            for s in (1.0, 1.1, 1.2, 1.3):
                for r in (0, 1, 2):
                    self.write_run(root, 'a0', r, s, 100 * s + r)
                    self.write_run(root, 'a2', r, s, 80 * s, identity=('PASS_PARTIAL', None, 'FAIL')[r])
            for s in (1.1, 1.3):                     # stress-only controller: no 1.0x runs
                self.write_run(root, 'a1', 0, s, 90 * s)
            S.main(['--root', str(root), '--output', str(Path(d) / 'out'), '--bootstrap', '100'])
            out = Path(d) / 'out'
            inc = json.loads((out / 'inclusion.json').read_text())
            checks = {(i['model'], i['run']): i['identity_check'] for i in inc['included']}
            self.assertEqual((checks['a0', 0], checks['a2', 0], checks['a2', 1], checks['a1', 0]),
                             ('PASS', 'PASS_PARTIAL', 'NOT_PERFORMED', 'NOT_PERFORMED'))
            self.assertNotIn(('a2', 2), checks)
            delta = pd.read_csv(out / 'delta_vs_1x_by_model.csv')
            row = delta.query("model == 'a0' and scale == 1.3 and topology_case == 'TP10' and metric == 'adjusted_energy_cost'").iloc[0]
            self.assertAlmostEqual(row.mean_delta, 30.0, places=6)
            absolute = pd.read_csv(out / 'absolute_by_model.csv')
            self.assertIn('a1', set(absolute.model))
            rates = pd.read_csv(out / 'episode_metrics_long.csv')
            self.assertTrue(np.allclose(rates.mean_correction_iterations, 0.5))
            self.assertTrue(np.allclose(rates.major_infeasible_rate, 0.5))
            self.assertIn(['a1', 0, 1.2], inc['missing_stress_runs'])


if __name__ == '__main__':
    unittest.main()
