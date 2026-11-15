"""Synthetic reporting checks only; no policy, environment or historical writes.

Run: python -m unittest experiments.load_stress.test_summary_reporting -v
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType

import numpy as np
import pandas as pd

from experiments.load_stress import summarize_load_stress as S


class Reporting(unittest.TestCase):
    def test_nullable_boolean(self):
        x = S.as_bool(pd.Series([True, False, 'true', 'False', 1, 0, np.nan, None, 'bad']))
        self.assertEqual(x.iloc[:6].tolist(), [True, False, True, False, True, False])
        self.assertEqual(x.isna().sum(), 3)
        self.assertEqual(float(x.mean()), .5)
        self.assertEqual(float((~x).mean()), .5)

    def test_all_unknown_boolean(self):
        self.assertTrue(pd.isna(S.as_bool(pd.Series([None, np.nan, 'bad'])).mean()))

    def test_valid_power_unchanged(self):
        r, p = [10, -20, 0, 4, -1], [5, -10, 0, 3, -1]
        counts = S.power_observations(json.dumps(r), json.dumps(p))
        rates = S.power_percentages(counts)
        self.assertEqual(counts['power_valid_decisions'], 5)
        self.assertAlmostEqual(rates['requested_power_removed_pct'], 100*16/35)
        self.assertEqual(rates['battery_correction_gt1kw_pct'], 40)

    def test_missing_invalid_pairs(self):
        counts = S.power_observations('[10, null, NaN, 4, 8]', '[5, 0, 0, Infinity, 8]')
        self.assertEqual([counts[k] for k in ('power_valid_decisions', 'power_missing_decisions', 'power_invalid_decisions')], [2, 1, 2])
        self.assertEqual(S.power_percentages(counts)['battery_correction_gt1kw_pct'], 50)

    def test_absent_power(self):
        c = S.power_observations(None, None)
        self.assertEqual(c['power_missing_decisions'], 5)
        self.assertTrue(all(np.isnan(x) for x in S.power_percentages(c).values()))

    def test_zero_denominator(self):
        rates = S.power_percentages(S.power_observations([0]*5, [0]*5))
        self.assertTrue(np.isnan(rates['requested_power_removed_pct']))
        self.assertEqual(rates['battery_correction_gt1kw_pct'], 0)

    def test_malformed_wrong_shape(self):
        for value in ('bad JSON', '[1, 2]', '{}'):
            self.assertEqual(S.power_observations(value, [0]*5)['power_invalid_decisions'], 5)

    def fixture(self, root, missing=False):
        ep = pd.DataFrame(dict(topology_case=['TP1', 'TP1'], start_index=[0, 96], total_energy_cost=[12., 7.],
            adjusted_energy_cost=[13., 8.], completed_episodes=[1, 0], failed_episodes=[0, 1],
            incomplete_episodes=[0, 1], feasible_rate=[1., 1.], converged_rate=[1., 1.]))
        ep.to_csv(root/'episode_summary.csv', index=False)
        rows = pd.DataFrame(dict(topology_case=['TP1']*4, start_index=[0, 0, 0, 96],
            converged=[True, False, None, False], feasible=[True, False, None, False],
            backup_activated=[True, False, None, False], marl_requested_power_kw=['[10,0,0,0,0]']*4,
            physical_limited_power_kw=['[5,0,0,0,0]']*4))
        if missing:
            rows = rows.drop(columns=['marl_requested_power_kw', 'physical_limited_power_kw'])
        rows.to_csv(root/'step_metrics.csv', index=False)
        (root/'run_manifest.json').write_text(json.dumps({'episode_plans': {'TP1': [{'start_index': x} for x in [0, 96, 192]]}}))
        return dict(dir=root, model='a0', run=0, scale=1.3, identity_check='PASS')

    def test_chunking_coverage_and_missing_day(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            item = self.fixture(Path(d))
            one = S.episode_table(item, chunksize=1)
            pd.testing.assert_frame_equal(one, S.episode_table(item, chunksize=100))
            self.assertEqual(one.loc[0, 'feasible_rate'], .5)
            self.assertEqual(one.loc[0, 'pf_failure_rate'], .5)
            self.assertEqual(one.loc[0, 'feasible_rate_unknown_steps'], 1)
            self.assertEqual(one.loc[0, 'total_energy_cost'], 12)
            self.assertTrue(one.loc[1:, 'total_energy_cost'].isna().all())
            coverage = S.coverage_table(one, ['model', 'run', 'scale', 'topology_case']).iloc[0]
            self.assertEqual(coverage.intended_episodes, 3)
            self.assertEqual(coverage.completed_episodes, 1)
            self.assertEqual(coverage.failed_episodes, 1)
            self.assertEqual(coverage.failed_episodes_unknown, 1)
            self.assertEqual(coverage.incomplete_episodes, 2)
            self.assertEqual(coverage.power_valid_decisions, 20)
            self.assertEqual(coverage.requested_power_removed_pct, 50)

    def test_old_records_without_power(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            table = S.episode_table(self.fixture(Path(d), missing=True))
            self.assertTrue(table.requested_power_removed_pct.isna().all())
            self.assertEqual(table.power_missing_decisions.sum(), 20)

    def test_comparison_has_both_coverage_and_paired_days(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            table = S.episode_table(self.fixture(Path(d)))
            nominal = table.assign(scale=1.)
            full = pd.concat([table, nominal], ignore_index=True)
            delta = S.paired(full, ['total_energy_cost'], 'scale == 1.3', 'scale == 1.', ['model', 'run']).assign(scale=1.3)
            summary = S.model_level(delta, ['total_energy_cost'], np.random.default_rng(42), 10, extra=('scale',))
            out = S.with_coverage(summary, full, ['model', 'scale', 'topology_case'], 'nominal').iloc[0]
            self.assertEqual(out.n_days, 1)
            self.assertEqual(out.intended_episodes, 3)
            self.assertEqual(out.reference_intended_episodes, 3)
            self.assertEqual(out.mean_delta, 0)

    def test_zero_reference_not_repeated_per_seed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            base = S.episode_table(self.fixture(Path(d)))
            full = pd.concat([base.assign(run=r) for r in range(3)] + [base.assign(model='zero')])
            summary = pd.DataFrame([dict(model='a0', scale=1.3, topology_case='TP1')])
            out = S.with_coverage(summary, full, ['model', 'scale', 'topology_case'], 'zero').iloc[0]
            self.assertEqual(out.intended_episodes, 9)
            self.assertEqual(out.reference_intended_episodes, 3)

    def test_original_topology_summary_paths(self):
        root = Path(__file__).resolve().parents[2]
        for name in ('run_tp_eval.py', 'run_tp_eval_ieee69.py'):
            tree = ast.parse((root/'experiments/topology_generalization'/name).read_text())
            functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'augment_summaries' for n in ast.walk(functions['summarize'])))
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'finish_episode' for n in ast.walk(functions['evaluate_one_episode'])))

    def test_existing_bootstrap_results_unchanged(self):
        frame = pd.DataFrame(dict(model=['a0']*4, run=[0, 0, 1, 1], topology_case=['TP1']*4,
                                  total_energy_cost=[1., 2., 3., 5.], requested_power_removed_pct=[2., 3., 4., 5.]))
        plain = S.summarize_deltas(frame, ['total_energy_cost'], ['run'], np.random.default_rng(3), 50)
        extra = S.summarize_deltas(frame, ['total_energy_cost', *S.POWER_METRICS[:1]], ['run'], np.random.default_rng(3), 50)
        pd.testing.assert_frame_equal(plain, extra[extra.metric == 'total_energy_cost'].reset_index(drop=True))

    def test_original_augmentation_retains_failed_economics_coverage(self):
        from utility.evaluation_reporting import augment_summaries
        steps = pd.DataFrame(dict(topology_case=['TP1', 'TP1'], episode_id=[0, 1],
            episode_completed=[True, False], episode_failed=[False, True],
            episode_incomplete=[False, True], episode_successful_steps=[96, 40],
            episode_attempted_steps=[96, 41]))
        ep = pd.DataFrame(dict(topology_case=['TP1', 'TP1'], episode_id=[0, 1], total_energy_cost=[12., 3.]))
        agg = pd.DataFrame(dict(topology_case=['TP1'], mean_total_energy_cost=[7.5]))
        ep, agg = augment_summaries(steps, ep, agg)
        self.assertEqual(ep.loc[0, 'total_energy_cost'], 12)
        self.assertTrue(np.isnan(ep.loc[1, 'total_energy_cost']))
        self.assertEqual(agg.loc[0, 'mean_total_energy_cost'], 12)
        self.assertEqual(agg.loc[0, 'economic_valid_episodes'], 1)
        self.assertEqual(agg.loc[0, 'economic_excluded_episodes'], 1)
        self.assertEqual(agg.loc[0, 'intended_episodes'], 2)
        self.assertEqual(agg.loc[0, 'failed_episodes'], 1)

    def test_legacy_economics_kept_and_flagged(self):
        # Records predating coverage columns keep their recorded totals (historical behaviour).
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            item = self.fixture(Path(d))
            ep = pd.read_csv(Path(d)/'episode_summary.csv')
            ep.drop(columns=['completed_episodes', 'failed_episodes', 'incomplete_episodes']).to_csv(Path(d)/'episode_summary.csv', index=False)
            (Path(d)/'run_manifest.json').write_text('{}')
            table = S.episode_table(item)
            self.assertEqual(table.total_energy_cost.tolist(), [12., 7.])
            self.assertEqual(set(table.economic_totals_status), {'unverified_legacy'})
            self.assertTrue(table.completed_episodes.isna().all())

    def test_known_partial_day_masked_and_flagged(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            table = S.episode_table(self.fixture(Path(d)))
            self.assertEqual(table.economic_totals_status.tolist(), ['verified', 'masked_incomplete_or_failed', 'masked_incomplete_or_failed'])

    def ieee69_steps(self, root, flags):
        pd.DataFrame(dict(topology_case='TP1', start_index=0, total_energy_cost=5., feasible_rate=1.), index=[0]) \
            .to_csv(root/'episode_summary.csv', index=False)
        pd.DataFrame(dict(topology_case='TP1', start_index=0, converged=flags, backup_activated=flags,
                          episode_failed=[False]*len(flags))).to_csv(root/'step_metrics.csv', index=False)
        return dict(dir=root, model='a0', run=0, scale=1.3, identity_check='PASS')

    def test_ieee69_unknown_flags_excluded_not_false(self):
        from experiments.load_stress import summarize_load_stress_ieee69 as S69
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            row = S69.episode_table(self.ieee69_steps(Path(d), [True, False, None, None])).iloc[0]
            self.assertEqual(row.pf_failure_rate, .5)          # previously 0.75: unknown counted as failure
            self.assertEqual(row.backup_activation_rate, .5)   # previously 0.25: unknown counted as False
            self.assertIsInstance(row.pf_failure_rate, float)   # plain float, not pandas Float64/<NA>
            all_unknown = S69.episode_table(self.ieee69_steps(Path(d), [None, None])).iloc[0]
            self.assertTrue(np.isnan(all_unknown.backup_activation_rate))

    def test_ieee69_fully_known_rates_unchanged(self):
        from experiments.load_stress import summarize_load_stress_ieee69 as S69
        old_bool = lambda s: s.map(lambda v: str(v).strip().lower() in ('true', '1', '1.0'))
        flags = [True, False, False, True, True]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            row = S69.episode_table(self.ieee69_steps(Path(d), flags)).iloc[0]
        s = pd.Series(flags)
        self.assertEqual(row.pf_failure_rate, float((~old_bool(s)).mean()))
        self.assertEqual(row.backup_activation_rate, float(old_bool(s).mean()))

    def test_pure_zero_validator_unknown_vs_known_is_mismatch(self):
        from experiments.load_stress import validate_pure_zero_ieee33 as V
        keys = dict(topology_case=['TP1']*2, start_index=[0, 0], step=[0, 1])
        pure = pd.DataFrame(dict(keys, feasible=[None, True]))
        ref = pd.DataFrame(dict(keys, feasible=[True, True], backup_activated=[False, False]))
        pure['backup_activated'] = [False, False]
        with patch.object(V, 'KEYS', ['topology_case', 'start_index', 'step']), \
                patch.object(V, 'NUMERIC', []), patch.object(V, 'FLAGS', ['feasible']):
            rows, _ = V.compare(pure, ref, 1e-9)
        self.assertEqual(rows.iloc[0].status, 'FAIL')
        self.assertEqual(rows.iloc[0].n_mismatch, 1)

    def test_summary_cli_writes_coverage(self):
        # Replace only runtime-heavy imports and directory reservation for this
        # synthetic reporting test. No environment or actor is instantiated.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            root = Path(d)
            runs = []
            for scale in (1., 1.3):
                source = root/str(scale)
                source.mkdir()
                runs.append(dict(self.fixture(source), scale=scale))
            evaluator = ModuleType('experiments.load_stress.evaluate_load_stress_ieee33')
            evaluator.MODELS, evaluator.TOPOLOGIES, evaluator.OBS_FEATURES = {'a0': ('a0_original', None)}, ['TP1'], ()
            stress = ModuleType('experiments.load_stress.stress_env')
            stress.ALLOWED_SCALES = (1., 1.3)
            manifest = ModuleType('utility.run_manifest')
            def reserve(path):
                path.mkdir(exist_ok=False)
                return path
            manifest.reserve_directory = reserve
            modules = {m.__name__: m for m in (evaluator, stress, manifest)}
            with patch.dict('sys.modules', modules), patch.object(S, 'discover', return_value=(runs, [])):
                S.main(['--root', str(root), '--output', str(root/'summary'), '--bootstrap', '10'])
            result = pd.read_csv(root/'summary/delta_vs_1x_by_model.csv')
            self.assertTrue((result.intended_episodes == 3).all())
            self.assertTrue((result.reference_incomplete_episodes == 2).all())
            cost = result[result.metric == 'total_energy_cost'].iloc[0]
            self.assertEqual(cost.n_days, 1)
            self.assertEqual(cost.mean_delta, 0)


if __name__ == '__main__':
    unittest.main()
