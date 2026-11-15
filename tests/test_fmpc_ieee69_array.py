"""Small filesystem fixtures only. No controller, forecast model or solver is run."""
import csv
from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.forecast_mpc import ieee69_array as w


def sample(folder, tp='TP1', count=96):
    folder.mkdir(exist_ok=True)
    day = dict(topology_case=tp, start_index=32064)
    for name in w.TABLES:
        records = [day.copy()]
        if name in ('step_metrics', 'fmpc_solves'):
            records = [dict(day, step=i, horizon_steps=96-i, requested_kw='[0.25, -1]',
                            date_time=f'2018-12-01 {i//4:02}:{15*(i%4):02}:00') for i in range(count)]
        elif name == 'episode_summary':
            records[0].update(intended_episodes=1, completed_episodes=int(count == 96),
                              incomplete_episodes=int(count != 96), failed_episodes=int(count != 96),
                              attempted_evaluation_steps=count, raw_energy_cost=15)
        elif name == 'fmpc_days':
            records[0].update(solves=count, solve_time_s=100)
        with (folder / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, records[0])
            writer.writeheader()
            writer.writerows(records)


class Workflow(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / 'new'
        self.ref = self.root / 'reference'
        sample(self.data)
        sample(self.ref)

    def test_day_comparison_preserves_metrics_excludes_timing(self):
        path = self.ref / 'fmpc_days.csv'
        path.write_text(path.read_text().replace(',100', ',999'))
        w.validate_tables(self.data, 'TP1', [32064])
        result = w.compare_day(self.data, self.ref, 32064)
        self.assertEqual(result['tables']['fmpc_solves']['matched_rows'], 96)
        path = self.data / 'fmpc_solves.csv'
        path.write_text(path.read_text().replace('0.25', '0.26', 1))
        with self.assertRaisesRegex(ValueError, 'requested_kw'):
            w.compare_day(self.data, self.ref, 32064)

    def test_only_reviewed_generator_revision_is_accepted_and_reported(self):
        path = 'evaluation/forecast_mpc/generate.py'
        solver = 'evaluation/oracle/solver.py'
        old = 'ffd4af7cfe0df51969b04b2e2ee8344101d8a797cd1e3dfc8401cda352c1e61d'
        new = '0003bcd06e99876bd6d55560c504f804f41473d5673b706c52add8ae9109f799'
        reference = {path: old, solver: 'a' * 64}
        current = {**reference, path: new}
        self.assertEqual(w.source_exceptions(reference, reference), [])
        exceptions = w.source_exceptions(reference, current)
        report = w.compare_day(self.data, self.ref, 32064, exceptions)
        entry, = report['source_exceptions']
        self.assertEqual((entry['path'], entry['reference_sha256'], entry['current_sha256']), (path, old, new))
        for base, candidate in (
                (reference, {**current, solver: 'b' * 64}),
                (reference, {**current, path: '0' * 64}),
                ({**reference, path: '0' * 64}, current),
                (current, reference),
                (reference, {solver: reference[solver]}),
                (reference, {**current, 'unexpected.py': 'c' * 64})):
            with self.subTest(base=base, candidate=candidate), self.assertRaises(ValueError):
                w.source_exceptions(base, candidate)

    def test_partial_and_duplicate(self):
        sample(self.data, count=40)
        w.validate_tables(self.data, 'TP1', [32064])
        with self.assertRaisesRegex(ValueError, 'completed'):
            w.compare_day(self.data, self.ref, 32064)
        p = self.data / 'step_metrics.csv'
        p.write_text(p.read_text() + p.read_text().splitlines()[-1] + '\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            w.validate_tables(self.data, 'TP1', [32064])

    def test_missing_solve_or_wrong_topology(self):
        with self.assertRaisesRegex(ValueError, 'topology'):
            w.validate_tables(self.data, 'TP6', [32064])
        path = self.data / 'fmpc_solves.csv'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, 'missing'):
            w.validate_tables(self.data, 'TP1', [32064])

    def test_publish_reuse_and_tampering(self):
        attempt = self.root / '.TP1-test'
        attempt.mkdir()
        records = attempt / 'result'
        self.data.rename(records)
        self.assertIsNone(w.checked(self.root, 'TP1', {'scale': 130}))
        w.publish(self.root, 'TP1', attempt, records, {'scale': 130})
        self.assertEqual(w.checked(self.root, 'TP1', {'scale': 130}), self.root / 'TP1' / 'tables')
        with self.assertRaisesRegex(ValueError, 'refusing to replace'):
            w.publish(self.root, 'TP1', attempt, records, {'scale': 130})
        with self.assertRaisesRegex(ValueError, 'stale'):
            w.checked(self.root, 'TP1', {'scale': 100})
        (self.root / 'TP1/tables/fmpc_days.csv').write_text('invalid')
        with self.assertRaisesRegex(ValueError, 'changed'):
            w.checked(self.root, 'TP1', {'scale': 130})

    def test_no_partial_aggregate(self):
        with patch.object(w, 'ROOT', self.root), patch.object(w, 'identity', return_value={}):
            with self.assertRaisesRegex(ValueError, 'missing completed TP1'):
                w.aggregate(130, {}, {})
        self.assertFalse((self.root / 'load_x130' / 'aggregate').exists())

    def test_aggregate_combines_all_six_tables_and_resumes(self):
        parent = self.root / 'load_x130'
        parent.mkdir()
        for tp in w.TPS:
            attempt = parent / ('.' + tp)
            attempt.mkdir()
            data = attempt / 'result'
            sample(data, tp)
            w.write(data / 'run_manifest.json', {})
            w.publish(parent, tp, attempt, data, {'tp': tp})
        # One synthetic day per topology. Production validates the real 31-day manifests.
        with patch.object(w, 'ROOT', self.root), patch.object(w, 'locked', return_value=nullcontext()), \
                patch.object(w, 'identity', side_effect=lambda pct, tp, *_: {'tp': tp}), \
                patch.object(w, 'check_manifest', return_value=[32064]):
            w.aggregate(130, {}, {})
            w.aggregate(130, {}, {})
        for name in w.TABLES:
            records = list(w.rows(parent / 'aggregate' / 'tables', name))
            self.assertEqual({r['topology_case'] for r in records}, set(w.TPS))
            self.assertEqual(len(records), 960 if name in ('step_metrics', 'fmpc_solves') else 10)


if __name__ == '__main__':
    unittest.main()
