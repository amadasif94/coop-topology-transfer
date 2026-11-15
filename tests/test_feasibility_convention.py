import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utility.feasibility import feasible, VOLTAGE_TOLERANCE_PU, CURRENT_RELATIVE_TOLERANCE
from environments.directional_backup import violation_metrics
from test_directional_backup import pf, run_rule


class Convention(unittest.TestCase):
    def test_voltage_boundaries_and_raw_excess(self):
        for bound, sign in ((.95, -1), (1.05, 1)):
            for fraction, expected in ((0, True), (.99, True), (1.01, False)):
                v = bound + sign*fraction*VOLTAGE_TOLERANCE_PU
                metrics = violation_metrics(pf([v]), [1])
                self.assertEqual(metrics['feasible'], expected)
                self.assertEqual(feasible([v], [.5], [1], True), expected)
                self.assertAlmostEqual(metrics['raw_max_voltage_violation'], fraction*VOLTAGE_TOLERANCE_PU)

    def test_current_tolerance_is_relative_to_each_rating(self):
        for rating in (1., 5., 255.):
            for fraction, expected in ((0, True), (.99, True), (1.01, False)):
                current = rating*(1+fraction*CURRENT_RELATIVE_TOLERANCE)
                metrics = violation_metrics(pf([1], [current]), [rating])
                self.assertEqual(metrics['feasible'], expected)
                self.assertEqual(feasible([1], [current], [rating], True), expected)
                self.assertAlmostEqual(metrics['raw_max_current_violation'], current-rating)

    def test_lexicographic_aggregate_then_adjustment(self):
        # Same limiting current for EVERY candidate. Positive power removes an
        # additional voltage violation, reducing the secondary mean only.
        fn = lambda p: pf([1.0 if p > 0 else .94], [1.1])
        best, diag, _ = run_rule(fn)
        self.assertEqual(diag['backup_initial_score'], diag['backup_final_score'])
        initial = violation_metrics(fn(0), [1])
        final = violation_metrics(best['pf_result'], [1])
        self.assertLess(final['secondary'], initial['secondary'])
        self.assertEqual(best['dess_power_kw'][0], 18.75)
        self.assertFalse(best['feasible'])


if __name__ == '__main__':
    unittest.main()
