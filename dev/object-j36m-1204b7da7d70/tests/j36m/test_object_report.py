import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from corrector_j36m_object_report import analyze, coverage


def record(method, success=None, status='completed', seed=1000):
    return {'case': {'method': method, 'condition': 'libero_object/task_1/init_0', 'seed': seed},
            'status': status, 'result': {'success': success}}


class ReportTests(unittest.TestCase):
    def test_coverage_counts_actual_steps_not_skip_cancel_totals(self):
        rows = [{'step': s, 'score': .1 if s in (20, 23) else None} for s in range(24)]
        rows += [{'step': 20, 'event': 'ogg', 'denoise': 1}, {'step': 21, 'cancelled_predictions': 20}]
        result = coverage(rows, 24)
        self.assertEqual(result['fraction'], .5)
        self.assertEqual(result['longest_gap_steps'], 2)
        self.assertIsNone(coverage([], 10)['fraction'])

    def test_pending_and_error_are_not_robot_failure_or_matched_pairs(self):
        report = analyze([record('H50', False), record('j_lvm', status='error'),
                          record('H20', status='running')])
        self.assertEqual(report['methods']['j_lvm']['errors'], 1)
        self.assertIsNone(report['methods']['j_lvm']['success_rate'])
        self.assertEqual(report['paired']['H50_vs_j_lvm']['matched'], 0)

    def test_strict_seed_matching_gain_and_regression_counts(self):
        report = analyze([record('j_lvm', False), record('j_lvm_ogg', True),
                          record('j_lvm', True, seed=1001), record('j_lvm_ogg', False, seed=1001),
                          record('j_lvm_ogg', True, seed=1002)])
        pair = report['paired']['j_lvm_vs_j_lvm_ogg']
        self.assertEqual(pair['matched'], 2)
        self.assertEqual(pair['method_only_success'], 1)
        self.assertEqual(pair['baseline_only_success'], 1)
        with self.assertRaises(ValueError): analyze([record('H50', True), record('H50', True)])


if __name__ == '__main__': unittest.main()
