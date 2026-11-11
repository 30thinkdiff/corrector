"""离线事件筛选的边界检查；不是策略性能测试。"""
import unittest

import torch

from studies.policy_generalization.review_grasp_events import summarize_events


def trace(heights, contacts):
    n = len(heights)
    return {"elapsed_s": torch.arange(n, dtype=torch.float64) * .05,
            "object_xyz": torch.stack([torch.zeros(n), torch.zeros(n), torch.tensor(heights)], dim=1),
            "both_fingers_same_physics_sample": torch.tensor(contacts, dtype=torch.bool),
            "gripper_xyz": torch.zeros(n, 3),
            "gripper_rotation_world": torch.eye(3).repeat(n, 1, 1)}


class ReviewEventsTests(unittest.TestCase):
    def test_success_outside_capture_is_not_reported_as_failure(self):
        result = summarize_events(trace([0., 0., 0., 0.], [False] * 4), True)
        self.assertIsNone(result['first_bilateral_interval_end_s'])
        self.assertIn('不足以解释', result['candidate_for_review'])
        self.assertTrue(result['final_success'])

    def test_drop_is_only_a_candidate_even_for_successful_placement(self):
        result = summarize_events(trace([0., .12, .15, .04], [False, True, True, False]), True)
        self.assertAlmostEqual(result['first_height_decrease_5cm_after_lift_s'], .15)
        self.assertTrue(result['final_success'])
        self.assertIn('正常放置', result['limits'])

    def test_transient_bilateral_contact_is_not_a_grasp_label(self):
        result = summarize_events(trace([0.] * 5, [False, True, False, False, False]), False)
        self.assertAlmostEqual(result['first_bilateral_interval_end_s'], .05)
        self.assertAlmostEqual(result['bilateral_absent_3_intervals_start_s'], .1)
        self.assertIsNone(result['first_rise_10cm_s'])

    def test_invalid_or_nonmonotonic_trace_is_rejected(self):
        t = trace([0., .1], [False, True])
        t['elapsed_s'][1] = 0
        with self.assertRaises(ValueError):
            summarize_events(t, False)


if __name__ == '__main__':
    unittest.main()
