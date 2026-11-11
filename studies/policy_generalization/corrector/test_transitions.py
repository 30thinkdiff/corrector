"""防止时序监督跨轨迹、漏帧和不合法k；保留用户的配对循环。"""
import unittest

from studies.policy_generalization.corrector.transitions import make_transition_pairs


class TransitionTests(unittest.TestCase):
    def test_boundary_and_k(self):
        ep, frame = [1, 1, 1, 2, 2], [0, 1, 2, 0, 1]
        self.assertEqual(make_transition_pairs(ep, frame), [(0, 1), (1, 2), (3, 4)])
        self.assertEqual(make_transition_pairs(ep, frame, 2), [(0, 2)])
        self.assertEqual(make_transition_pairs(ep, frame, 10), [])

    def test_empty_and_length(self):
        self.assertEqual(make_transition_pairs([], []), [])
        with self.assertRaises(ValueError):
            make_transition_pairs([1], [])

    def test_bad_k(self):
        for k in (True, False, 1.0, 1.5, "1", None):
            with self.subTest(k=k), self.assertRaises(TypeError):
                make_transition_pairs([1, 1], [0, 1], k)
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                make_transition_pairs([1, 1], [0, 1], k)

    def test_gap_and_duplicate(self):
        for frames in ([0, 2, 3], [0, 0, 1], [2, 1, 0]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                make_transition_pairs([1] * 3, frames)

    def test_disjoint_episode_blocks(self):
        with self.assertRaises(ValueError):
            make_transition_pairs([1, 2, 1], [0, 0, 1])
