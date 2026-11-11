"""冻结预测器的输入梯度、批次边界与鲁棒阈值；不等于机器人闭环验证。"""
import unittest

import numpy as np
import torch

from .model import CorrectorConfig, VisualDynamicsMLP
from .online_practice import guided_velocity, robust_limits, visual_mismatch


class OnlinePracticeTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(7)
        model = VisualDynamicsMLP(CorrectorConfig(token_dim=4, action_dim=3,
                                                action_embed_dim=3, widths=(8, 8))).double()
        model.eval().requires_grad_(False)
        return (model, torch.randn(1, 2, 4, dtype=torch.float64),
                torch.randn(1, 5, 6, dtype=torch.float64), torch.randn(1, 5, 6, dtype=torch.float64),
                torch.tensor([.5], dtype=torch.float64), torch.randn(1, 2, 4, dtype=torch.float64))

    def test_mismatch_is_per_sample_and_zero_direction_is_not_failure_label(self):
        a = torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 0.]]])
        b = torch.tensor([[[1., 0.]], [[-1., 0.]], [[0., 0.]]])
        torch.testing.assert_close(visual_mismatch(a, b), torch.tensor([0., 2., 1.]))

    def test_threshold_accepts_array_and_rejects_nonfinite_history(self):
        on, off = robust_limits(np.array([0., .2, .4, .6]), noise_floor=0.)
        self.assertAlmostEqual(on, .9); self.assertAlmostEqual(off, .7)
        self.assertIsInstance(on, float)
        self.assertEqual(robust_limits([.1] * 5), (.25, .25))
        for bad in ([], [float('nan')], [float('inf')], [[.1, .2]]):
            with self.assertRaises(ValueError): robust_limits(bad)

    def test_guidance_matches_finite_difference_without_mutating_inputs_or_weights(self):
        model, z, x, v, t, target = self.inputs()
        before = [p.detach().clone() for p in model.parameters()]
        original = v.clone()
        eta = .01
        with torch.no_grad():
            guided, value = guided_velocity(model, z, x, v, t, target, eta)
            def objective(velocity):
                clean = x - t[:, None, None] * velocity
                return visual_mismatch(model(z, clean[:, 0, :3]), target).mean().item()
            eps = 1e-5
            plus, minus = v.clone(), v.clone()
            plus[0, 0, 0] += eps; minus[0, 0, 0] -= eps
            finite_difference = (objective(plus) - objective(minus)) / (2 * eps)
            measured = ((v - guided) / eta)[0, 0, 0].item()
            self.assertAlmostEqual(measured, finite_difference, places=7)
            self.assertAlmostEqual(value, objective(v))
            self.assertLess(objective(guided), value)
        torch.testing.assert_close(v, original, atol=0, rtol=0)
        torch.testing.assert_close(guided[:, 1:], v[:, 1:], atol=0, rtol=0)
        torch.testing.assert_close(guided[:, 0, 3:], v[:, 0, 3:], atol=0, rtol=0)
        self.assertFalse(guided.requires_grad)
        for p, old in zip(model.parameters(), before):
            self.assertIsNone(p.grad); torch.testing.assert_close(p, old, atol=0, rtol=0)

    def test_zero_time_has_zero_guidance_and_requires_explicit_freezing(self):
        model, z, x, v, t, target = self.inputs()
        guided, _ = guided_velocity(model, z, x, v, t * 0, target)
        torch.testing.assert_close(guided, v, atol=0, rtol=0)
        model.requires_grad_(True)
        with self.assertRaises(ValueError): guided_velocity(model, z, x, v, t, target)
        model.requires_grad_(False)
        with torch.inference_mode(), self.assertRaises(RuntimeError):
            guided_velocity(model, z, x, v, t, target)


if __name__ == '__main__':
    unittest.main()
