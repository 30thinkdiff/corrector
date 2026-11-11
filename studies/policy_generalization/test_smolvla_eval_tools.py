"""CPU验证评测接入，指标答案留给用户；不伪装真实GPU验证。"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eval_tools", ROOT / "scripts/smolvla_eval_tools.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
spec = importlib.util.spec_from_file_location("workbench", ROOT / "scripts/smolvla_training_workbench.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)


class EvaluationToolsTests(unittest.TestCase):
    def test_checker_accepts_expected_values_and_rejections(self):
        function = Mock(side_effect=[4., 2., 3., ValueError(), ValueError(), TypeError()])
        ev.check_metric(function)
        self.assertEqual(function.call_count, 6)

    def test_checker_rejects_tuple_or_wrong_result(self):
        for bad in [(4., {}), 99.]:
            with self.assertRaises(AssertionError):
                ev.check_metric(Mock(return_value=bad))

    def test_checker_rejects_silent_all_padding(self):
        with self.assertRaisesRegex(AssertionError, 'padding'):
            ev.check_metric(Mock(side_effect=[4., 2., 3., 0.]))

    def test_pair_uses_same_noise_without_mutating_inputs(self):
        import torch
        from contextlib import contextmanager

        class Policy:
            disabled = False
            def reset(self):
                pass
            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False
            def predict_action_chunk(self, batch, noise):
                result = batch['x'] + noise + (0 if self.disabled else 2)
                batch['x'].zero_()
                noise.zero_()
                return result

        policy = Policy()
        batch, noise = {'x': torch.ones(1, 2, 2)}, torch.ones(1, 2, 2)
        base, trained = ev.predict_pair(policy, batch, noise)
        torch.testing.assert_close(base, torch.full_like(base, 2))
        torch.testing.assert_close(trained, torch.full_like(trained, 4))
        self.assertTrue(batch['x'].eq(1).all())
        self.assertTrue(noise.eq(1).all())
        self.assertFalse(policy.disabled)

    def test_score_runs_on_cpu_and_preserves_predictions(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'paired_predictions.pt'
            torch.save({'base': torch.ones(1, 2, 2), 'trained': torch.full((1, 2, 2), 2.),
                        'target': torch.zeros(1, 2, 2), 'is_pad': torch.zeros(1, 2, dtype=torch.bool)}, data)
            before = wb.digest(data)
            wb.write_json(root / 'evaluation_manifest.json', {'complete': True, 'samples': [{'frame': 0}], 'prediction_sha256': before})
            metric_file = root / 'metric.py'
            metric_file.write_text('# fixture; user implementation is not changed\n')
            args = SimpleNamespace(metric_file=metric_file, predictions=data)
            # 接入测试只用全有效的常数数据，不包含用户masked MAE的实现答案。
            with patch.object(ev, 'check_metric'), patch.object(ev, 'load_metric', return_value=lambda p, t, m: float(p[0, 0, 0])), patch.object(torch.cuda, 'is_available', side_effect=AssertionError('must stay on CPU')):
                ev.score_entry(args, SimpleNamespace(say=lambda x: None), wb)
            scored = json.loads(next(root.glob('score_*.json')).read_text())
            self.assertEqual(scored['base_mae'], 1.)
            self.assertEqual(scored['trained_mae'], 2.)
            self.assertEqual(scored['trained_minus_base_mae'], 1.)
            self.assertEqual(wb.digest(data), before)

    def test_partial_export_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wb.write_json(root / 'evaluation_manifest.json', {'complete': False})
            args = SimpleNamespace(metric_file=Path(__file__), predictions=root / 'paired_predictions.pt')
            with patch.object(ev, 'check_metric'), patch.object(ev, 'load_metric', return_value=Mock()):
                with self.assertRaisesRegex(RuntimeError, '未完成'):
                    ev.score_entry(args, SimpleNamespace(say=lambda x: None), wb)


if __name__ == '__main__':
    unittest.main()
