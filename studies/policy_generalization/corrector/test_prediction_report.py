"""预测报告验证冻结索引和训练集均值基线；构造缓存不是真实视觉成绩。"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

from .test_training_entry import ROOT, SOURCE, fake_cache, runner
from . import test_training_entry

spec = importlib.util.spec_from_file_location('prediction_report', ROOT / 'scripts/corrector_prediction_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@unittest.skipUnless((SOURCE / '.git').exists(), '需要固定上游审计源码')
class PredictionReportTests(unittest.TestCase):
    def test_batched_cpu_report_matches_single_and_renders_task_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root/'features')
            for directory in (root/'features').glob('episode_*'):
                meta = json.loads((directory/'metadata.json').read_text())
                meta['raw_provenance'] = {'task_id': 20 if meta['split'] == 'train' else 21}
                runner.archive.write_json(directory/'metadata.json', meta)
                runner.archive.write_json(directory/'COMPLETE.json', {'files': {
                    p.name: runner.archive.digest(p) for p in directory.iterdir() if p.name != 'COMPLETE.json'}})
            args = test_training_entry.TrainingEntryTests().args(root, 'run')
            args.stop_after_epoch = 1
            args.output.mkdir(); runner.run(args)
            values = []
            for batch in (1, 3):
                output = root/f'report_{batch}'; output.mkdir()
                inputs = Namespace(run=args.output, features=args.features, source_root=SOURCE,
                                   output=output, raw_root=None, device='cpu', batch_size=batch, loader_workers=0)
                report.predict(inputs)
                values.append(json.loads((output/'summary.json').read_text()))
            for key in values[0]['metrics']:
                self.assertAlmostEqual(values[0]['metrics'][key], values[1]['metrics'][key], places=6)
            self.assertEqual(values[1]['per_task'][0]['task_id'], 21)
            if importlib.util.find_spec('matplotlib'):
                report.plot(inputs)
            else:
                # plot本来就不依赖Torch；允许用本机已有的绘图解释器，不安装或修改环境。
                plot_python = shutil.which('python3')
                if not plot_python or subprocess.run([plot_python, '-c', 'import matplotlib, numpy'],
                                                    capture_output=True).returncode:
                    self.skipTest('CPU批处理指标已检查；本地缺少可用绘图解释器')
                subprocess.run([plot_python, str(ROOT/'scripts/corrector_prediction_report.py'), 'plot',
                                '--output', str(inputs.output)], check=True, capture_output=True)
            self.assertGreater((inputs.output/'per_task_metrics.png').stat().st_size, 0)

    def test_intermediate_snapshot_is_evaluated_and_foreign_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / 'features')
            args = test_training_entry.TrainingEntryTests().args(root, 'run')
            args.snapshot_every, args.stop_after_epoch = 1, 1
            args.output.mkdir(); runner.run(args)
            output = root / 'report'; output.mkdir()
            inputs = Namespace(run=args.output, features=args.features, source_root=SOURCE,
                               output=output, raw_root=None, checkpoint=args.output / 'snapshots/epoch_0000.pt')
            # 无需训练完整结束，也不借用best文件代替选中的初始模型。
            (args.output / 'result.json').unlink()
            report.predict(inputs)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['checkpoint_epoch'], 0)
            self.assertEqual(summary['optimizer_step'], 0)
            inputs.checkpoint = args.output / 'snapshots/epoch_0001.pt'
            with self.assertRaises(ValueError): report.predict(inputs)  # 不混写两份权重报告。
            inputs.output = root / 'other'; inputs.output.mkdir()
            import torch
            foreign = torch.load(inputs.checkpoint, weights_only=True)
            foreign['feature_protocol']['seed'] = 999
            inputs.checkpoint = root / 'foreign.pt'; torch.save(foreign, inputs.checkpoint)
            with self.assertRaises(ValueError): report.predict(inputs)

    def test_report_uses_frozen_validation_and_rejects_changed_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / 'features')
            args = test_training_entry.TrainingEntryTests().args(root, 'run')
            args.stop_after_epoch = 1
            args.output.mkdir(); runner.run(args)
            output = root / 'report'; output.mkdir()
            inputs = Namespace(run=args.output, features=args.features, source_root=SOURCE, output=output, raw_root=None)
            report.predict(inputs)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['sample_count'], 6)
            self.assertEqual(summary['first_pair'], {'episode': 825, 'frame': 0, 'future_frame': 1})
            self.assertEqual(summary['split']['shared_episodes'], [])
            values = np.load(output / 'first_pair.npz')
            self.assertEqual(values['predicted_delta'].shape, (2, 4))
            _, cls, _, _, _ = runner.reference_api(SOURCE)
            groups, _, _ = runner.datasets(args.features, None, cls, 1)
            mean = np.mean([item['delta_z'].numpy() for item in groups['train']], axis=0)
            expected = np.mean([((mean-item['delta_z'].numpy())**2).mean() for item in groups['validation']])
            self.assertAlmostEqual(summary['metrics']['train_mean_mse'], float(expected), places=6)
            frozen = np.load(args.output / 'split_indices.npz')
            np.savez_compressed(args.output / 'split_indices.npz', train=frozen['train'], validation=frozen['validation'][::-1])
            with self.assertRaises(ValueError):
                report.predict(inputs)


class BatchedPredictionReportTests(unittest.TestCase):
    def test_batch_metrics_equal_original_single_sample_definitions(self):
        generator = torch.Generator().manual_seed(42)
        p, t = [torch.randn(5, 3, 7, generator=generator) for _ in range(2)]
        mean = torch.randn(3, 7, generator=generator)
        batched = report.prediction_metrics(p, t, mean)
        for i in range(5):
            expected = {'cosine': F.cosine_similarity(p[i].reshape(1, -1), t[i].reshape(1, -1))[0],
                        'mse': F.mse_loss(p[i], t[i]), 'zero_mse': t[i].square().mean(),
                        'train_mean_mse': F.mse_loss(mean, t[i]),
                        'train_mean_cosine': F.cosine_similarity(mean.reshape(1, -1), t[i].reshape(1, -1))[0],
                        'pred_norm': p[i].norm(), 'target_norm': t[i].norm()}
            for key, value in expected.items():
                torch.testing.assert_close(batched[key][i], value)

    def test_validation_row_handles_subset_crossing_episode_boundary(self):
        first = Namespace(data_path=Path('/a'), ep_idx=np.full(4, 813), pairs=[(0, 2), (1, 3)])
        second = Namespace(data_path=Path('/b'), ep_idx=np.full(5, 825), pairs=[(0, 2), (1, 3), (2, 4)])
        validation = Namespace(dataset=Namespace(datasets=[first, second], cumulative_sizes=[2, 5]), indices=[4, 0, 2])
        arrays = {'/a': np.arange(4)+10, '/b': np.arange(5)+20}
        self.assertEqual(report.validation_row(validation, arrays, 0), {'episode': 825, 'frame': 22, 'future_frame': 24})
        self.assertEqual(report.validation_row(validation, arrays, 1), {'episode': 813, 'frame': 10, 'future_frame': 12})



if __name__ == '__main__':
    unittest.main()
