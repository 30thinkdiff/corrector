"""恢复后处理必须复用冻结权重/代码，且不调用训练。"""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from .test_training_queue import queue

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('finish_queue', ROOT/'scripts/corrector_finish_queue.py')
finish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finish)


class FinishQueueTests(unittest.TestCase):
    def fixture(self, root):
        code = root/'code'; code.mkdir()
        (code/'fixed.py').write_text('# synthetic frozen source')
        output = root/'queue'
        args = queue.parser().parse_args(['run', '--output', str(output), '--archive', str(root/'archive'),
                                         '--runtime', str(root/'runtime')])
        queue.write(args.archive/'manifest.json', {'synthetic': True})
        frozen = {'output': str(output), 'archive': str(args.archive), 'runtime': str(args.runtime),
                  'effective_loader_workers': 0, 'code_sha256': {'fixed.py': queue.digest(code/'fixed.py')},
                  'manifest_sha256': queue.digest(args.archive/'manifest.json')}
        queue.write(output/'queue_config.json', frozen)
        result = {'checkpoint_reload_verified': True, 'epochs_completed': 5, 'optimizer_steps': 490,
                  'best_validation': {'loss': 1.2}}
        queue.write(output/'predictor/result.json', result)
        queue.write(output/'predictor/training_status.json', {'status': 'completed'})
        (output/'predictor/best_model.pt').write_bytes(b'fixture, not trained weights')
        sha = queue.digest(output/'predictor/best_model.pt')
        report = output/'reports'/sha[:16]
        queue.write(report/'summary.json', {'checkpoint_sha256': sha, 'sample_count': 10})
        (output/'logs').mkdir()
        return code, output, args, frozen, report

    def test_refuses_zero_training_and_changed_code_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, output, args, frozen, report = self.fixture(Path(tmp))
            finish.validate(queue, code, args, frozen)
            result = queue.read(output/'predictor/result.json')
            result['optimizer_steps'] = 0
            queue.write(output/'predictor/result.json', result)
            with self.assertRaises(RuntimeError):
                finish.validate(queue, code, args, frozen)
            result['optimizer_steps'] = 490
            queue.write(output/'predictor/result.json', result)
            (code/'fixed.py').write_text('changed')
            with self.assertRaises(RuntimeError):
                finish.validate(queue, code, args, frozen)
            (code/'fixed.py').write_text('# synthetic frozen source')
            (output/'predictor/best_model.pt').write_bytes(b'changed checkpoint')
            with self.assertRaises(FileNotFoundError):
                finish.validate(queue, code, args, frozen)

    def test_resume_calls_plot_smoke_paired_and_never_fit_or_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, output, args, frozen, report = self.fixture(Path(tmp))
            calls = []
            def child(runner, cmd, label, **kwargs):
                calls.append((label, cmd))
                if label in {'smoke', 'paired'}:
                    queue.write(Path(cmd[cmd.index('--output')+1])/'execution.json', {'status': 'completed', 'errors': 0})
            inputs = argparse.Namespace(code_dir=code, output=output, eval_hours=2, plots_only=False)
            with patch.object(finish, 'original_queue', return_value=queue), \
                 patch.object(queue.Queue, 'child', new=child), patch.object(queue.Queue, 'wait_gpu', return_value=True):
                finish.finish(inputs)
            self.assertEqual([label for label, cmd in calls], ['plots', 'smoke', 'paired'])
            self.assertEqual(Path(calls[0][1][0]), args.runtime/'sim-venv/bin/python')
            self.assertEqual(queue.read(output/'queue_status.json')['status'], 'completed')
            self.assertEqual(queue.read(output/'queue_config.json'), frozen)

    def test_inspect_omits_episode_lists_and_never_loads_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, output, args, frozen, report = self.fixture(Path(tmp))
            result = queue.read(output/'predictor/result.json')
            result['split'] = {'samples': {'train': 100, 'validation': 10}, 'episodes': list(range(1000))}
            queue.write(output/'predictor/result.json', result)
            (output/'logs/plots.log').write_text("ModuleNotFoundError: No module named 'matplotlib'\n")
            with redirect_stdout(io.StringIO()) as text:
                finish.inspect_run(output)
            compact = json.loads(text.getvalue())
            self.assertEqual(compact['training_result']['optimizer_steps'], 490)
            self.assertNotIn('episodes', text.getvalue())
            self.assertIn('matplotlib', compact['plots_log_tail'][0])


if __name__ == '__main__':
    unittest.main()
