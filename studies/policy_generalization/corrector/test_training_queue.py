"""队列编排契约；模拟子进程不代表真实4090/仿真结果。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('training_queue', ROOT/'scripts/corrector_training_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
MANIFEST = ROOT/'studies/policy_generalization/corrector/manifests/libero_object10_8695891.json'


class QueueTests(unittest.TestCase):
    def args(self, root):
        return queue.parser().parse_args(['run', '--archive', str(root/'archive'), '--workdir', str(root/'work'),
            '--runtime', str(root/'runtime'), '--source-root', str(root/'source'), '--output', str(root/'run')])

    def test_partial_features_never_count_as_published_and_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'records': [{'episode': 1, 'frames': 10, 'split': 'train'}]}
            directory = root/'features/episode_1'
            directory.mkdir(parents=True)
            queue.write(directory/'metadata.json', {'num_frames': 10, 'split': 'train'})
            self.assertEqual(queue.manifest_progress(root, manifest), (0, 1))
            queue.write(directory/'COMPLETE.json', {'files': {}})
            self.assertEqual(queue.manifest_progress(root, manifest), (1, 1))
            queue.write(directory/'metadata.json', {'num_frames': 9, 'split': 'train'})
            with self.assertRaises(ValueError):
                queue.manifest_progress(root, manifest)

    def test_existing_writer_lock_detected_without_stopping_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with queue.lock(root/'.feature-writer.lock'):
                self.assertTrue(queue.writer_busy(root))
            self.assertFalse(queue.writer_busy(root))

    def test_small_container_shared_memory_uses_single_process_loader(self):
        self.assertEqual(queue.select_loader_workers(2, 32, 64*2**20), 0)
        self.assertEqual(queue.select_loader_workers(2, 32, 2**30), 2)

    def test_resume_requires_latest_and_training_plan_matches_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            cmd = queue.training_command(args, 'fit')
            self.assertNotIn('--resume', cmd)
            self.assertEqual(cmd[cmd.index('--epochs')+1], '150')
            self.assertEqual(cmd[cmd.index('--loss-type')+1], 'both')
            queue.write(args.output/'predictor/protocol.json', {})
            with self.assertRaises(RuntimeError):
                queue.training_command(args, 'fit')
            (args.output/'predictor/latest.pt').touch()
            self.assertIn('--resume', queue.training_command(args, 'fit'))

    def test_eval_plan_multitask_matched_and_no_forced_events(self):
        args = self.args(Path('/tmp/test_corrector_queue'))
        cmd = queue.eval_command(args, Path('/tmp/best.pt'), Path('/tmp/results'), hours=2)
        self.assertEqual(cmd[cmd.index('--task-ids')+1:cmd.index('--initial-state-ids')], list(map(str, range(10))))
        self.assertEqual(cmd[cmd.index('--methods')+1:cmd.index('--corrector-checkpoint')],
                         ['H50', 'H20', 'monitor', 'lvm', 'ogg'])
        self.assertNotIn('--debug-force-step', cmd)
        self.assertEqual(args.eval_initials*args.eval_seeds*10*5, 300)

    def test_plot_uses_existing_simulation_plot_dependencies(self):
        args = self.args(Path('/tmp/test_corrector_queue'))
        cmd = queue.plot_command(args, Path('/tmp/report'))
        self.assertEqual(Path(cmd[0]), args.runtime/'sim-venv/bin/python')

    def test_child_budget_requests_boundary_stop_and_logs_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            (args.output/'logs').mkdir(parents=True)
            args.poll_seconds = .01
            run = queue.Queue(args)
            run.env = dict(__import__('os').environ)
            stop = args.output/'TRAIN_STOP'
            script = 'import pathlib,time,sys\np=pathlib.Path(sys.argv[1])\nwhile not p.exists(): time.sleep(.01)'
            self.assertTrue(run.child([sys.executable, '-c', script, str(stop)], 'boundary', stop_path=stop, seconds=.01))
            self.assertTrue(stop.exists())
            with self.assertRaisesRegex(RuntimeError, '退出码2'):
                run.child([sys.executable, '-c', 'raise SystemExit(2)'], 'failed')

    def test_complete_queue_gates_training_and_selects_new_best_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            args.min_free_gib = .01  # 本地/tmp空间与远端50GiB数据盘无关。
            queue.write(args.archive/'manifest.json', json.loads(MANIFEST.read_text()))
            (args.output/'logs').mkdir(parents=True)
            run = queue.Queue(args)
            calls = []
            def child(cmd, label, **kw):
                calls.append(label)
                if label == 'training':
                    queue.write(args.output/'predictor/result.json', {'checkpoint_reload_verified': True})
                    queue.write(args.output/'predictor/training_status.json', {'status': 'completed'})
                    (args.output/'predictor/best_model.pt').write_bytes(b'test fixture, not real weights')
                if label in {'smoke', 'paired'}:
                    queue.write(Path(cmd[cmd.index('--output')+1])/'execution.json', {'status': 'completed', 'errors': 0})
            with patch.object(queue.subprocess, 'check_output', return_value=queue.SOURCE_SHA+'\n'), \
                 patch.object(run, 'wait_features', return_value=True), patch.object(run, 'wait_gpu', return_value=True), \
                 patch.object(run, 'child', side_effect=child):
                run.run()
            self.assertEqual(calls, ['inspect', 'training', 'prediction_report', 'plots', 'smoke', 'paired'])
            self.assertEqual(run.state['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
