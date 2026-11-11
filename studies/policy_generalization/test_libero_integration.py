"""CPU验证用户指标、回合超时/播放FPS和录像；不冒充完整VLA闭环。"""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("libero_tools", ROOT / "scripts/smolvla_libero_tools.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
from studies.policy_generalization.evaluation_metrics import masked_action_mae


class MetricAndLiberoTests(unittest.TestCase):
    def test_namespace_import_in_clean_interpreter_not_just_installed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            inner = source / 'libero/libero'
            inner.mkdir(parents=True)
            (inner / '__init__.py').write_text('sentinel = 42\n')
            subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(root / 'venv')], check=True)
            python = root / 'venv/bin/python'
            site = Path(subprocess.check_output([str(python), '-c',
                            "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
            metadata = site / 'libero-0.1.0.dist-info'
            metadata.mkdir()
            (metadata / 'METADATA').write_text('Name: libero\nVersion: 0.1.0\n')
            # 复现“已安装0.1.0，但源码根目录不在sys.path”：全新进程，隔离PYTHONPATH。
            subprocess.run([str(python), '-I', '-c',
                "import importlib.util,importlib.metadata; assert importlib.metadata.version('libero')=='0.1.0'; assert importlib.util.find_spec('libero') is None"], check=True)
            sim.register_source_root(site, source)
            subprocess.run([str(python), '-I', '-c',
                "import libero.libero as m; assert m.sentinel==42; print('namespace import PASS')"], check=True)

    def test_user_metric_weights_valid_scalars_and_ignores_padding_nan(self):
        import torch
        p = torch.tensor([[[1., 3.], [float('nan'), float('nan')]], [[5., 7.], [9., 11.]]])
        mask = torch.tensor([[False, True], [False, False]])
        self.assertEqual(masked_action_mae(p, torch.zeros_like(p), mask), 6.)

    def test_user_metric_rejects_broadcastable_wrong_mask_and_empty_actions(self):
        import torch
        for p, mask in [(torch.zeros(1, 2, 2), torch.zeros(1, 1, dtype=torch.bool)),
                        (torch.zeros(1, 2, 0), torch.zeros(1, 2, dtype=torch.bool))]:
            with self.assertRaises(ValueError):
                masked_action_mae(p, p, mask)

    def test_task_match_uses_exact_language_not_view_suffix_or_dataset_id(self):
        tasks = [SimpleNamespace(language=sim.TASK_LANGUAGE + " view 1"), SimpleNamespace(language=sim.TASK_LANGUAGE)]
        benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_object": lambda: SimpleNamespace(tasks=tasks)})
        self.assertEqual(sim.find_task(benchmark)[0], 1)
        tasks[:] = tasks[:1]
        with self.assertRaises(RuntimeError):
            sim.find_task(benchmark)

    def test_timeout_and_video_metadata_are_instance_scoped(self):
        import gymnasium as gym
        import numpy as np
        class Env(gym.Env):
            metadata = {"render_fps": 80}
            observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(1, dtype=np.float32), 0., False, False, {}
        env = gym.vector.SyncVectorEnv([Env])
        try:
            sim.configure_vector(env, 20, 2)
            self.assertEqual(env.metadata['render_fps'], 20)
            self.assertEqual(Env.metadata['render_fps'], 80)
            env.reset(seed=[1])
            self.assertFalse(env.step(np.zeros((1, 1)))[3][0])
            self.assertTrue(env.step(np.zeros((1, 1)))[3][0])
        finally:
            env.close()

    def test_real_mp4_write_decode_and_fps_check(self):
        import numpy as np
        from lerobot.utils.io_utils import write_video
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.mp4'
            write_video(path, [np.zeros((32, 32, 3), dtype=np.uint8)] * 3, 20)
            row = sim.verify_video(path, 20)
            self.assertEqual(row['frames'], 3)
            self.assertAlmostEqual(row['duration_s'], .15)
            with self.assertRaises(RuntimeError):
                sim.verify_video(path, 10)


if __name__ == '__main__':
    unittest.main()
