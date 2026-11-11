"""练习框架、用户核心计算与单步更新检查；使用小构造张量，不代表真实训练效果。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .dataset import CachedTransitionDataset, build_episode_splits
from .model import CorrectorConfig, VisualDynamicsMLP
from .transitions import make_transition_pairs
from .losses import corrector_loss


def cache(path, episode, split):
    path.mkdir()
    np.save(path / "z_q.npy", np.zeros((2, 2, 4), dtype=np.int8))
    np.save(path / "z_scale.npy", np.ones((2, 2), dtype=np.float16))
    np.save(path / "actions.npy", np.zeros((2, 7), dtype=np.float16))
    np.save(path / "episode_index.npy", np.array([episode, episode]))
    np.save(path / "frame_index.npy", np.array([0, 1]))
    (path / "metadata.json").write_text(json.dumps({"split": split}))
    (path / "COMPLETE.json").write_text("{}")  # 只测试split组合；SHA检验有独立归档测试


class PracticeFrameworkTests(unittest.TestCase):
    def test_completed_core_methods_pass_practice_checks(self):
        # scripts是本项目的目录，不是Python包；按文件加载，避免误导入ROS同名包。
        script = Path(__file__).resolve().parents[3] / "scripts/check_corrector_practice.py"
        spec = importlib.util.spec_from_file_location("corrector_practice_checks", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for stage in ("dataset", "block", "model", "loss", "step"):
            with self.subTest(stage=stage):
                module.check(stage)

    def test_cosine_loss_keeps_samples_separate(self):
        # 两个样本分别同向和反向，cos分别1和-1，因此平均cos loss为1。
        pred = torch.tensor([[[1., 0.]], [[0., -3.]]], requires_grad=True)
        target = torch.tensor([[[1., 0.]], [[0., 1.]]])
        total, mse, cos = corrector_loss(pred, target)
        torch.testing.assert_close(mse, torch.tensor(4.))
        torch.testing.assert_close(cos, torch.tensor(1.))
        torch.testing.assert_close(total, torch.tensor(5.))
        total.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_dataset_really_calls_user_pair_function_and_splits_by_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache(root / "episode_813", 813, "train")
            cache(root / "episode_825", 825, "validation")
            with patch("studies.policy_generalization.corrector.dataset.make_transition_pairs", wraps=make_transition_pairs) as spy:
                ds = CachedTransitionDataset(root / "episode_813")
                self.assertEqual(ds.pairs, [(0, 1)])
                self.assertEqual(spy.call_count, 1)
            groups = build_episode_splits(root)
            self.assertEqual(len(groups["train"]), 1)
            self.assertEqual(len(groups["validation"]), 1)
            cache(root / "episode_duplicate", 813, "validation")
            with self.assertRaises(ValueError):
                build_episode_splits(root)

    def test_model_registers_trainable_layers_before_forward_is_written(self):
        model = VisualDynamicsMLP(CorrectorConfig(token_dim=4, action_embed_dim=3, widths=(8, 8)))
        state = model.state_dict()
        self.assertEqual(tuple(state["action_proj.weight"].shape), (3, 7))
        self.assertEqual(tuple(state["in_proj.weight"].shape), (8, 7))
        self.assertEqual(tuple(state["out_proj.weight"].shape), (4, 8))
        self.assertIn("blocks.1.fc2.weight", state)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
