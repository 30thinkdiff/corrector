"""固定上游源码的接口检查：小构造张量，不冒充真实视觉特征或训练成绩。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import json

import numpy as np
import torch

from studies.policy_generalization.corrector.transitions import make_transition_pairs
from studies.policy_generalization.corrector.model import CorrectorConfig, VisualDynamicsMLP

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/source_audit/vla-corrector/src"


@unittest.skipUnless((SOURCE / "siglip_dynamics/dataset.py").exists(), "需要本地只读上游审计源码")
class UpstreamContractTests(unittest.TestCase):
    def test_user_mlp_matches_upstream_outputs_and_gradients(self):
        sys.path.append(str(SOURCE))
        try:
            from siglip_dynamics.config import ModelScale, SiglipMLPConfig
            from siglip_dynamics.MLP import SiglipResidualMLP
        finally:
            sys.path.remove(str(SOURCE))
        # 等宽和变宽两种配置；防止最后一个block被漏掉而shape检查仍然通过。
        for widths in ((8, 8), (8, 6, 4)):
            with self.subTest(widths=widths):
                torch.manual_seed(7)
                reference = SiglipResidualMLP(SiglipMLPConfig(
                    token_dim=4, action_dim=7, action_embed_dim=3,
                    scale=ModelScale.CUSTOM, custom_widths=widths))
                model = VisualDynamicsMLP(CorrectorConfig(
                    token_dim=4, action_dim=7, action_embed_dim=3, widths=widths))
                model.load_state_dict(reference.state_dict(), strict=True)
                z, a = torch.randn(2, 3, 4), torch.randn(2, 7)
                actual, expected = model(z, a), reference(z, a)
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                expected.square().mean().backward()
                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    torch.testing.assert_close(param.grad, dict(reference.named_parameters())[name].grad)

    def test_contiguous_pair_parity_and_delta(self):
        spec = importlib.util.spec_from_file_location("audited_corrector_dataset", SOURCE / "siglip_dynamics/dataset.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = [813] * 4 + [818] * 3
            frames = [0, 1, 2, 3, 0, 1, 2]
            q = np.arange(7 * 2 * 4, dtype=np.int8).reshape(7, 2, 4)
            np.save(root / "z_q.npy", q)
            np.save(root / "z_scale.npy", np.ones((7, 2), dtype=np.float16) * .01)
            np.save(root / "actions.npy", np.ones((7, 7), dtype=np.float16))
            np.save(root / "episode_index.npy", np.array(episodes, dtype=np.int32))
            (root / "metadata.json").write_text(json.dumps({"format": "siglip_frame_cache_v2"}))
            for k in (1, 2):
                ds = module.SiglipDynamicsDataset(str(root), k_step=k, h_window=1)
                self.assertEqual(ds.pairs, make_transition_pairs(episodes, frames, k))
                item = ds[0]
                self.assertEqual(tuple(item["z_t"].shape), (2, 4))
                self.assertEqual(tuple(item["a_t"].shape), (7,))
                torch.testing.assert_close(item["delta_z"], torch.full((2, 4), 8 * k * float(np.float16(.01))))
                batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
                self.assertEqual(tuple(batch["z_t"].shape), (2, 2, 4))
                self.assertEqual(tuple(batch["delta_z"].shape), (2, 2, 4))

    def test_mlp_broadcasts_one_action_to_each_token(self):
        sys.path.append(str(SOURCE))
        try:
            from siglip_dynamics.config import ModelScale, SiglipMLPConfig
            from siglip_dynamics.MLP import SiglipResidualMLP
        finally:
            sys.path.remove(str(SOURCE))
        cfg = SiglipMLPConfig(token_dim=4, action_dim=7, action_embed_dim=3,
                             scale=ModelScale.CUSTOM, custom_widths=(8, 8))
        torch.manual_seed(7)
        model = SiglipResidualMLP(cfg)
        z = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 10
        a = torch.ones((2, 7))
        captured = []
        hook = model.in_proj.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        try:
            pred = model(z, a)
        finally:
            hook.remove()
        self.assertEqual(tuple(pred.shape), (2, 2, 4))
        self.assertEqual(tuple(captured[0].shape), (2, 2, 7))
        torch.testing.assert_close(captured[0][..., :4], z)
        torch.testing.assert_close(captured[0][:, 0, 4:], model.action_proj(a))
        torch.testing.assert_close(captured[0][:, 0, 4:], captured[0][:, 1, 4:])
