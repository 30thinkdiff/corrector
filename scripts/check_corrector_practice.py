"""CPU调用用户核心方法；构造小张量检查接口，不计为真实Corrector训练/评测。"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from studies.policy_generalization.corrector.dataset import CachedTransitionDataset
from studies.policy_generalization.corrector.losses import corrector_loss
from studies.policy_generalization.corrector.model import CorrectorConfig, ResidualMLPBlock, VisualDynamicsMLP
from studies.policy_generalization.corrector.training_step import train_corrector_batch


def check(stage):
    torch.manual_seed(7)
    if stage in {"mismatch", "threshold", "guidance"}:
        from studies.policy_generalization.corrector.online_practice import visual_mismatch, robust_limits, guided_velocity
        if stage == "mismatch":
            predicted = torch.tensor([[[1., 0.]], [[1., 0.]]])
            observed = torch.tensor([[[1., 0.]], [[-1., 0.]]])
            torch.testing.assert_close(visual_mismatch(predicted, observed), torch.tensor([0., 2.]))
        elif stage == "threshold":
            on, off = robust_limits([0., .2, .4, .6], noise_floor=0.)
            assert np.isclose(on, .9) and np.isclose(off, .7)
        else:
            model = VisualDynamicsMLP(CorrectorConfig(token_dim=4, action_dim=7, action_embed_dim=3, widths=(8, 8)))
            model.eval().requires_grad_(False)
            z, target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
            x, velocity, t = torch.randn(2, 5, 10), torch.randn(2, 5, 10), torch.tensor([.5, .7])
            with torch.no_grad():
                result, loss = guided_velocity(model, z, x, velocity, t, target)
            assert result.shape == velocity.shape and not result.requires_grad
            assert isinstance(loss, float) and np.isfinite(loss)
            assert (result[:, 0, :7] - velocity[:, 0, :7]).abs().sum() > 0
            torch.testing.assert_close(result[:, 1:], velocity[:, 1:])
            torch.testing.assert_close(result[:, 0, 7:], velocity[:, 0, 7:])
            assert all(p.grad is None for p in model.parameters())
    elif stage == "dataset":
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.save(path / "z_q.npy", np.arange(32, dtype=np.int8).reshape(4, 2, 4))
            np.save(path / "z_scale.npy", np.full((4, 2), .01, dtype=np.float16))
            np.save(path / "actions.npy", np.ones((4, 7), dtype=np.float16))
            np.save(path / "episode_index.npy", np.array([813, 813, 818, 818]))
            np.save(path / "frame_index.npy", np.array([0, 1, 0, 1]))
            (path / "metadata.json").write_text(json.dumps({"split": "train"}))
            ds = CachedTransitionDataset(path)
            assert ds.pairs == [(0, 1), (2, 3)]
            item = ds[0]  # 真实调用用户__getitem__
            assert set(item) == {"z_t", "a_t", "delta_z"}
            assert all(v.dtype == torch.float32 for v in item.values())
            torch.testing.assert_close(item["delta_z"], torch.full((2, 4), 8 * float(np.float16(.01))))
            assert next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))["z_t"].shape == (2, 2, 4)
    elif stage == "block":
        block = ResidualMLPBlock(4)
        x = torch.randn(2, 3, 4)
        y = block(x)
        assert y.shape == x.shape
        y.square().mean().backward()
        assert block.fc1.weight.grad is not None and block.fc1.weight.grad.abs().sum() > 0
        with torch.no_grad():
            block.fc2.weight.zero_(); block.fc2.bias.zero_()
        torch.testing.assert_close(block(x), x)
    elif stage == "loss":
        target = torch.tensor([[[2., 4.]]])
        for kind, expected in [("mse", 2.5), ("cosine", 0.), ("both", 2.5)]:
            pred = torch.tensor([[[1., 2.]]], requires_grad=True)
            loss, mse, cos = corrector_loss(pred, target, loss_type=kind)
            assert loss.ndim == mse.ndim == cos.ndim == 0
            torch.testing.assert_close(loss, torch.tensor(expected), atol=1e-6, rtol=1e-6)
            loss.backward()
            assert pred.grad is not None and torch.isfinite(pred.grad).all()
        pred = torch.tensor([[[-1., -2.]]])
        torch.testing.assert_close(corrector_loss(pred, target, beta=.5)[0], torch.tensor(23.5))
    else:
        cfg = CorrectorConfig(token_dim=4, action_dim=7, action_embed_dim=3, widths=(8, 8))
        model = VisualDynamicsMLP(cfg)
        z, a = torch.randn(2, 3, 4), torch.randn(2, 7)
        if stage == "model":
            result = model(z, a)
            assert result.shape == z.shape and torch.isfinite(result).all()
            result.square().mean().backward()
            assert model.action_proj.weight.grad is not None and model.action_proj.weight.grad.abs().sum() > 0
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            before = {k: v.detach().clone() for k, v in model.named_parameters()}
            value = train_corrector_batch(model, {"z_t": z, "a_t": a, "delta_z": torch.randn_like(z)}, optimizer)
            assert isinstance(value, float) and np.isfinite(value)
            assert any(not torch.equal(before[k], v) for k, v in model.named_parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "online", "dataset", "block", "model", "loss", "step",
                                            "mismatch", "threshold", "guidance"], default="all")
    args = parser.parse_args()
    stages = (["dataset", "block", "model", "loss", "step"] if args.stage == "all" else
              ["mismatch", "threshold", "guidance"] if args.stage == "online" else [args.stage])
    incomplete = False
    for stage in stages:
        try:
            check(stage)
            print(stage, "PASS（仅CPU接口/数学检查）")
        except NotImplementedError as error:
            incomplete = True
            print(stage, "待用户实现：", error)
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
