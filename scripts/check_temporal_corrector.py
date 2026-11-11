"""小CPU张量验收历史Transformer练习；reference为显式上游示范，不自动替代学生实现。"""
import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.policy_generalization.corrector.temporal_practice import TemporalConfig, TemporalPredictor, history_window


def reference_model(source, cfg):
    path = source/'src/siglip_dynamics/transformer.py'
    spec = importlib.util.spec_from_file_location('_reference_temporal_transformer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = asdict(cfg); options.pop('max_history')
    if cfg.max_history != 512:
        raise ValueError('本练习与固定上游比较时保留其512个时间embedding')
    # 上游norm_first关闭nested优化；仅屏蔽这个已知的可选优化提示。
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='enable_nested_tensor is True.*norm_first', category=UserWarning)
        return module.SiglipResidualTransformer(**options)


def check_history(implementation):
    frames = torch.arange(8*2*4).reshape(8, 2, 4).float()
    ids = torch.tensor([7]*4+[8]*4)
    if implementation == 'student':
        select = history_window
    else:
        def select(z, episodes, t, h):
            start = t-h+1
            if start < 0 or not bool((episodes[start:t+1] == episodes[t]).all()):
                raise ValueError('历史跨边界')
            return z[start:t+1]
    actual = select(frames, ids, 6, 3)
    torch.testing.assert_close(actual, frames[4:7])
    if actual.shape != (3, 2, 4):
        raise AssertionError('历史形状错误')
    for t, h in ((0, 3), (4, 2)):
        try:
            select(frames, ids, t, h)
        except ValueError:
            pass
        else:
            raise AssertionError('必须拒绝负下标和跨episode历史')
    print('history通过：仅使用截至t的同轨迹历史，不跨episode。')


def check_model(implementation, source):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = TemporalConfig(token_dim=8, action_dim=7, d_model=16, n_heads=4, n_layers=1)
    student = TemporalPredictor(cfg).eval()
    reference = reference_model(source, cfg).eval()
    reference.load_state_dict(student.state_dict(), strict=True)
    selected = student if implementation == 'student' else reference
    for h in (1, 3):
        z = torch.randn(2, h, 4, 8, requires_grad=True)
        action = torch.randn(2, 7, requires_grad=True)
        target = torch.randn(2, 4, 8)
        pred = selected(z, action)
        if pred.shape != target.shape:
            raise AssertionError('输出应是当前帧之后的变化[B,L,D]，不能保留历史轴')
        loss = (pred-target).square().mean()
        grads = torch.autograd.grad(loss, [z, action, *selected.parameters()])
        if any(not torch.isfinite(g).all() for g in grads) or action.grad is not None:
            raise AssertionError('梯度需有限；autograd.grad不向输入.grad累加')
        if float(grads[1].norm()) == 0:
            raise AssertionError('动作分支必须参与计算图')
        if implementation == 'student':
            peer = reference(z, action)
            torch.testing.assert_close(pred, peer, rtol=1e-5, atol=1e-6)
            expected = torch.autograd.grad((peer-target).square().mean(), [z, action, *reference.parameters()])
            for actual, wanted in zip(grads, expected):
                torch.testing.assert_close(actual, wanted, rtol=1e-4, atol=1e-6)
        print(f'model通过：H={h}，序列长={1+h*4}，输出={tuple(pred.shape)}，输入/参数梯度有限。')
    print('这里只检查构造数据与源码一致性，不是训练性能或机器人成功率。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['history', 'model', 'all'], default='all')
    parser.add_argument('--implementation', choices=['student', 'reference'], default='student')
    parser.add_argument('--source-root', type=Path, default=ROOT/'outputs/source_audit/vla-corrector')
    args = parser.parse_args()
    try:
        if args.stage in {'history', 'all'}:
            check_history(args.implementation)
        if args.stage in {'model', 'all'}:
            check_model(args.implementation, args.source_root)
    except NotImplementedError as error:
        print(f'待你实现：{error}；请编辑temporal_practice.py，不在Markdown里填代码。')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
