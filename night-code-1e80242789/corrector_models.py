"""固定上游预测器工厂：保留旧MLP权重协议，增加历史Transformer，不导入练习实现。"""
from dataclasses import dataclass, asdict
import importlib.util
import hashlib
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    token_dim: int = 960
    action_dim: int = 7
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 5
    dropout: float = 0.


def transformer_class(source_root):
    path = Path(source_root)/'src/siglip_dynamics/transformer.py'
    spec = importlib.util.spec_from_file_location('_fixed_corrector_transformer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class UpstreamTransformer(module.SiglipResidualTransformer):
        uses_history = True

        def __init__(self, cfg):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='enable_nested_tensor is True.*norm_first', category=UserWarning)
                super().__init__(**asdict(cfg))
            self.cfg = cfg

    return UpstreamTransformer


def source_hashes(base, source_root, model_type, digest):
    result = dict(base)
    if model_type == 'transformer':
        result['transformer.py'] = digest(Path(source_root)/'src/siglip_dynamics/transformer.py')
    return result


def from_protocol(protocol, config_module, mlp_class, source_root):
    expected = protocol.get('model_factory_sha256')
    if expected and expected != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError('预测器工厂版本与训练时不同；使用原冻结部署包')
    kind = protocol.get('model_type', 'mlp')
    cfg = dict(protocol['model'])
    if kind == 'mlp':
        cfg['scale'] = config_module.ModelScale(cfg['scale'])
        cfg['custom_widths'] = tuple(cfg['custom_widths'])
        return precision_class(mlp_class, protocol.get('precision','float32')), config_module.SiglipMLPConfig(**cfg)
    if kind == 'transformer':
        return precision_class(transformer_class(source_root), protocol.get('precision','float32')), TransformerConfig(**cfg)
    raise ValueError(f'未知预测器类型：{kind}')


def precision_class(cls, precision):
    if precision == 'float32': return cls
    if precision != 'amp-bfloat16': raise ValueError('未知预测器精度')

    class AutocastPredictor(cls):
        def forward(self, visual, action):
            # 仅包围预测器；不改变冻结VLA的执行精度。参数/优化器仍FP32，输出回FP32。
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=visual.is_cuda):
                return super().forward(visual, action).float()

    return AutocastPredictor


def model_input(model, batch):
    return batch['z_hist'] if getattr(model, 'uses_history', False) else batch['z_t']


def history_at(history, step, frames, frame_stride):
    """只取已观测到的、按训练帧间距排列的历史；不足时返回None，不填充未来。"""
    wanted = [step-(frames-1-i)*frame_stride for i in range(frames)]
    available = dict(history)
    if any(t not in available for t in wanted):
        return None
    return torch.stack([available[t] for t in wanted], dim=1)


def guided_history_velocity(predictor, z_hist, x_tau, velocity, timestep, target, eta=1.):
    """与原单动作OGG同一目标/首动作约定，历史上下文固定，动作输入保留梯度。"""
    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError('OGG需要冻结且eval的预测器')
    with torch.enable_grad():
        v = velocity.detach().clone().requires_grad_(True)
        clean = x_tau.detach()-timestep[:, None, None]*v
        predicted = predictor(z_hist.detach(), clean[:, 0, :predictor.cfg.action_dim])
        loss = (1-F.cosine_similarity(predicted.flatten(1), target.detach().flatten(1), dim=1)).mean()
        gradient = torch.autograd.grad(loss, v)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError('历史OGG损失/梯度非有限')
    return (v-eta*gradient).detach(), loss.item()
