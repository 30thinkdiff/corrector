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
    if model_type in {'transformer','transformer-action'}:
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
    if kind == 'transformer-action':
        return precision_class(action_transformer_class(source_root), protocol.get('precision','float32')), ActionSequenceConfig(**cfg)
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

@dataclass
class ActionSequenceConfig(TransformerConfig):
    action_steps: int = 10


def action_transformer_class(source_root):
    """上游视觉Transformer的动作序列扩展；每个动作单独编码时间位置。"""
    base = transformer_class(source_root)

    class ActionSequenceTransformer(base):
        uses_action_sequence = True

        def __init__(self, cfg):
            super().__init__(TransformerConfig(**{k:v for k,v in asdict(cfg).items() if k!='action_steps'}))
            if cfg.action_steps < 1: raise ValueError('动作序列须非空')
            self.cfg = cfg
            self.action_time_embed = torch.nn.Embedding(cfg.action_steps, cfg.d_model)

        def forward(self, visual, actions):
            if visual.ndim == 3: visual = visual.unsqueeze(1)
            if (visual.ndim != 4 or actions.ndim != 3 or
                    actions.shape != (visual.shape[0],self.cfg.action_steps,self.cfg.action_dim)):
                raise ValueError('需要[B,H,L,D]视觉和[B,K,A]动作序列，不广播/补零')
            b,h,l,d = visual.shape
            if d != self.cfg.token_dim or h>512: raise ValueError('视觉维度/历史长度错误')
            x = self.z_proj(visual)
            x = x + self.time_embed(torch.arange(h,device=x.device))[None,:,None,:]
            a = self.a_proj(actions) + self.action_time_embed(torch.arange(self.cfg.action_steps,device=x.device))[None]
            y = self.encoder(torch.cat([a,x.reshape(b,h*l,-1)],dim=1))
            last = y[:,self.cfg.action_steps:].reshape(b,h,l,-1)[:,-1]
            return self.out_proj(self.norm(last))
    return ActionSequenceTransformer


def action_input(model, batch):
    return batch['a_seq'] if getattr(model,'uses_action_sequence',False) else batch['a_t']


class ActionSequenceDataset(torch.utils.data.Dataset):
    """只从同一示范轨迹读取[t,t+K)实际动作；不使用未来图像作为输入。"""
    def __init__(self, base, action_steps):
        self.base,self.action_steps = base,action_steps
        if not 1<=action_steps<=base.k_step: raise ValueError('动作范围必须位于预测终点之前')
        self.ep_idx,self.data_path = base.ep_idx,base.data_path
        self.pairs = base.pairs

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        import numpy as np
        item = self.base[index]
        t,future = self.pairs[index]
        stop=t+self.action_steps
        if stop>future or not np.all(self.ep_idx[t:stop]==self.ep_idx[t]):
            raise ValueError('动作序列越过预测终点或episode边界')
        item['a_seq']=torch.from_numpy(np.array(self.base.a_t_np[t:stop],dtype=np.float32,copy=True))
        return item


def planned_actions(current, queue, steps, stride):
    """当前返回动作＋尚未执行的规划队列，按数据帧间隔取样；不足则不预测。"""
    indices=torch.arange(steps,device=current.device)*stride
    available=[current,*list(queue)]
    if len(available)<=int(indices[-1]): return None
    return torch.stack(available,dim=1)[:,indices,:7].float()


def guided_sequence_velocity(predictor, visual, x_tau, velocity, timestep, target, stride, eta=1.):
    """对本次生成动作块的选中时间位置引导，固定视觉/纠偏目标，不偷看未来观测。"""
    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError('OGG需要冻结eval预测器')
    indices=torch.arange(predictor.cfg.action_steps,device=x_tau.device)*stride
    if int(indices[-1])>=x_tau.shape[1]: raise ValueError('生成动作块不足以覆盖预测条件')
    with torch.enable_grad():
        v=velocity.detach().clone().requires_grad_(True)
        clean=x_tau.detach()-timestep[:,None,None]*v
        actions=clean[:,indices,:predictor.cfg.action_dim]
        predicted=predictor(visual.detach(),actions)
        loss=(1-F.cosine_similarity(predicted.flatten(1),target.detach().flatten(1),dim=1)).mean()
        gradient=torch.autograd.grad(loss,v)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError('动作序列OGG损失/梯度非有限')
    return (v-eta*gradient).detach(),loss.item()
