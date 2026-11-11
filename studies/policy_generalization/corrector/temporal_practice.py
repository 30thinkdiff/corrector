"""历史条件Transformer练习：外围/层已准备，三个核心方法由用户填写。

参考固定上游siglip_dynamics/transformer.py与dataset.py。不会被当前训练/在线任务导入。
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalConfig:
    token_dim: int = 960
    action_dim: int = 7
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 4
    max_history: int = 512


def history_window(z_frames, episode_ids, t, h_window):
    """任务T1：[N,L,D]→[H,L,D]，包含t，不含t之后的数据。

    先确定起始下标，再确认所选历史与t属于同一episode，然后返回历史切片。
    跨轨迹应raise ValueError，不能负下标绕到轨迹末尾或拼接另一episode。
    使用的操作：下标运算、切片、torch.all、Tensor.item。不要取未来z_(t+k)。
    """
    if z_frames.ndim != 3 or episode_ids.shape != (len(z_frames),):
        raise ValueError('需要z_frames[N,L,D]、episode_ids[N]')
    if not isinstance(t, int) or not isinstance(h_window, int) or h_window < 1 or not h_window-1 <= t < len(z_frames):
        raise ValueError('历史长度/当前下标越界')
    raise NotImplementedError('T1：提取同episode、截至t的H帧历史')


class TemporalPredictor(nn.Module):
    def __init__(self, cfg=TemporalConfig()):
        super().__init__()
        if min(cfg.token_dim, cfg.action_dim, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.max_history) < 1 or cfg.d_model % cfg.n_heads:
            raise ValueError('维度须为正，d_model须能被n_heads整除')
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.token_dim, cfg.d_model)
        self.a_proj = nn.Linear(cfg.action_dim, cfg.d_model)
        self.time_embed = nn.Embedding(cfg.max_history, cfg.d_model)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=4*cfg.d_model, dropout=0., activation='gelu', batch_first=True, norm_first=True)
        # norm_first路径不使用nested tensor优化，显式关闭，避免无意义的兼容性warning。
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.token_dim)

    def pack_tokens(self, z_hist, a_t):
        """任务T2：[B,H,L,D]和[B,A]→[B,1+H*L,C]。

        视觉投影→加历史时间embedding→合并H/L轴→最前面拼一个动作token。
        H=3,L=128时序列长385；保持batch轴和时间顺序。
        常用操作：Linear、torch.arange(device=...)、Embedding、view/reshape、unsqueeze、torch.cat。
        不要在forward新建Linear/Embedding，不要把batch轴一起压平。
        """
        raise NotImplementedError('T2：构造动作token与历史视觉token序列')

    def last_frame_tokens(self, encoded, h_window, n_tokens):
        """任务T3：[B,1+H*L,C]→[B,L,C]。

        去掉最前面的动作token，恢复历史和视觉token轴，只取最后一帧的L个输出。
        后面的norm/out_proj已由forward调用。不要对全部token直接mean。
        """
        raise NotImplementedError('T3：提取最后一帧的视觉token输出')

    def forward(self, z_hist, a_t):
        if z_hist.ndim != 4 or a_t.ndim != 2:
            raise ValueError('练习入口显式接收[B,H,L,D]和[B,A]')
        b, h, l, d = z_hist.shape
        if not 1 <= h <= self.cfg.max_history or l < 1 or d != self.cfg.token_dim or a_t.shape != (b, self.cfg.action_dim):
            raise ValueError('历史/视觉维度/动作batch不一致')
        packed = self.pack_tokens(z_hist, a_t)
        if packed.shape != (b, 1+h*l, self.cfg.d_model):
            raise ValueError('T2输出形状错误')
        encoded = self.encoder(packed)
        last = self.last_frame_tokens(encoded, h, l)
        if last.shape != (b, l, self.cfg.d_model):
            raise ValueError('T3输出形状错误')
        return self.out_proj(self.norm(last))
