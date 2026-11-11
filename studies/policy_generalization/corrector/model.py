"""Corrector学习实现：层结构已搭好，forward由用户亲写。

结构参考Apache-2.0的ZJU-OmniAI/vla-corrector@927a06d/src/siglip_dynamics/MLP.py。
初版仅MLP、h_window=1；不用改outputs中的参考源码。
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CorrectorConfig:
    token_dim: int = 960          # 真正训练时从缓存metadata读取
    action_dim: int = 7
    action_embed_dim: int = 256
    widths: tuple[int, ...] = (256, 256)  # 实践起点；不是原论文参数规模声明
    dropout: float = 0.0


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """练习A：[B,L,W]→[B,L,W]；最后输出x加上变换后的分支。

        可用层：norm→fc1→act→drop→fc2→drop。
        参考MLP.py的_MLPBlock.forward；保留原x用于最后的残差相加。
        """
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)  # Dropout：训练时随机置零并缩放；当前p=0，不改变数值。
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class VisualDynamicsMLP(nn.Module):
    def __init__(self, cfg: CorrectorConfig):
        super().__init__()
        if len(cfg.widths) < 2 or any(w <= 0 for w in cfg.widths):
            raise ValueError("至少两个正整数width；与参考MLP层结构约定一致")
        self.cfg = cfg
        self.action_proj = nn.Linear(cfg.action_dim, cfg.action_embed_dim)
        self.in_proj = nn.Linear(cfg.token_dim + cfg.action_embed_dim, cfg.widths[0])
        self.blocks = nn.ModuleList([ResidualMLPBlock(w, cfg.dropout) for w in cfg.widths])
        self.transitions = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(a), nn.Linear(a, b), nn.SiLU()) if a != b else nn.Identity()
            for a, b in zip(cfg.widths[:-1], cfg.widths[1:])
        ])
        self.out_norm = nn.LayerNorm(cfg.widths[-1])
        self.out_proj = nn.Linear(cfg.widths[-1], cfg.token_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z_t, a_t):
        """练习B：z_t[B,L,D]，a_t[B,A]→delta_hat[B,L,D]。

        第一版只支持带batch的输入，暂不要求参考实现的单样本兼容分支。
        1. action_proj将a_t投影到[B,C]。
        2. 插入token轴并展开到[B,L,C]。
        3. 在最后一维拼接z_t和动作特征，然后in_proj。
        4. 遍历blocks；每个block后按需调用对应transitions。
        5. out_norm、out_proj，返回变化预测。
        不detach、不转NumPy，否则训练梯度会断开。
        """
        if z_t.ndim != 3 or a_t.ndim != 2:
            raise ValueError("需要z_t[B,L,D]和a_t[B,A]")
        if z_t.shape[0] != a_t.shape[0] or z_t.shape[-1] != self.cfg.token_dim or a_t.shape[-1] != self.cfg.action_dim:
            raise ValueError("batch或特征宽度与配置不一致")
        embeded_action = self.action_proj(a_t)  # [B,C]
        # unsqueeze(1)等价于[:, None, :]；[B,C]→[B,1,C]→[B,L,C]。
        embeded_action = embeded_action.unsqueeze(1).expand(-1, z_t.shape[1], -1)
        x = torch.cat([z_t, embeded_action], dim=-1)  # [B,L,D+C]
        x = self.in_proj(x)  # [B,L,W]
        # N个block之间只有N-1个transition；zip会按较短列表截断，漏掉最后一个block。
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)  # 连接相邻block；宽度相同则为Identity。
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x
