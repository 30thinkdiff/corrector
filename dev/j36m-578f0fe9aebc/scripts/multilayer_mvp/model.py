"""Full-resolution delta supervision, bounded O(tokens * width²) shared trunk.

Global means are conditioning summaries only. Targets/outputs remain unpooled.
Connector tokens have no assumed 2-D/camera correspondence.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    variant: str = 'joint'
    raw_shape: tuple = (2, 1024, 768)
    connector_shape: tuple = (128, 960)
    width: int = 96
    depth: int = 2


class SharedBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                                 nn.GELU(), nn.Linear(2*width, width))

    def forward(self, x):
        return x + self.net(x)


class MultiLayerCorrector(nn.Module):
    def __init__(self, cfg=ModelConfig()):
        super().__init__()
        if cfg.variant not in ('connector', 'joint') or cfg.width < 4 or cfg.depth < 1:
            raise ValueError('invalid model config')
        self.cfg = cfg
        c, p, d = cfg.raw_shape
        self.side = math.isqrt(p)
        if self.side**2 != p:
            raise ValueError('spatial tokens must form a square grid')
        w = cfg.width
        self.connector_projection = nn.Linear(cfg.connector_shape[1], w)
        self.action_projection = nn.Linear(7, w)
        self.connector_position = nn.Parameter(torch.zeros(1, cfg.connector_shape[0], w))
        self.backbone = nn.Sequential(*(SharedBlock(w) for _ in range(cfg.depth)))
        self.heads = nn.ModuleDict({'delta_connector': nn.Linear(w, cfg.connector_shape[1])})
        if cfg.variant == 'joint':
            self.block_02_projection = nn.Linear(d, w)
            self.block_08_projection = nn.Linear(d, w)
            self.spatial_position = nn.Parameter(torch.zeros(1, c*p, w))
            self.local = nn.Conv2d(w, w, 3, padding=1, groups=w)
            self.heads.update({'delta_block_02': nn.Linear(w, d), 'delta_block_08': nn.Linear(w, d)})

    def forward(self, inputs):
        action = inputs['action']
        z = inputs['connector']
        if action.ndim != 2 or action.shape[-1] != 7 or tuple(z.shape[1:]) != tuple(self.cfg.connector_shape):
            raise ValueError('expected current action [B,7] and Connector [B,128,960]')
        h_z = self.connector_projection(z) + self.connector_position
        a = self.action_projection(action)[:, None, :]
        if self.cfg.variant == 'connector':
            return {'delta_connector': self.heads['delta_connector'](self.backbone(h_z + a))}
        c, p, d = self.cfg.raw_shape
        b = z.shape[0]
        for name in ('block_02', 'block_08'):
            if tuple(inputs[name].shape) != (b, c, p, d):
                raise ValueError(f'{name}: full-resolution current-frame shape mismatch')
        h = (self.block_02_projection(inputs['block_02']) + self.block_08_projection(inputs['block_08']))
        h = h.reshape(b, c*p, -1) + self.spatial_position
        # Explicit global cross-stream conditioning; does not change supervision space.
        spatial = h + h_z.mean(1, keepdim=True) + a
        grid = spatial.reshape(b*c, self.side, self.side, -1).permute(0, 3, 1, 2)
        spatial = spatial + self.local(grid).permute(0, 2, 3, 1).reshape(b, c*p, -1)
        connector = h_z + h.mean(1, keepdim=True) + a
        shared = self.backbone(torch.cat((spatial, connector), dim=1))
        return {'delta_block_02': self.heads['delta_block_02'](shared[:, :c*p]).reshape(b, c, p, d),
                'delta_block_08': self.heads['delta_block_08'](shared[:, :c*p]).reshape(b, c, p, d),
                'delta_connector': self.heads['delta_connector'](shared[:, c*p:])}

    def config_dict(self):
        return asdict(self.cfg)
