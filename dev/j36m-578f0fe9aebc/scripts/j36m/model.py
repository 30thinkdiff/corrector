"""Formal fixed J-36M. Grouping is a permutation, never target pooling."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .reference_mlp.MLP import SiglipResidualMLP
from .reference_mlp.config import SiglipMLPConfig, ModelScale

LAYERS = ('block_02', 'block_08', 'connector')
SHAPES = {'block_02': (2,1024,768), 'block_08': (2,1024,768), 'connector': (128,960)}


def patchify(x):
    if tuple(x.shape[1:]) != (2,1024,768):
        raise ValueError('requires full [B,2,1024,768]')
    return x.reshape(-1,2,8,4,8,4,768).permute(0,1,2,4,3,5,6).reshape(-1,128,12288)


def unpatchify(x):
    if tuple(x.shape[1:]) != (128,12288):
        raise ValueError('requires [B,128,12288] local groups')
    return x.reshape(-1,2,8,8,4,4,768).permute(0,1,2,4,3,5,6).reshape(-1,2,1024,768)


def reference_pixel_shuffle(x, factor=4):
    """Verbatim layout operations from Transformers v5.5.4 SmolVLMConnector.
    Runtime station audit ALSO checks installed connector.pixel_shuffle against this.
    """
    b,s,d=x.size();h=w=int(s**.5)
    if h*w != s or h%factor:
        raise ValueError('non-square / non-divisible vision tokens')
    x=x.view(b,h,w,d).view(b,h,w//factor,d*factor).permute(0,2,1,3)
    return x.reshape(b,w//factor,h//factor,d*factor**2).permute(0,2,1,3).reshape(b,s//factor**2,d*factor**2)


class Joint36M(nn.Module):
    architecture = 'J-36M-v1'
    def __init__(self, input_rms=None, gradient_checkpointing=False):
        super().__init__()
        self.cfg=SimpleNamespace(action_dim=7,token_dim=960)
        self.gradient_checkpointing=gradient_checkpointing
        self.adapters=nn.ModuleDict({k:nn.Linear(12288 if k!='connector' else 960,512) for k in LAYERS})
        self.fusion=nn.Linear(1536,768)
        self.action_embed=nn.Sequential(nn.Linear(7,256),nn.SiLU())
        self.film=nn.Linear(256,1536)
        self.backbone=nn.ModuleList([nn.TransformerEncoderLayer(768,8,1536,dropout=0.,batch_first=True,norm_first=True)
                                    for _ in range(3)])
        self.heads=nn.ModuleDict({k:nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,12288))
                                  for k in LAYERS[:2]})
        self.heads['connector']=nn.Linear(768,960)
        self.local=nn.ModuleDict({k:nn.Sequential(nn.Linear(1024,128),nn.SiLU(),nn.Linear(128,768)) for k in LAYERS[:2]})
        self.camera=nn.Parameter(torch.empty(2,768));self.position=nn.Parameter(torch.empty(64,768))
        nn.init.normal_(self.camera,std=.02);nn.init.normal_(self.position,std=.02)
        values=input_rms or {k:1. for k in LAYERS}
        scale=torch.tensor([float(values[k]) for k in LAYERS])
        if not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('positive finite train-only input RMS required')
        self.register_buffer('input_rms',scale)
        if sum(p.numel() for p in self.parameters())!=36793536:
            raise RuntimeError('formal architecture parameter budget changed')

    def forward(self, inputs):
        a=inputs['action']
        if a.ndim!=2 or a.shape[1]!=7:
            raise ValueError('one current action [B,7] only')
        x={}
        for i,k in enumerate(LAYERS):
            if inputs[k].shape != (a.shape[0],*SHAPES[k]):
                raise ValueError('full-resolution visual shape mismatch: '+k)
            x[k]=inputs[k]/self.input_rms[i]
        h=self.fusion(torch.cat([self.adapters[k](patchify(x[k]) if k!='connector' else x[k]) for k in LAYERS],-1))
        pos=(self.camera[:,None,:]+self.position[None,:,:]).reshape(1,128,768)
        emb=self.action_embed(a);gamma,beta=self.film(emb).chunk(2,-1)
        h=(h+pos)*(1+gamma[:,None,:])+beta[:,None,:]
        for layer in self.backbone:
            h=checkpoint(layer,h,use_reentrant=False) if self.gradient_checkpointing and self.training else layer(h)
        out={'delta_connector':self.heads['connector'](h)}
        for k in LAYERS[:2]:
            local_action=emb[:,None,None,:].expand(-1,2,1024,-1)
            out['delta_'+k]=unpatchify(self.heads[k](h))+self.local[k](torch.cat([x[k],local_action],-1))
        return out


class ReferenceSingle(nn.Module):
    """Verbatim upstream M20 residual MLP, only token_dim differs for S2/S8."""
    def __init__(self, layer):
        super().__init__()
        if layer not in LAYERS:raise ValueError(layer)
        self.layer=layer
        self.model=SiglipResidualMLP(SiglipMLPConfig(token_dim=960 if layer=='connector' else 768,scale=ModelScale.M20))
        self.cfg=self.model.cfg

    def forward(self,inputs):
        z=inputs[self.layer]
        shape=z.shape
        pred=self.model(z.reshape(shape[0],-1,shape[-1]),inputs['action'])
        return {'delta_'+self.layer:pred.reshape(shape)}


def make_model(variant,input_rms=None,gradient_checkpointing=False):
    if variant in ('J','J_equal','JC'):return Joint36M(input_rms,gradient_checkpointing)
    return ReferenceSingle({'C0':'connector','S2':'block_02','S8':'block_08'}[variant])
