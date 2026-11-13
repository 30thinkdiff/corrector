"""Frozen train-only normalization; separate supervised/monitor/guidance objectives."""
from dataclasses import dataclass,asdict
import torch
import math
import torch.nn.functional as F
from .model import LAYERS


@dataclass(frozen=True)
class LossConfig:
    beta: float=1.
    weights: tuple=(.25,.25,.50)
    eps: float=1e-12
    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.beta,self.eps,*self.weights)) or self.eps<=0 or self.beta<0 or len(self.weights)!=3 or min(self.weights)<0 or abs(sum(self.weights)-1)>1e-8:
            raise ValueError('invalid frozen loss configuration')
    def serializable(self):return asdict(self)


def objective(pred,targets,scales,cfg=LossConfig()):
    rows={};total=next(iter(pred.values())).new_zeros((),dtype=torch.float32)
    selected=[k for k in LAYERS if 'delta_'+k in pred]
    denom=sum(cfg.weights[LAYERS.index(k)] for k in selected)
    if denom<=0:raise ValueError('no supervised head')
    for k in selected:
        key='delta_'+k;p,y=pred[key].float(),targets[key].float()
        if p.shape!=y.shape:raise ValueError('no target broadcasting: '+key)
        mse=(p-y).square().mean();norm=mse/(scales[key]**2+cfg.eps)
        pf,yf=p.flatten(1),y.flatten(1)
        valid=(pf.norm(dim=1)>1e-8)&(yf.norm(dim=1)>1e-8)
        cosine=(1-F.cosine_similarity(pf[valid],yf[valid],dim=-1)).mean() if valid.any() else p.sum()*0
        value=norm+cfg.beta*cosine
        if not torch.isfinite(value):raise FloatingPointError(key)
        rows[k]={'mse':mse,'normalized_mse':norm,'cosine_loss':cosine,'loss':value}
        total=total+cfg.weights[LAYERS.index(k)]/denom*value
    return total,rows


def shared_gradient_audit(rows,model):
    """Use shared fusion matrix for J; explicit probe rather than all-parameter copies."""
    parameter=model.fusion.weight
    grads={k:torch.autograd.grad(v['loss'],parameter,retain_graph=True)[0].detach().float().flatten() for k,v in rows.items()}
    return {'parameter':'fusion.weight','norms':{k:float(g.norm()) for k,g in grads.items()},
            'cosines':{a+'/'+b:float(F.cosine_similarity(grads[a],grads[b],dim=0)) for i,a in enumerate(grads) for b in list(grads)[i+1:]}}
