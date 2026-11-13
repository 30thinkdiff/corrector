"""Multilayer bridge over the existing OnlineCorrector state machine/action queue.

No second rollout controller. Single environment, synchronous instance-scoped patch.
"""
from contextlib import contextmanager
import copy,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch import nn
import corrector_online_runtime as legacy
from corrector_multilayer import ReadOnlyVisionTap
from .model import LAYERS,SHAPES
from multilayer_mvp.data import CAMERAS

METHODS={'j_monitor':'monitor','j_lvm':'lvm','j_lvm_ogg':'ogg'}
SIZES={k:int(torch.tensor(SHAPES[k]).prod()) for k in LAYERS}


def pack(values):
    return torch.cat([values[k].reshape(values[k].shape[0],-1) for k in LAYERS],dim=1).unsqueeze(-1)


def unpack(value):
    if value.ndim!=3 or value.shape[1:]!=(sum(SIZES.values()),1):raise ValueError('invalid packed visual bridge')
    out={};start=0
    for k in LAYERS:
        out[k]=value[:,start:start+SIZES[k],0].reshape(value.shape[0],*SHAPES[k]);start+=SIZES[k]
    return out


class JointBridge(nn.Module):
    def __init__(self,joint):
        super().__init__();self.joint=joint;self.cfg=SimpleNamespace(token_dim=1,action_dim=7)
    def forward(self,z,action):
        predicted=self.joint(dict(unpack(z),action=action))
        return pack({k:predicted['delta_'+k] for k in LAYERS})


def layer_scores(expected,observed,calibration):
    details={};numerator=0.;denominator=0.
    for k in LAYERS:
        p,y=expected[k].float().flatten(1),observed[k].float().flatten(1)
        if p.shape!=y.shape:raise ValueError('layer mismatch')
        finite=bool(torch.isfinite(p).all() and torch.isfinite(y).all())
        pn=float(p.square().mean().sqrt()) if finite else None
        yn=float(y.square().mean().sqrt()) if finite else None
        floor=max(float(calibration['norm_floor'][k]),1e-8)
        valid=finite and pn>=floor and yn>=floor
        score=float((1-F.cosine_similarity(p,y,dim=1)).clamp(0,2).mean()) if valid else None
        details[k]={'valid':valid,'predicted_rms':pn,'observed_rms':yn,'score':score,'norm_floor':floor,
                    'reason':None if valid else 'nonfinite_or_low_norm'}
        if valid:
            weight=calibration['weights'][k];numerator+=weight*score;denominator+=weight
    return (numerator/denominator if denominator>0 else None),details


def guided_velocity(joint,context,x_t,velocity,timestep,targets,calibration,eta=1.):
    if joint.training or any(p.requires_grad for p in joint.parameters()):raise ValueError('frozen eval J required')
    if velocity.shape!=x_t.shape or velocity.ndim!=3 or velocity.shape[-1]<7:raise ValueError('flow action shape')
    if timestep.shape!=(velocity.shape[0],) or not torch.isfinite(timestep).all() or (timestep<0).any() or (timestep>1).any():raise ValueError('flow timestep')
    if not 0<eta<1e6:raise ValueError('invalid eta')
    with torch.inference_mode(False),torch.enable_grad():
        v=velocity.detach().clone().requires_grad_(True)
        action=(x_t.detach().clone()-timestep[:,None,None]*v)[:,0,:7]
        out=joint(dict({k:value.detach().clone() for k,value in context.items()},action=action))
        losses={};norms={}
        for k in LAYERS:
            target=targets[k].detach().float().flatten(1);pred=out['delta_'+k].float().flatten(1)
            floor=max(calibration['norm_floor'][k],1e-8)
            if (torch.isfinite(pred).all() and torch.isfinite(target).all()
                and float(pred.detach().square().mean().sqrt())>=floor and float(target.square().mean().sqrt())>=floor
                and calibration['weights'][k]>0):
                losses[k]=(1-F.cosine_similarity(pred,target,dim=-1)).mean()
        if not losses:return velocity.detach(),{'skipped':'no_reliable_correction_layer','gradient_norm':0.,'updated':False}
        denom=sum(calibration['weights'][k] for k in losses)
        total=sum(calibration['weights'][k]/denom*l for k,l in losses.items())
        gradients={k:torch.autograd.grad(l,v,retain_graph=True)[0] for k,l in losses.items()}
        gradient=torch.autograd.grad(total,v)[0]
        if not torch.isfinite(total) or not torch.isfinite(gradient).all():raise FloatingPointError('OGG loss/gradient')
        guided=(v-eta*gradient).detach()
        return guided,{'loss':float(total.detach()),'per_layer_loss':{k:float(l.detach()) for k,l in losses.items()},
                       'per_layer_gradient_norm':{k:float(g.norm()) for k,g in gradients.items()},
                       'gradient_cosines':{a+'/'+b:float(F.cosine_similarity(gradients[a].flatten(),gradients[b].flatten(),dim=0)) for i,a in enumerate(gradients) for b in list(gradients)[i+1:]},
                       'gradient_norm':float(gradient.norm()),'velocity_update_norm':float((guided-velocity).norm()),
                       'candidate_action_update_norm':float((timestep[:,None,None]*(guided-velocity))[:,:,:7].norm()),
                       'updated':bool(gradient.norm()>0),'valid_layers':list(losses)}


class JointOnlineCorrector(legacy.OnlineCorrector):
    def __init__(self,policy,joint,calibration,method,source_root,directory,*,checkpoint_sha,
                 fps=20,stride=1,eta=1.,guidance_every=3,force_step=-1):
        if method not in METHODS:raise ValueError(method)
        if list(policy.config.image_features)!=CAMERAS:raise ValueError('online camera sequence differs from training')
        if fps!=20:raise ValueError('formal online protocol requires 20Hz')
        if calibration.get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('calibration/model mismatch')
        if calibration.get('status')!='frozen' and method!='j_monitor':raise ValueError('calibrate normal monitor-only runs before interventions')
        if set(calibration['weights'])!=set(LAYERS) or min(calibration['weights'].values())<0 or sum(calibration['weights'].values())<=0:raise ValueError('invalid fusion weights')
        if not (Path(source_root)/'src/lerobot/scripts/lerobot_eval_modified_detection.py').is_file():
            raise FileNotFoundError('fixed source must already exist; no implicit fetch')
        self.calibration=copy.deepcopy(calibration);self.archive_method=method
        self.layer_details={};self.last_score=None;self.plan_id=0;self.last_cancelled=0
        bridge=JointBridge(joint).eval().requires_grad_(False)
        protocol={'prediction_seconds':1.,'fps':10.,'k_step':10,'h_window':1,'anchor_history':1}
        super().__init__(policy,bridge,protocol,METHODS[method],Path(source_root),directory,fps=fps,stride=stride,
                         eta=eta,guidance_every=guidance_every,implementation='reference',force_step=force_step)
        self.state_settings=calibration['state_settings']
        for key,value in self.state_settings.items():
            if key not in {'window_size','base_noise_floor','ewma_alpha','k_on','k_off','trigger_margin','jump_trigger',
                           'trigger_consecutive_steps','reset_consecutive_steps','cooldown_steps','hard_retrigger_margin','hard_retrigger_consecutive_steps'}:
                raise ValueError('unsupported threshold parameter')
            setattr(self.state,key,value)
        self.normal_history=list(calibration['normal_history'])
        if len(self.normal_history)<2:raise ValueError('normal-only calibration history required')
        self.state.bootstrapped=True;self.state.clean_history=self.normal_history.copy()
        threshold_step=self.threshold_step
        def frozen_threshold(score,state):
            if self.last_score is None:
                return (state.dynamic_threshold[-1] if state.dynamic_threshold else 0.),False
            state.clean_history=self.normal_history.copy()
            result=threshold_step(score,state)
            state.clean_history=self.normal_history.copy()  # test trajectories never refit threshold history
            return result
        self.threshold_step=frozen_threshold
        base_interrupt=self.interrupt
        def interrupt(mode,trigger):
            decision=base_interrupt(mode,trigger) and self.last_score is not None
            if decision:
                self.last_cancelled=len(self.pending);self.cancelled_predictions+=len(self.pending);self.pending.clear()
            return decision
        self.interrupt=interrupt
        original_select=self.original_select
        def select_with_plan_change(*args,**kwargs):
            if not self.policy._queues['action']:
                self.plan_id+=1
                # Current-action forecasts do not condition on the remainder of the old chunk.
                # Natural chunk refresh therefore preserves them, as in legacy single-action LVM.
                # Corrective interruptions explicitly cancel outstanding predictions above.
                self.log({'step':self.step,'event':'new_action_plan','plan_id':self.plan_id,
                          'preserved_single_action_predictions':len(self.pending)})
            return original_select(*args,**kwargs)
        self.original_select=select_with_plan_change

    @torch.no_grad()
    def encode(self,batch):
        images,masks=self.policy.prepare_images(batch)
        if len(images)!=2 or any(not bool(m.all()) for m in masks):raise ValueError('two complete cameras required')
        wrapper=self.policy.model.vlm_with_expert;views=[]
        with ReadOnlyVisionTap(wrapper,(2,8)) as tap:
            for key,image in zip(self.policy.config.image_features,images,strict=True):
                with tap.capture(key) as values:wrapper.embed_image(image)
                views.append(values)
        values={k:torch.stack([v[k].float() for v in views],dim=1) for k in LAYERS[:2]}
        values['connector']=torch.cat([v['connector'].float() for v in views],dim=1)
        for k in LAYERS:
            if values[k].shape!=(1,*SHAPES[k]) or not torch.isfinite(values[k]).all():raise ValueError('online full-resolution shape/numerics: '+k)
        return pack(values).detach()

    def _score(self,expected,observed):
        self.last_score,self.layer_details=layer_scores(unpack(expected),unpack(observed),self.calibration)
        return expected.new_tensor([self.last_score if self.last_score is not None else 0.])

    def select_action(self,*args,**kwargs):
        # Existing method owns due times, state machine, interruption and queue pop.
        self.last_score=None;self.layer_details={}
        with patch.object(legacy,'visual_mismatch',self._score):return super().select_action(*args,**kwargs)

    def log(self,row):
        row=dict(row,method=self.archive_method,plan_id=self.plan_id)
        if 'score' in row and self.step>=self.k:
            row.update(layer_scores=self.layer_details,score=self.last_score,
                       scoring_valid=self.last_score is not None,cancelled_on_interrupt=self.last_cancelled)
            if self.last_score is None:row['prediction_scoring_skipped']='no_reliable_layer' if self.layer_details else 'no_due_prediction'
        if row.get('interrupted'):
            targets=unpack(self.target_now)
            row['correction_targets']={k:{'shape':list(v.shape),'rms':float(v.float().square().mean().sqrt())} for k,v in targets.items()}
            path=self.directory/f'joint_target_{self.step:06d}.pt'
            if path.exists():raise FileExistsError('event target already published')
            torch.save({k:v.detach().cpu() for k,v in targets.items()},path)
            row['correction_target_file']=path.name
        super().log(row)

    def denoise_step(self,prefix_pad_masks,past_key_values,x_t,timestep):
        v=self.original_denoise(prefix_pad_masks,past_key_values,x_t,timestep)
        if self.active and self.denoise_index%self.every==0:
            if v.is_cuda:torch.cuda.synchronize(v.device)
            tick=time.perf_counter()
            v,report=guided_velocity(self.predictor.joint,unpack(self.z_now),x_t,v,timestep,
                                     unpack(self.target_now),self.calibration,self.eta)
            if v.is_cuda:torch.cuda.synchronize(v.device)
            self.guidance_calls+=int(report['updated'])
            self.log({'step':self.step,'denoise':self.denoise_index,'event':'joint_ogg',**report,'latency_s':time.perf_counter()-tick})
        self.denoise_index+=1
        return v
