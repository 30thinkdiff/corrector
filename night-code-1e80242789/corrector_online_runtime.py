"""LeRobot0.6.1实例级LVM/OGG适配；源状态机复用，单环境、普通动作队列。"""
from __future__ import annotations

import ast
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import torch
import corrector_models as models
try:
    from studies.policy_generalization.corrector.online_practice import guided_velocity, visual_mismatch
except ModuleNotFoundError as error:
    if not error.name.startswith('studies'): raise
    from corrector_online_practice import guided_velocity, visual_mismatch


def student_should_interrupt(method, triggered):
    """monitor只记分；lvm/ogg在触发时清空待执行队列。始终返回bool。"""
    if method == 'monitor': return False
    if method in {'lvm','ogg'}: return bool(triggered)
    # H50/H20走普通策略，不进入Corrector；拼错在线模式时及时报错。
    raise ValueError(f'未知Corrector模式：{method!r}')


def student_correction_target(expected, observed):
    """同形状[B,L,D]的期望/实测视觉变化→纠偏目标；不是未来真值。"""
    if not isinstance(expected, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError('期望变化和实测变化应为Tensor')
    if expected.shape != observed.shape or expected.ndim != 3 or expected.numel() == 0:
        raise ValueError('需要同形状的非空[B,L,D]，避免减法静默广播')
    if (expected.device != observed.device or not expected.is_floating_point()
            or not observed.is_floating_point()):
        raise ValueError('视觉变化应为同设备浮点Tensor')
    return expected - observed


def decisions(implementation):
    if implementation == 'reference':
        return lambda method, triggered: method in {'lvm','ogg'} and bool(triggered), lambda e,o:e-o
    if implementation != 'student': raise ValueError('未知控制实现')
    # 启动GPU加载前检查空位；明确选择reference才使用对照，不隐式补答案。
    student_should_interrupt('monitor',True)
    student_correction_target(torch.zeros(1,1,1),torch.zeros(1,1,1))
    return student_should_interrupt,student_correction_target


def monitor_api(source_root):
    import corrector_data_workbench as archive
    archive.ensure_source(source_root)
    path=source_root/'src/lerobot/scripts/lerobot_eval_modified_detection.py'
    tree=ast.parse(path.read_text())
    names={'RobustThresholdState','robust_threshold_step'}
    body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in body}!=names:raise RuntimeError('固定上游缺少监控状态机')
    name='_fixed_corrector_threshold'
    module=types.ModuleType(name);sys.modules[name]=module
    module.__dict__.update(np=np,dataclass=dataclass,field=field)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*body],type_ignores=[])),str(path),'exec'),module.__dict__)
    return module.RobustThresholdState,module.robust_threshold_step,archive.digest(path)


def load_predictor(checkpoint,source_root,device):
    import corrector_train as train
    config,_,cls,_,hashes=train.reference_api(source_root)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    protocol=saved.get('feature_protocol')
    if not protocol or protocol['source_files']!=models.source_hashes(hashes,source_root,protocol.get('model_type','mlp'),train.archive.digest):raise RuntimeError('需要带完整来源协议的best_model或评估快照')
    if protocol.get('encoder_policy_revision',train.archive.POLICY_SHA)!=train.archive.POLICY_SHA:
        raise RuntimeError('预测器视觉编码器与当前冻结VLA版本不一致')
    if 'encoder_policy_revision' not in protocol:
        protocol={**protocol,'encoder_policy_revision':train.archive.POLICY_SHA,'legacy_encoder_revision_inferred':True}
    cls,cfg=models.from_protocol(protocol,config,cls,source_root)
    model=cls(cfg).to(device)
    model.load_state_dict(saved['state_dict'],strict=True);model.eval().requires_grad_(False)
    return model,protocol


class OnlineCorrector:
    def __init__(self,policy,predictor,protocol,method,source_root,directory,*,fps=20,stride=1,
                 eta=1.,guidance_every=3,implementation='reference',trace=True,event_tensors=False,
                 force_step=-1,cooldown=0):
        if method not in {'monitor','lvm','ogg'}:raise ValueError('未知Corrector模式')
        self.interrupt,self.target=decisions(implementation)
        State,self.threshold_step,self.monitor_sha=monitor_api(source_root)
        self.state=State(cooldown_steps=cooldown)
        self.policy,self.predictor,self.method=policy,predictor,method
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        physical_steps=protocol['prediction_seconds']*fps
        if not np.isclose(physical_steps,round(physical_steps)):
            raise ValueError('预测跨度不能对应整数控制步；需要基于时间戳的另一套调度')
        self.k=int(round(physical_steps));self.stride=stride
        if self.k<1 or stride<1 or self.k%stride:raise ValueError('跨度须为monitor-stride的正整数倍')
        if getattr(policy.config,'rtc_config',None) is not None:raise ValueError('首版不支持RTC异步动作拼接')
        self.h_window=int(protocol['h_window'])
        if self.h_window<1 or predictor.cfg.action_dim!=7:raise ValueError('需要正历史长度/7维动作')
        if self.h_window>1 and not getattr(predictor,'uses_history',False):raise ValueError('多帧历史需Transformer预测器')
        ratio=fps/float(protocol.get('fps',fps))
        if not np.isclose(ratio,round(ratio)) or ratio<1:raise ValueError('历史帧间距不能对应整数控制步')
        self.history_stride=int(round(ratio))
        self.warmup_steps=(max(self.h_window,int(protocol.get('anchor_history',1)))-1)*self.history_stride
        if self.history_stride%stride and self.h_window>1:raise ValueError('监测stride必须整除训练历史帧间距')
        self.history=deque(maxlen=(self.h_window-1)*self.history_stride+1)
        self.context_now=None
        self.eta,self.every=eta,guidance_every
        if self.every<1 or (method=='ogg' and eta<=0):raise ValueError('OGG需要正eta和引导间隔')
        self.trace,self.event_tensors,self.force_step=trace,event_tensors,force_step
        if force_step>=0 and (force_step<self.k or force_step%stride):raise ValueError('强制事件须在预测到期后并对齐monitor-stride')
        self.pending=deque();self.step=0;self.monitor_s=0.;self.events=0;self.guidance_calls=0
        self.active=False;self.z_now=None;self.target_now=None;self.denoise_index=0
        self.original_select=policy.select_action;self.original_denoise=policy.model.denoise_step

    def log(self,row):
        if self.trace:
            with (self.directory/'monitor_steps.jsonl').open('a') as file:
                file.write(json.dumps(row,allow_nan=False)+'\n');file.flush()

    @torch.no_grad()
    def encode(self,batch):
        if len(self.policy.config.image_features)!=2 or any(k not in batch for k in self.policy.config.image_features):
            raise ValueError('需要与训练相同的两路相机顺序')
        images,masks=self.policy.prepare_images(batch)
        if len(images)!=2 or any(not bool(m.all()) for m in masks):raise ValueError('缺失相机不能静默补图')
        z=torch.cat([self.policy.model.vlm_with_expert.embed_image(img) for img in images],dim=1).float()
        if z.shape[0]!=1 or z.shape[-1]!=self.predictor.cfg.token_dim:raise ValueError('当前首版要求单环境且token维度一致')
        if not torch.isfinite(z).all():raise FloatingPointError('在线视觉特征非有限')
        return z.detach()

    def select_action(self,batch,*args,**kwargs):
        # eval_policy外层是inference_mode：局部退出，再用no_grad限制VLA图；OGG单独enable_grad。
        with torch.inference_mode(False),torch.no_grad():
            batch={k:(v.clone() if isinstance(v,torch.Tensor) and v.is_inference() else v) for k,v in batch.items()}
            started=time.perf_counter();z=None;row={'step':self.step,'score':None,'threshold':None,'trigger':False}
            if self.step%self.stride==0:
                z=self.encode(batch);self.z_now=z
                self.history.append((self.step,z))
                self.context_now=models.history_at(self.history,self.step,self.h_window,self.history_stride)
                if self.pending and self.pending[0][0]<self.step:raise RuntimeError('预测到期队列时间错位')
                if self.pending and self.pending[0][0]==self.step:
                    _,origin,expected=self.pending.popleft();observed=z-origin
                    score=float(visual_mismatch(expected,observed).item())
                    if not np.isfinite(score):raise FloatingPointError('在线偏离分数非有限')
                    threshold,trigger=self.threshold_step(score,self.state)
                    forced=self.step==self.force_step
                    row.update(score=score,threshold=threshold,trigger=bool(trigger),debug_forced=forced,
                               observed_norm=float(observed.norm()),expected_norm=float(expected.norm()))
                    if self.interrupt(self.method,trigger or forced):
                        discarded=len(self.policy._queues['action'])
                        self.policy._queues['action'].clear()  # 缓存只在本实例；不reset监控历史。
                        self.events+=1;row.update(interrupted=True,discarded_actions=discarded)
                        self.target_now=self.target(expected,observed).detach()
                        self.active=self.method=='ogg';self.denoise_index=0
                        if self.event_tensors:
                            torch.save({'z_now':z.cpu(),'expected':expected.cpu(),'observed':observed.cpu(),
                                        'target':self.target_now.cpu()},self.directory/f'event_{self.step:04d}.pt')
                if self.step==self.force_step and row['score'] is None:raise ValueError('强制事件时尚无到期预测，请增大force-step并对齐stride')
            if z is not None and z.is_cuda:torch.cuda.synchronize(z.device)
            self.monitor_s+=time.perf_counter()-started
            try:action=self.original_select(batch,*args,**kwargs)
            finally:self.active=False  # OGG只对事件后本次查询生效，不泄漏到后续普通块。
            if z is not None and self.context_now is not None and self.step>=self.warmup_steps:
                tick=time.perf_counter()
                context=self.context_now if getattr(self.predictor,'uses_history',False) else z
                predicted=self.predictor(context,action[:,:self.predictor.cfg.action_dim].float()).detach()
                if not torch.isfinite(predicted).all():raise FloatingPointError('预测器输出非有限')
                self.pending.append((self.step+self.k,z,predicted))
                if z.is_cuda:torch.cuda.synchronize(z.device)
                self.monitor_s+=time.perf_counter()-tick
            self.log(row);self.step+=1
            return action

    def denoise_step(self,prefix_pad_masks,past_key_values,x_t,timestep):
        v=self.original_denoise(prefix_pad_masks,past_key_values,x_t,timestep)
        if self.active and self.denoise_index%self.every==0:
            if getattr(self.predictor,'uses_history',False):
                v,loss=models.guided_history_velocity(self.predictor,self.context_now,x_t,v,timestep,self.target_now,eta=self.eta)
            else:
                v,loss=guided_velocity(self.predictor,self.z_now,x_t,v,timestep,self.target_now,eta=self.eta)
            self.guidance_calls+=1;self.log({'step':self.step,'denoise':self.denoise_index,'guidance_loss':loss})
        self.denoise_index+=1
        return v

    @contextmanager
    def installed(self):
        with patch.object(self.policy,'select_action',self.select_action),patch.object(self.policy.model,'denoise_step',self.denoise_step):
            yield self
