"""同一真实轨迹上的五模型诊断；候选动作在生成当时采集，推理延后批量进行。"""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def prediction_metrics(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 3 or not torch.isfinite(predicted).all():
        raise ValueError('预测shape/数值异常')
    p,t=predicted.flatten(1).float(),target.flatten(1).float()
    mse=(p-t).square().mean(1);energy=t.square().mean(1)
    return {'mse':mse,'target_energy':energy,'cosine_loss':1-F.cosine_similarity(p,t,dim=1),
            'pred_energy':p.square().mean(1)}


def model_inputs(frames, actions, origins, h_window, action_steps):
    # 当前观测是origin；未来真实图像只出现在target，不进入模型输入。
    visual=torch.stack([frames[[t-2*(h_window-1-i) for i in range(h_window)]] for t in origins])
    condition=torch.stack([actions[t:t+20:2] if action_steps==10 else actions[t] for t in origins])
    target=torch.stack([frames[t+20]-frames[t] for t in origins])
    if action_steps==10 and condition.shape[1:]!=(10,7):raise ValueError('真实历史动作不足')
    return visual,condition,target


class TrajectoryDiagnostics:
    def __init__(self,models,protocols,directory):
        self.models=models;self.protocols=protocols;self.directory=Path(directory)
        self.frames=[];self.actions=[];self.plans={};self.serials=[];self.selected=None
        self.capture_s=0.;self.last_z=None

    def observe(self,z):
        if len(self.frames)!=len(self.actions):raise RuntimeError('观测/实际提交时序错位')
        if z.shape!=(1,128,960):raise ValueError('视觉特征协议变化')
        self.last_z=z.detach();self.frames.append(z[0].detach().cpu().float().clone())

    def submit(self,normalized,selected,queue,serial):
        t=len(self.actions)
        if len(self.frames)!=t+1 or normalized.shape!=(1,7):raise RuntimeError('真实提交动作缺失')
        self.actions.append(normalized[0].detach().cpu().float().clone());self.serials.append(serial)
        if t>=4 and t%2==0:
            values=[selected.detach().cpu().float(),*[a.detach().cpu().float() for a in queue]]
            current=values[0].reshape(7)
            sequence=torch.stack([values[i].reshape(7) for i in range(0,19,2)]) if len(values)>=19 else None
            self.plans[t]={'single':current,'sequence':sequence,'serial':serial}

    @torch.no_grad()
    def evaluate(self,audit_sample=False):
        if len(self.frames)!=len(self.actions) or not self.actions:raise RuntimeError('轨迹未完整提交')
        frames=torch.stack(self.frames);actions=torch.stack(self.actions);n=len(actions)
        origins=list(range(4,n-20,2));summaries=[];benchmark=[]
        shared_planned={t for t in origins if self.plans[t]['sequence'] is not None
            and all(self.serials[j]==self.plans[t]['serial'] for j in range(t,t+20,2))
            and float((self.plans[t]['sequence']-actions[t:t+20:2]).abs().max())<=1e-5}
        with gzip.open(self.directory/'action_conditions.json.gz','wt') as output:
            json.dump({'actual_normalized_commands':actions.tolist(),'plan_serial_by_step':self.serials,
                'candidates_at_origin':{str(t):{'single':p['single'].tolist(),
                    'sequence':p['sequence'].tolist() if p['sequence'] is not None else None,
                    'serial':p['serial']} for t,p in self.plans.items()}},output)
        windows=self.directory/'prediction_windows.jsonl.gz'
        with gzip.open(windows,'wt') as output:
            for name,model in self.models.items():
                protocol=self.protocols[name];h=int(protocol['h_window']);a=int(protocol.get('action_steps',1))
                if h not in (1,3) or a not in (1,10):raise ValueError('模型输入协议不支持，禁止猜测')
                device=next(model.parameters()).device;aggregates=defaultdict(list)
                for role in ['executed','planned']:
                    allowed=[t for t in origins if role=='executed' or a==1 or self.plans[t]['sequence'] is not None]
                    batch_size=64;cursor=0;latencies=[]
                    while cursor<len(allowed):
                        ts=allowed[cursor:cursor+batch_size]
                        try:
                            visual,condition,target=model_inputs(frames,actions,ts,h,a)
                            if role=='planned':condition=torch.stack([self.plans[t]['sequence' if a==10 else 'single'] for t in ts])
                            x=visual if getattr(model,'uses_history',False) else visual[:,0]
                            x,condition,target=x.to(device),condition.to(device),target.to(device)
                            if device.type=='cuda':torch.cuda.synchronize(device)
                            start=time.perf_counter();predicted=model(x,condition)
                            if device.type=='cuda':torch.cuda.synchronize(device)
                            elapsed=time.perf_counter()-start;latencies.append({'batch':len(ts),'seconds':elapsed})
                            metrics={k:v.cpu().tolist() for k,v in prediction_metrics(predicted,target).items()}
                        except torch.cuda.OutOfMemoryError:
                            if batch_size==1:raise
                            x=condition=target=predicted=None
                            batch_size=max(1,batch_size//2);torch.cuda.empty_cache();continue
                        for i,t in enumerate(ts):
                            row={'origin':t,'due':t+20,'model':name,'role':role,**{k:v[i] for k,v in metrics.items()}}
                            row['relative_mse']=row['mse']/row['target_energy'] if row['target_energy']>1e-12 else None
                            if role=='planned':
                                actual=actions[t:t+20:2] if a==10 else actions[t]
                                candidate=self.plans[t]['sequence' if a==10 else 'single']
                                row['condition_max_abs_difference']=float((candidate-actual).abs().max())
                                row['same_plan']=a==1 or all(self.serials[j]==self.plans[t]['serial'] for j in range(t,t+20,2))
                                row['condition_honored']=row['same_plan'] and row['condition_max_abs_difference']<=1e-5
                            else:row['condition_honored']=True
                            category=role if role=='executed' else ('planned_honored' if row['condition_honored'] else 'planned_changed')
                            for scope in ['full']+(['early'] if t<=44 else []):
                                aggregates[(category,scope)].append(row)
                                if role=='planned' and t in shared_planned:aggregates[('planned_shared_honored',scope)].append(row)
                            output.write(json.dumps(row,allow_nan=False)+'\n')
                        if audit_sample and role=='executed' and cursor==0 and ts:
                            # 每个模型同一真实窗口的可复算输入、目标及输出，只有新生成的小样本。
                            np.savez_compressed(self.directory/f'audit_{name}.npz',visual=visual[0].numpy(),
                                action=condition[0].cpu().numpy(),target=target[0].cpu().numpy(),prediction=predicted[0].cpu().numpy())
                        cursor+=len(ts)
                    if allowed:
                        # 独立同步batch1测速；不把批量吞吐冒充在线单步时延，首两次作为预热。
                        v,c,_=model_inputs(frames,actions,[allowed[0]],h,a)
                        if role=='planned':c=torch.stack([self.plans[allowed[0]]['sequence' if a==10 else 'single']])
                        v=(v if getattr(model,'uses_history',False) else v[:,0]).to(device);c=c.to(device)
                        single=[]
                        for repeat in range(7):
                            if device.type=='cuda':torch.cuda.synchronize(device)
                            began=time.perf_counter();model(v,c)
                            if device.type=='cuda':torch.cuda.synchronize(device)
                            if repeat>=2:single.append(time.perf_counter()-began)
                        benchmark.append({'model':name,'role':role,'batch_size_selected':batch_size,
                            'batch_calls':latencies,'synchronized_batch1_seconds':single,'warmup_excluded':2})
                for (role,scope),sub in aggregates.items():
                    row={'model':name,'condition_role':role,'scope':scope,'windows':len(sub),
                         'early_prefix_complete':n>=65,'eligible_windows':len(origins) if scope=='full' else len([t for t in origins if t<=44])}
                    for key in ['mse','cosine_loss','target_energy','pred_energy','relative_mse']:
                        valid=[s[key] for s in sub if s[key] is not None];row[key]=sum(valid)/len(valid) if valid else None
                    summaries.append(row)
        (self.directory/'prediction_summary.json').write_text(json.dumps({'models':summaries,'latencies':benchmark,
            'window_sampling':'origin4,6,...; due=origin+20; common history uses origin-4,-2,0',
            'diagnostic_timing':'deferred batch inference after rollout; never drives control',
            'feature_memory_bytes':frames.numel()*4,'windows_sha256':hashlib.sha256(windows.read_bytes()).hexdigest()},indent=2)+'\n')
        return summaries
