"""被动到期一致性监测：Z_t先于u_t，历史条件只能来自已提交的控制命令。"""
from collections import deque
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from corrector_online_runtime import OnlineCorrector


def sync(value):
    if value.is_cuda:torch.cuda.synchronize(value.device)


@contextmanager
def lazy_controller_capture(unwrapped,runtime):
    """LIBERO第一次reset才创建_env；在reset返回前挂钩，首条策略动作前已生效。"""
    original_reset=unwrapped.reset
    hooked=set()
    with ExitStack() as hooks:
        def reset(*a,**kw):
            runtime.in_reset=True
            try:
                out=original_reset(*a,**kw)
                raw_env=unwrapped._env
                if raw_env is None or not callable(getattr(raw_env,'step',None)):
                    raise RuntimeError('LIBERO reset后仍无控制器，停止，不猜测接口')
                if id(raw_env) not in hooked:
                    original_step=raw_env.step
                    def controller(action,*args,_original=original_step,**kwargs):
                        return runtime.controller_step(_original,action,*args,**kwargs)
                    hooks.enter_context(patch.object(raw_env,'step',controller))
                    hooked.add(id(raw_env))
                return out
            finally:
                runtime.in_reset=False
        with patch.object(unwrapped,'reset',reset):yield


class ExecutedHistory:
    """20Hz观测/控制时钟；10Hz条件取origin+[0,2,...,18]，不含当前动作。"""
    def __init__(self,predictor):
        self.predictor=predictor
        self.reset()

    def reset(self):
        self.frames=deque(maxlen=21);self.actions=deque(maxlen=20)
        self.observations=0;self.submitted=0;self.latest=None

    def submit(self,step,action):
        if step!=self.submitted or self.observations!=step+1:
            raise RuntimeError('控制提交必须位于本步观测之后，且不能重复/缺失')
        if action.shape!=(1,7) or not torch.isfinite(action).all():
            raise ValueError('必须记录真实提交动作的归一化[1,7]，不补造历史')
        self.actions.append((step,action.detach().clone()))
        self.submitted+=1

    @torch.no_grad()
    def observe(self,step,z):
        if step!=self.observations or self.submitted!=step:
            raise RuntimeError('Z_t必须在u_(t-1)提交完成后、u_t提交前观测')
        self.frames.append((step,z.detach().clone()));self.observations+=1
        self.latest={'step':step,'executed_score':None,'eligible_executed':step>=20,
                     'eligible_common':step>=24}
        if step<20:return self.latest
        origin=step-20;frames=dict(self.frames);actions=dict(self.actions)
        if origin not in frames or set(actions)!=set(range(origin,step)):
            raise RuntimeError('到期历史不完整，禁止填零/复制动作冒充真实历史')
        indices=list(range(origin,step,2))
        sequence=torch.stack([actions[i] for i in indices],dim=1)
        expected=self.predictor(frames[origin][:,None],sequence).detach()
        observed=z-frames[origin]
        if expected.shape!=observed.shape or not torch.isfinite(expected).all():
            raise RuntimeError('到期预测与真实视觉变化shape/数值不兼容')
        flat_e,flat_o=expected.flatten(1),observed.flatten(1)
        score=float((1-F.cosine_similarity(flat_e,flat_o,dim=1)).item())
        self.latest.update(origin_step=origin,action_indices=indices,action_window=[origin,step-1],
            prediction_due_step=step,executed_score=score,expected_norm=float(flat_e.norm()),
            observed_norm=float(flat_o.norm()),dot=float((flat_e*flat_o).sum()),
            origin_z_sha256=hashlib.sha256(frames[origin].cpu().contiguous().numpy().tobytes()).hexdigest(),
            observed_z_sha256=hashlib.sha256(z.cpu().contiguous().numpy().tobytes()).hexdigest())
        return self.latest


class PassiveExecutedMonitor(OnlineCorrector):
    def __init__(self,policy,predictor,protocol,source_root,directory,normalizer,baseline_actions):
        super().__init__(policy,predictor,protocol,'monitor',source_root,directory,
                         fps=20,stride=1,implementation='reference',trace=True)
        if protocol.get('action_steps')!=10 or protocol['h_window']!=1 or self.k!=20 or self.history_stride!=2:
            raise ValueError('本Pilot只接受冻结A10-H1、1秒/20步协议')
        # 不运行旧LVM阈值状态机，更不把它的阈值沿用为新评分的判据。
        self.threshold_step=lambda score,state:(None,False)
        self.executed=ExecutedHistory(predictor)
        self.normalizer=normalizer;self.baseline_actions=baseline_actions
        self.selected=None;self.in_reset=False;self.raw_commands=[];self.outer_action=None
        self.predictor_times=[];self.encoding_times=[];self.normalization_times=[]
        self.planned_conditions=[]
        self.phase='planned';self.reset_count=0;self.settling_commands=0
        self.original_reset=policy.reset;self.original_forward=predictor.forward
        self.feature_path=self.directory/'observed_features.f32'
        self.feature_file=self.feature_path.open('xb')

    def reset(self):
        # 官方每回合新建本实例。意外多回合复用拒绝，而不静默混合/覆盖采集文件。
        if self.reset_count or self.step or self.raw_commands:
            raise RuntimeError('单回合采集实例意外重置；请为下一回合创建新实例')
        self.original_reset();self.executed.reset();self.history.clear();self.pending.clear()
        self.reset_count+=1

    def timed_forward(self,visual,action):
        sync(visual);began=time.perf_counter()
        out=self.original_forward(visual,action)
        sync(out)
        self.predictor_times.append({'step':self.step,'role':self.phase,'seconds':time.perf_counter()-began})
        if self.phase=='planned':
            self.planned_conditions.append({'origin_step':self.step,'due_step':self.step+20,
                'offsets':list(range(0,20,2)),'normalized_actions':action.detach().cpu()[0].tolist()})
        return out

    @torch.no_grad()
    def encode(self,batch):
        started=time.perf_counter();z=super().encode(batch);sync(z)
        self.encoding_times.append({'step':self.step,'seconds':time.perf_counter()-started})
        if z.shape!=(1,128,960):raise RuntimeError('实际视觉特征shape变化')
        self.feature_file.write(z.detach().cpu().contiguous().numpy().tobytes())
        self.phase='executed'
        try:self.executed.observe(self.step,z)
        finally:self.phase='planned'
        return z

    def select_action(self,batch,*a,**kw):
        if self.selected is not None:raise RuntimeError('上一条动作尚未提交控制器')
        step=self.step
        action=super().select_action(batch,*a,**kw)
        self.selected={'step':step,'policy_normalized':action.detach().clone()}
        if self.events or self.guidance_calls or self.active:
            raise RuntimeError('monitor-only意外产生中断或OGG')
        return action

    def log(self,row):
        if 'score' not in row:return super().log(row)
        if row.get('interrupted') or row.get('trigger') or row.get('threshold') is not None:
            raise RuntimeError('被动监测不允许沿用旧阈值/触发中断')
        row={**row,'planned_expected_norm':row.get('expected_norm'),
             'planned_observed_norm':row.get('observed_norm'),
             **self.executed.latest,'planned_score':row['score'],
             'threshold_applied':False,'interruptions_allowed':False,'ogg_allowed':False}
        super().log(row)

    def controller_step(self,original,action,*a,**kw):
        if self.in_reset:
            self.settling_commands+=1
            return original(action,*a,**kw)
        if self.selected is None:raise RuntimeError('没有对应观测/策略动作的控制提交')
        step=self.selected['step']
        raw=np.array(action,dtype=np.float32,copy=True)
        if raw.shape!=(7,) or not np.isfinite(raw).all():raise RuntimeError('实际控制器输入非有限[7]')
        if self.outer_action is None or not np.array_equal(raw,self.outer_action):
            raise RuntimeError('外层动作与真实控制器提交不一致，先报告接口问题')
        # 整条轨迹与此前纯H50逐值对照，发现改动在调用物理控制前停止。
        if step>=len(self.baseline_actions) or raw.tolist()!=self.baseline_actions[step]:
            raise RuntimeError(f'monitor-only在step{step}改变了已核验H50动作，不继续控制')
        began=time.perf_counter();normalized=self.normalizer(raw)
        sync(normalized);self.normalization_times.append({'step':step,'seconds':time.perf_counter()-began})
        self.executed.submit(step,normalized)
        entry={'step':step,'simulation_seconds':step/20,'raw_controller_action':raw.tolist(),
               'processor_normalized_action':normalized.detach().cpu().reshape(-1).tolist(),
               'policy_normalized_action':self.selected['policy_normalized'].detach().cpu().reshape(-1).tolist(),
               'submitted_wall_seconds':time.perf_counter(),'returned':False}
        self.raw_commands.append(entry);self.selected=None
        out=original(action,*a,**kw)
        entry['returned']=True
        return out

    @contextmanager
    def installed(self):
        with super().installed(), patch.object(self.policy,'reset',self.reset), \
             patch.object(self.predictor,'forward',self.timed_forward):
            yield self

    def close(self):
        self.feature_file.flush();self.feature_file.close()
        if self.step!=len(self.raw_commands) or self.step!=len(self.baseline_actions):
            raise RuntimeError('回合长度/控制提交与纯H50参考不一致')
        if self.selected is not None or not all(r['returned'] for r in self.raw_commands):
            raise RuntimeError('存在没有完成控制提交的动作')
        if self.feature_path.stat().st_size!=self.step*128*960*4:
            raise RuntimeError('真实视觉特征字节数与观测时刻不一致')
        write=lambda n,v:(self.directory/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
        write('controller_commands.json',self.raw_commands)
        write('predictor_latencies.json',self.predictor_times)
        write('shared_visual_latencies.json',self.encoding_times)
        write('normalization_latencies.json',self.normalization_times)
        write('planned_action_conditions.json',self.planned_conditions)
        write('feature_capture.json',{'path':str(self.feature_path),'dtype':'float32','shape':[self.step,128,960],
            'sha256':hashlib.sha256(self.feature_path.read_bytes()).hexdigest(),
            'observation_clock':'Z_t before u_t; after successful controller submission u_(t-1)',
            'controller_commands':self.step,'reset_settling_commands_excluded':self.settling_commands,
            'initial_reset_settling':'not instrumented; hook installed after lazy initialization and before first policy command',
            'reset_count':self.reset_count,'baseline_actions_exactly_equal':True})
