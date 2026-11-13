"""固定参考分位数规则下的规划/已执行动作LVM，只有普通重规划，无OGG。"""
from collections import deque
from contextlib import contextmanager
import math
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from corrector_online_runtime import OnlineCorrector
from corrector_executed_monitor import ExecutedHistory,sync
from corrector_models import planned_actions


class FrozenTriggerRule:
    def __init__(self,config):
        self.config=dict(config);self.ewma=None;self.count=0;self.last_step=None;self.next_allowed=0
        if not 0<config['alpha']<=1 or not 0<config['threshold']<2:
            raise ValueError('规则alpha/threshold不合法')
        if config['consecutive']<1 or config['cooldown_steps']<1:raise ValueError('触发持续/冷却应为正')

    def receive(self,score,step):
        if not math.isfinite(score):raise ValueError('非有限分数')
        if self.last_step is not None and step<=self.last_step:raise ValueError('评分时钟倒退')
        if self.last_step is not None and step!=self.last_step+1:self.count=0
        self.last_step=step
        alpha=self.config['alpha'];self.ewma=score if self.ewma is None else alpha*score+(1-alpha)*self.ewma
        self.count=self.count+1 if self.ewma>self.config['threshold'] and step>=self.next_allowed else 0
        triggered=self.count>=self.config['consecutive']
        if triggered:self.count=0;self.next_allowed=step+self.config['cooldown_steps']
        return triggered


class ConsistencyLVM(OnlineCorrector):
    def __init__(self,policy,predictor,protocol,source_root,directory,mode,rule,normalizer):
        super().__init__(policy,predictor,protocol,'monitor',source_root,directory,
                         fps=20,stride=1,implementation='reference',trace=True)
        if mode not in {'planned_lvm','executed_lvm'}:raise ValueError('未知闭环模式')
        if self.k!=20 or self.h_window!=1 or self.history_stride!=2 or self.action_steps!=10:
            raise ValueError('必须保持原A10-H1/1秒/20步协议')
        self.mode=mode;self.rule=FrozenTriggerRule(rule);self.normalizer=normalizer
        self.executed=ExecutedHistory(predictor) if mode=='executed_lvm' else None
        self.pending=deque();self.raw_commands=[];self.selected=None;self.in_reset=False;self.outer_action=None
        self.settling_commands=0;self.original_reset=policy.reset;self.resets=0
        self.encoding_s=0.;self.predictor_s=0.;self.submissions=0

    def reset(self):
        if self.resets or self.step:raise RuntimeError('请为每个回合创建新闭环实例')
        self.original_reset();self.pending.clear();self.resets+=1
        if self.executed:self.executed.reset()

    def select_action(self,batch,*a,**kw):
        if self.selected is not None:raise RuntimeError('上一动作尚未实际提交')
        with torch.inference_mode(False),torch.no_grad():
            batch={k:v.clone() if isinstance(v,torch.Tensor) and v.is_inference() else v for k,v in batch.items()}
            started=time.perf_counter();before=time.perf_counter();z=super().encode(batch);sync(z)
            self.encoding_s+=time.perf_counter()-before
            row={'step':self.step,'mode':self.mode,'score':None,'trigger':False,'threshold':self.rule.config['threshold']}
            if self.mode=='executed_lvm':
                before=time.perf_counter();data=self.executed.observe(self.step,z);sync(z)
                self.predictor_s+=time.perf_counter()-before
                if self.step>=24:row.update(score=data['executed_score'],origin_step=data['origin_step'],action_indices=data['action_indices'])
            elif self.pending and self.pending[0][0]<=self.step:
                due,origin,expected=self.pending.popleft()
                if due!=self.step:raise RuntimeError('规划预测到期时钟错位')
                observed=z-origin
                row['score']=float((1-F.cosine_similarity(expected.flatten(1),observed.flatten(1),dim=1)).item())
            if row['score'] is not None:
                row['trigger']=self.rule.receive(row['score'],self.step)
                row.update(ewma=self.rule.ewma,next_allowed=self.rule.next_allowed)
            if row['trigger']:
                row['discarded_actions']=len(self.policy._queues['action'])
                self.policy._queues['action'].clear();self.events+=1
                if self.mode=='planned_lvm':
                    row['cancelled_on_interrupt']=len(self.pending)
                    self.cancelled_predictions+=len(self.pending);self.pending.clear()
                # 已执行历史必须保留，换未来规划不会让真实过去失效。
            queue_was_empty=not bool(self.policy._queues['action'])
            sync(z);self.monitor_s+=time.perf_counter()-started
            action=self.original_select(batch,*a,**kw)
            if self.mode=='planned_lvm':
                if queue_was_empty:
                    row['cancelled_on_new_plan']=len(self.pending)
                    self.cancelled_predictions+=len(self.pending);self.pending.clear()
                if self.step>=4:
                    condition=planned_actions(action,self.policy._queues['action'],10,2)
                    if condition is None:
                        row['prediction_skipped']='insufficient_planned_actions';self.sequence_skips+=1
                    else:
                        before=time.perf_counter();expected=self.predictor(z[:,None],condition).detach();sync(expected)
                        elapsed=time.perf_counter()-before;self.predictor_s+=elapsed;self.monitor_s+=elapsed
                        if expected.shape!=z.shape or not torch.isfinite(expected).all():raise RuntimeError('规划预测shape/数值不符')
                        self.pending.append((self.step+20,z,expected))
            self.log(row);self.selected={'step':self.step,'policy_action':action.detach().cpu().reshape(-1).tolist()}
            self.step+=1
            return action

    def controller_step(self,original,action,*a,**kw):
        if self.in_reset:
            self.settling_commands+=1;return original(action,*a,**kw)
        if self.selected is None:raise RuntimeError('缺失对应策略输出的真实控制提交')
        raw=np.asarray(action,dtype=np.float32)
        if raw.shape!=(7,) or self.outer_action is None or not np.array_equal(raw,self.outer_action):
            raise RuntimeError('底层控制命令与外层输入不一致')
        step=self.selected['step']
        if step!=self.submissions:raise RuntimeError('真实提交动作步序不连续')
        normalized=self.normalizer(raw)
        if self.executed:self.executed.submit(step,normalized)
        record={'step':step,'raw_controller_action':raw.tolist(),
                'normalized_action':normalized.detach().cpu().reshape(-1).tolist(),'returned':False}
        self.raw_commands.append(record);self.selected=None;self.submissions+=1
        out=original(action,*a,**kw);record['returned']=True
        return out

    @contextmanager
    def installed(self):
        # 不安装OGG/denoise hook，不能意外改变生成器速度。
        with patch.object(self.policy,'select_action',self.select_action),patch.object(self.policy,'reset',self.reset):yield self
