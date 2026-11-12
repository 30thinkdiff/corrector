"""独立记录包装：调用原OGG/监测算法，仅捕获实际数组，不替换冻结数学定义。"""
import hashlib
import json
from pathlib import Path
import time
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from corrector_online_runtime import OnlineCorrector
from corrector_hybrid import HybridCorrector


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditMixin:
    def __init__(self,*a,audit_tag='',**kw):
        super().__init__(*a,**kw)
        self.audit_tag=audit_tag;self.audit_s=0.;self.target_files=[];self.tensor_files=[]
        self.last_event=None;self.target=self.capture_target;self._capture_guide=False;self._guide_row=None
        self.forecast_inputs={};self.guide_seconds=0.

    @contextmanager
    def installed(self):
        def forecast(module,inputs,output):
            if torch.is_grad_enabled():return  # OGG候选求导不能冒充原检测预测。
            began=time.perf_counter()
            self.forecast_inputs[self.step]={'detector_visual_input':inputs[0].detach().cpu().float().clone(),
                'detector_action_condition':inputs[1].detach().cpu().float().clone(),'detector_prediction':output.detach().cpu().float().clone()}
            for origin in list(self.forecast_inputs):
                if origin<self.step-self.k:del self.forecast_inputs[origin]
            self.audit_s+=time.perf_counter()-began
        handle=self.predictor.register_forward_hook(forecast)
        try:
            with super().installed():yield self
        finally:handle.remove()

    def arrays(self,name,values):
        began=time.perf_counter();path=self.directory/'tensors'/name;path.parent.mkdir(exist_ok=True)
        arrays={k:v.detach().cpu().float().numpy() if isinstance(v,torch.Tensor) else np.asarray(v) for k,v in values.items()}
        if any(a.dtype.hasobject or not np.isfinite(a).all() for a in arrays.values()):raise ValueError('记录数组非法，停止，不补造')
        np.savez_compressed(path,**arrays);entry={'path':path.relative_to(self.directory).as_posix(),'sha256':sha(path),
            'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}}
        self.tensor_files.append(entry);self.audit_s+=time.perf_counter()-began;return entry

    def capture_target(self,expected,observed):
        if expected.shape!=observed.shape or expected.shape!=self.z_now.shape:raise RuntimeError('目标shape不一致')
        target=(expected-observed).detach()
        self.last_event=f'event_{self.step:04d}'
        arrays={'expected_delta':expected,'observed_delta':observed,'target_delta':target,'current_z':self.z_now}
        origin=self.step-self.k;original=self.forecast_inputs.get(origin)
        if original is None or not torch.equal(original['detector_prediction'],expected.detach().cpu().float()):raise RuntimeError('原检测预测输入未按到期origin保留，停止，不补造')
        arrays.update(original)
        if self.context_now is not None:arrays['visual_context']=self.context_now
        item=self.arrays(self.last_event+'.npz',arrays)
        row={'type':'hybrid_trigger','event':self.last_event,'step':self.step,'prediction_origin_step':self.step-self.k,
             'target_shape':list(target.shape),'target_norm':float(target.norm()),'expected_norm':float(expected.norm()),
             'observed_norm':float(observed.norm()),'target_path':item['path'],'target_sha256':item['sha256'],
             'target_source':'detector_expected_delta_minus_observed_delta','tensor_format':'npz_numeric_no_pickle'}
        self.target_files.append(row);super().log(row);return target

    def log(self,row):
        if self._capture_guide and ('guidance_loss' in row or 'ogg_loss' in row):self._guide_row=dict(row);return
        row=dict(row)
        if 'score' in row:
            row.update(score_valid=row['score'] is not None,pending_predictions=len(self.pending),
                       prediction_created=bool(self.pending and self.pending[-1][0]==self.step+self.k))
            if row['score'] is not None:row['prediction_origin_step']=self.step-self.k
        super().log(row)

    def denoise_step(self,*a,**kw):
        if not(self.active and self.denoise_index%self.every==0):return super().denoise_step(*a,**kw)
        x=a[2] if len(a)>2 else kw['x_t'];t=a[3] if len(a)>3 else kw['timestep'];before=[];gradients=[];original=self.original_denoise;autograd=torch.autograd.grad
        if x.is_cuda:torch.cuda.synchronize()
        began=time.perf_counter()
        def base(*args,**kwargs):
            out=original(*args,**kwargs);before.append(out.detach());return out
        def capture(*args,**kwargs):
            out=autograd(*args,**kwargs);gradients.append(out[0].detach().clone());return out
        index=self.denoise_index;self._guide_row=None;self._capture_guide=True
        try:
            with patch.object(self,'original_denoise',base),patch.object(torch.autograd,'grad',capture):out=super().denoise_step(*a,**kw)
        finally:self._capture_guide=False
        if x.is_cuda:torch.cuda.synchronize()
        elapsed=time.perf_counter()-began;self.guide_seconds+=elapsed
        if len(before)!=1 or len(gradients)!=1 or self._guide_row is None:raise RuntimeError('OGG实际调用/梯度记录不完整')
        v=before[0];gradient=gradients[0];update=out-v;clean_before=x-t[:,None,None]*v;clean_after=x-t[:,None,None]*out
        if not torch.equal(out,(v-self.eta*gradient).detach()):raise RuntimeError('同一次实际梯度的原OGG速度更新恒等式不成立')
        guide=getattr(self,'guide',self.predictor);offsets=list(range(0,2*guide.cfg.action_steps,2)) if getattr(guide,'uses_action_sequence',False) else [0]
        per_action=(clean_after-clean_before)[:,:,:7].norm(dim=-1)
        unsupported=[i for i in range(x.shape[1]) if i not in offsets]
        if unsupported and torch.count_nonzero(gradient[:,unsupported]).item():raise RuntimeError('OGG梯度作用到未登记动作位置')
        entry=self.arrays(f'{self.last_event}_denoise_{index:02d}.npz',{'candidate_clean_before':clean_before[:,:,:7],
            'candidate_clean_after':clean_after[:,:,:7],'velocity_before':v,'velocity_after':out,'gradient':gradient})
        row={**self._guide_row,'event':self.last_event,'condition_offsets':offsets,'gradient_l2':float(gradient.norm()),
            'original_update_formula_exact_equal':True,
            'guidance_seconds':elapsed,
            'velocity_update_l2':float(update.norm()),'clean_action_update_l2':float((clean_after-clean_before).norm()),
            'per_action_clean_update_l2':per_action.detach().cpu().tolist(),'candidate_arrays':entry,
            'update_units':'normalized candidate clean actions; not measured executed displacement'}
        super().log(row);return out


class AuditedOnline(AuditMixin,OnlineCorrector):pass
class AuditedHybrid(AuditMixin,HybridCorrector):pass
