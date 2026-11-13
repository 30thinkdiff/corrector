"""混合Corrector消融：冻结检测器产生目标，独立冻结引导器仅在OGG事件调用。

不修改冻结模型工厂；分别以原工厂加载两类checkpoint，不做特征/时间/动作缩放。
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from corrector_online_runtime import OnlineCorrector


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for raw in iter(lambda: file.read(2**20), b''):
            h.update(raw)
    return h.hexdigest()


def load_frozen(checkpoint, factory_path, source_root, device, expected_sha=None):
    """保持各checkpoint的工厂SHA断言，隔离模块名，不替换全局corrector_models。"""
    import corrector_train as train
    config, _, mlp, _, source_hashes = train.reference_api(source_root)
    before = digest(checkpoint)
    if expected_sha is not None and before != expected_sha:
        raise RuntimeError('checkpoint不是已归档的冻结权重，拒绝加载')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    protocol = saved['feature_protocol']
    factory_sha = digest(factory_path)
    if protocol['model_factory_sha256'] != factory_sha:
        raise RuntimeError('冻结工厂与checkpoint协议不符，不加载')
    name = '_hybrid_original_factory_'+factory_sha[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, factory_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    factory = sys.modules[name]
    expected = factory.source_hashes(source_hashes, source_root, protocol['model_type'], digest)
    if expected != protocol['source_files']:
        raise RuntimeError('参考模型源码SHA与冻结训练协议不同')
    cls, cfg = factory.from_protocol(protocol, config, mlp, source_root)
    model = cls(cfg).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model, protocol, {'checkpoint': str(checkpoint), 'checkpoint_sha256': before,
                            'factory': str(factory_path), 'factory_sha256': factory_sha}


def compatible(single, sequence, fps=20):
    """协议不符直接拒绝；动作条件长度不同是实验变量，不进行适配填充。"""
    if single.get('model_type') != 'transformer' or sequence.get('model_type') != 'transformer-action':
        raise ValueError('本Pilot固定T-H1/A1与T-H1/A10，避免另加MLP架构变量')
    for key in ['encoder_policy_revision', 'source_sha', 'data_manifest_sha256', 'fps', 'k_step',
                'prediction_seconds', 'h_window', 'anchor_history', 'precision', 'records']:
        if key not in single or key not in sequence or single[key] != sequence[key]:
            raise ValueError(f'预测器协议不兼容：{key}；禁止缩放或重标')
    if single['h_window'] != 1 or single['fps'] != 10 or single['k_step'] != 10 or single['prediction_seconds'] != 1:
        raise ValueError('需要原10Hz、k10、1秒、H1协议')
    for key in ['token_dim', 'action_dim', 'd_model', 'n_heads', 'n_layers', 'dropout']:
        if single['model'][key] != sequence['model'][key]:
            raise ValueError('视觉/动作/骨干配置不兼容：'+key)
    if sequence.get('action_steps') != 10 or sequence['model'].get('action_steps') != 10:
        raise ValueError('A10模型动作长度错误')
    ratio = fps/single['fps']
    if ratio != int(ratio) or ratio < 1:
        raise ValueError('数据动作间距无法对应整数控制步')
    return {'compatible': True, 'visual_shape': [1, 128, single['model']['token_dim']],
            'prediction_seconds': 1., 'online_due_steps': int(fps), 'online_fps': fps,
            'data_fps': single['fps'], 'guide_action_offsets': [i*int(ratio) for i in range(10)],
            'target_definition': 'detector_expected_delta - (observed_z - origin_z)',
            'action_representation': 'frozen policy processor normalized; no temporary rescaling',
            'condition_distribution': 'training demonstrated executed actions; deployment planned candidate actions',
            'causal_limit': 'B vs C changes coverage, scores AND detector-generated targets, not coverage alone'}


def feature_evidence(archive, single, sequence):
    """逐条核对共享训练特征元信息，禁止仅靠token维度认为空间相同。"""
    result = None
    for record in single['records']:
        path = archive/'features'/f"episode_{record['episode']}"/'metadata.json'
        if digest(path) != record['files']['metadata.json']:
            raise RuntimeError('特征metadata与训练冻结记录不同：'+str(path))
        metadata = json.loads(path.read_text())
        values = {k: metadata[k] for k in ['l_tokens','d_dim','action_dim','fps','policy_id',
            'policy_revision','image_keys','camera_keys','action_representation','action_normalization','encoder_info']}
        if values['l_tokens'] != 128 or values['d_dim'] != 960 or values['action_dim'] != 7:
            raise RuntimeError('视觉token或动作shape不一致')
        if (values['policy_revision'] != single['encoder_policy_revision'] or values['fps'] != single['fps']
                or values['action_representation'] != 'checkpoint_processor_normalized'):
            raise RuntimeError('编码器、时基或动作归一化不兼容')
        if result is not None and result != values:
            raise RuntimeError('训练特征空间/相机顺序/处理器不一致')
        result = values
    if not result or single['records'] != sequence['records']:
        raise RuntimeError('预测器没有可核验的共享特征记录')
    return {'episodes_checked': len(single['records']), 'shared_feature_metadata': result}


def guidance_update(predictor, visual, x_t, velocity, timestep, target, offsets, eta=1.):
    """原cosine/clean=x-t*v约定；返回逐位置更新量，目标仅来自检测器。"""
    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError('引导器必须冻结且eval')
    if torch.is_inference_mode_enabled():
        raise RuntimeError('OGG需要离开inference_mode')
    if x_t.shape != velocity.shape or x_t.ndim != 3 or timestep.shape != (x_t.shape[0],):
        raise ValueError('flow输入shape不兼容')
    if not np.isfinite(eta) or eta <= 0 or max(offsets) >= x_t.shape[1]:
        raise ValueError('eta或候选动作范围错误')
    with torch.enable_grad():
        v = velocity.detach().clone().requires_grad_(True)
        clean = x_t.detach()-timestep[:,None,None]*v
        sequence = bool(getattr(predictor, 'uses_action_sequence', False))
        if sequence:
            if len(offsets) != predictor.cfg.action_steps:
                raise ValueError('引导器动作条件长度不符')
            action = clean[:, offsets, :predictor.cfg.action_dim]
        else:
            if offsets != [0]:
                raise ValueError('单动作引导器只能使用首动作')
            action = clean[:, 0, :predictor.cfg.action_dim]
        context = visual.detach().unsqueeze(1) if getattr(predictor, 'uses_history', False) else visual.detach()
        predicted = predictor(context, action)
        if predicted.shape != target.shape or predicted.shape != visual.shape:
            raise ValueError('引导输出与检测器目标特征空间shape不符，禁止广播/缩放')
        loss = (1-F.cosine_similarity(predicted.flatten(1), target.detach().flatten(1), dim=1)).mean()
        gradient = torch.autograd.grad(loss, v)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError('OGG Loss或梯度非有限')
    update = -eta*gradient
    clean_update = -timestep[:,None,None]*update
    per_action = clean_update[:, :, :predictor.cfg.action_dim].norm(dim=-1)
    stats = {'ogg_loss': float(loss.detach()), 'gradient_l2': float(gradient.norm()),
             'velocity_update_l2': float(update.norm()), 'clean_action_update_l2': float(clean_update.norm()),
             'per_action_clean_update_l2': per_action.detach().cpu().tolist(),
             'condition_offsets': offsets,
             'update_units': 'normalized candidate clean actions, not measured executed robot displacement'}
    return (v+update).detach(), stats


class HybridCorrector(OnlineCorrector):
    """保留原检测调度/覆盖规则，仅将OGG使用的预测器从检测器中分离。"""
    def __init__(self, policy, detector, detector_protocol, guide, guide_protocol, source_root, directory,
                 *, label, force_step=-1):
        super().__init__(policy, detector, detector_protocol, 'ogg', source_root, directory,
                         fps=20, stride=1, eta=1., guidance_every=3, implementation='reference',
                         trace=True, event_tensors=False, force_step=force_step, cooldown=0)
        self.guide, self.guide_protocol, self.label = guide, guide_protocol, label
        self.guide_s = 0.
        self.guide_offsets = [i*2 for i in range(10)] if getattr(guide,'uses_action_sequence',False) else [0]
        self.target = self.detector_target
        self.wall_start = time.perf_counter()
        self.last_event = None
        self.target_files = []

    def detector_target(self, expected, observed):
        if expected.shape != observed.shape or expected.shape != self.z_now.shape:
            raise ValueError('检测器目标形状错误')
        target = (expected-observed).detach()
        if not torch.isfinite(target).all():
            raise FloatingPointError('纠偏目标非有限')
        event = f'event_{self.step:04d}'
        path = self.directory/(event+'.pt')
        torch.save({'expected_delta':expected.detach().cpu(), 'observed_delta':observed.detach().cpu(),
                    'target':target.cpu(), 'z_now':self.z_now.detach().cpu()}, path)
        self.last_event = event
        entry = {'event': event, 'step':self.step, 'simulation_seconds': self.step/20,
                 'prediction_origin_step': self.step-self.k,
                 'wall_seconds': time.perf_counter()-self.wall_start, 'target_path':str(path),
                 'target_sha256':digest(path), 'target_shape':list(target.shape),
                 'target_norm':float(target.norm()), 'expected_norm':float(expected.norm()),
                 'observed_norm':float(observed.norm()), 'target_source':'detector_expected_minus_observed'}
        self.target_files.append(entry)
        self.log({'type':'hybrid_trigger', 'configuration':self.label, **entry})
        return target

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        v = self.original_denoise(prefix_pad_masks, past_key_values, x_t, timestep)
        if self.active and self.denoise_index % self.every == 0:
            if self.last_event is None or self.target_now is None:
                raise RuntimeError('没有检测器事件和纠偏目标时禁止调用引导器')
            if x_t.is_cuda:
                torch.cuda.synchronize()
            began = time.perf_counter()
            v, stats = guidance_update(self.guide, self.z_now, x_t, v, timestep,
                                      self.target_now, self.guide_offsets, self.eta)
            if x_t.is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-began
            self.guide_s += elapsed
            self.guidance_calls += 1
            self.log({'type':'hybrid_guidance', 'configuration':self.label, 'event':self.last_event,
                      'step':self.step, 'denoise':self.denoise_index, 'timestep':timestep.detach().cpu().tolist(),
                      'guidance_seconds':elapsed, **stats})
        self.denoise_index += 1
        return v
