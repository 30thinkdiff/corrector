"""Independent ordinary-LIBERO adapter over an immutable, trained J36M core.

No training, downloads, model conversion or edits to installed packages. Default
queue ends at the natural pilot; formal Object10 is a separate explicit phase.
"""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

METHODS = ('H50', 'H20', 'c0_lvm', 'c0_ogg', 'j_lvm', 'j_lvm_ogg')
BASE_MANIFEST = 'a742e42b77fa2d05592c6b7b639881c80ed28515aa57524bf4eab7b6d499d0cc'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def phase_cases(phase):
    if phase == 'passive':
        tasks, inits, seeds, methods = [0], [6], [9100], ['H50', 'c0_monitor', 'j_monitor']
    elif phase == 'calibration':
        tasks, inits, seeds, methods = range(10), [7, 8], [9000, 9001], ['c0_monitor', 'j_monitor']
        # Two predeclared independent normal rollouts per task/model, not a 2x2 search.
        return [(t, i, s, m) for t in tasks for i, s in zip(inits, seeds) for m in methods]
    elif phase == 'forced':
        tasks, inits, seeds, methods = [0], [6], [9101], list(METHODS[2:])
    elif phase == 'pilot':
        tasks, inits, seeds, methods = [1, 9], [0, 1], [1000], METHODS
    elif phase == 'formal':
        tasks, inits, seeds, methods = range(10), [0, 1, 2], [1000, 1001], METHODS
    else:
        raise ValueError('unknown phase')
    return [(t, i, s, m) for t in tasks for i in inits for s in seeds for m in methods]


def fit_normal(trials, model_sha, manifest_sha, layers):
    """Same normal MAD estimator as J calibration, with explicit Object10 domain.

    Failed H50 trajectories remain in the archive but do not define normal data.
    Never relabel an Object10 trial as a Plus/default trial to bypass its guard.
    """
    import numpy as np
    seen = set()
    for trial in trials:
        match = re.fullmatch(r'libero_object/task_([0-9])/init_([78])', trial['condition'])
        key = (trial['condition'], trial['seed'])
        if key in seen:
            raise ValueError('duplicate normal calibration case')
        seen.add(key)
        if (trial['seed'] not in (9000, 9001) or trial['init'] not in (7, 8)
                or trial['seed'] != 9000 + trial['init'] - 7
                or not match or int(match[2]) != trial['init']
                or trial['checkpoint_sha256'] != model_sha or not trial['normal_success']):
            raise ValueError('independent successful normal Object10 cases required')
    if len(trials) < 2:
        raise ValueError('fewer than two successful normal rollouts; no threshold guessed')
    rows = [row for trial in trials for row in trial['rows'] if row.get('layer_scores')]
    if len(rows) < 30:
        raise ValueError('insufficient normal scoring samples')
    reliability, floors = {}, {}
    for layer in layers:
        valid = [r['layer_scores'][layer] for r in rows if r['layer_scores'][layer].get('valid')]
        if len(valid) < 10:
            raise ValueError('insufficient valid normal layer scores: ' + layer)
        scores = np.array([r['score'] for r in valid])
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite calibration values')
        reliability[layer] = (len(valid) / len(rows)) / (float(np.median(abs(scores - np.median(scores)))) + .05)
        floors[layer] = max(1e-8, .1 * float(np.quantile([min(r['predicted_rms'], r['observed_rms']) for r in valid], .01)))
    total = sum(reliability.values())
    weights = {k: v / total for k, v in reliability.items()}
    history = []
    for trial in trials:
        previous = None
        for row in trial['rows']:
            detail = row.get('layer_scores', {})
            valid = [k for k in layers if detail.get(k, {}).get('valid') and
                     min(detail[k]['predicted_rms'], detail[k]['observed_rms']) >= floors[k]]
            denominator = sum(weights[k] for k in valid)
            if not denominator:
                continue
            score = sum(weights[k] * detail[k]['score'] for k in valid) / denominator
            previous = score if previous is None else .35 * score + .65 * previous
            history.append(previous)
    if len(history) < 30:
        raise ValueError('insufficient reliable normal history')
    return {'format': 'j36m_object_calibration_v1', 'domain': 'ordinary_libero_object', 'status': 'frozen',
            'checkpoint_sha256': model_sha, 'training_manifest_sha256': manifest_sha,
            'weights': weights, 'norm_floor': floors, 'normal_history': history,
            'cases': [{'condition': t['condition'], 'seed': t['seed']} for t in trials],
            'selection': 'all successful predeclared H50 monitor-only runs; failed runs retained, no seed search',
            'state_settings': {'window_size': len(history), 'base_noise_floor': float(np.quantile(history, .95)),
                               'ewma_alpha': .35, 'k_on': 3., 'k_off': 2., 'trigger_margin': .02,
                               'jump_trigger': .06, 'trigger_consecutive_steps': 5,
                               'reset_consecutive_steps': 5, 'cooldown_steps': 20}}


def connector_class():
    import torch
    import torch.nn.functional as F
    import corrector_online_runtime as legacy

    class ConnectorRuntime(legacy.OnlineCorrector):
        """Original connector MLP, existing controller; frozen normal-only thresholds."""
        def __init__(self, policy, predictor, cal, method, source, directory, force=-1):
            self.cal, self.last_score, self.details = cal, None, {}
            super().__init__(policy, predictor, {'prediction_seconds': 1., 'fps': 10., 'k_step': 10, 'h_window': 1},
                             method.removeprefix('c0_'), source, directory, force_step=force)
            for key, value in cal['state_settings'].items():
                setattr(self.state, key, value)
            self.state.bootstrapped = True
            self.state.clean_history = list(cal['normal_history'])
            original_threshold, original_interrupt = self.threshold_step, self.interrupt

            def threshold(score, state):
                if self.last_score is None:
                    return (state.dynamic_threshold[-1] if state.dynamic_threshold else 0.), False
                state.clean_history = list(cal['normal_history'])
                result = original_threshold(score, state)
                state.clean_history = list(cal['normal_history'])
                return result

            def interrupt(mode, trigger):
                decision = original_interrupt(mode, trigger) and self.last_score is not None
                if decision:
                    self.cancelled_predictions += len(self.pending)
                    self.pending.clear()
                return decision
            self.threshold_step, self.interrupt = threshold, interrupt

        def mismatch(self, predicted, observed):
            p, y = predicted.float().flatten(1), observed.float().flatten(1)
            finite = bool(torch.isfinite(p).all() and torch.isfinite(y).all())
            pn = float(p.square().mean().sqrt()) if finite else None
            yn = float(y.square().mean().sqrt()) if finite else None
            valid = finite and min(pn, yn) >= self.cal['norm_floor']['connector']
            self.last_score = float((1 - F.cosine_similarity(p, y)).clamp(0, 2).mean()) if valid else None
            self.details = {'connector': {'valid': valid, 'score': self.last_score,
                            'predicted_rms': pn, 'observed_rms': yn}}
            return predicted.new_tensor([self.last_score if valid else 0.])

        def select_action(self, *a, **kw):
            self.last_score, self.details = None, {}
            with patch.object(legacy, 'visual_mismatch', self.mismatch):
                return super().select_action(*a, **kw)

        def log(self, row):
            if 'score' in row:
                row = dict(row, score=self.last_score, layer_scores=self.details,
                           scoring_valid=self.last_score is not None)
            if row.get('interrupted'):
                name = f'connector_target_{self.step:06d}.pt'
                torch.save(self.target_now.detach().cpu(), self.directory / name)
                row = dict(row, correction_target_file=name)
            return super().log(row)
    return ConnectorRuntime


def verify_core(core):
    marker = core / 'DEPLOYMENT_MANIFEST.json'
    if digest(marker) != BASE_MANIFEST:
        raise ValueError('unexpected immutable trained core manifest')
    for name, sha in read(marker)['files'].items():
        if digest(core / name) != sha:
            raise ValueError('frozen core changed: ' + name)


def initialize(args):
    verify_core(args.core)
    sys.path[:0] = [str(args.core / 'scripts'), str(args.core)]
    os.environ.update(HF_HOME=str(args.runtime / 'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      LEROBOT_SRC=str(args.runtime / 'lerobot/src'), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                      LIBERO_CONFIG_PATH=str(args.runtime / 'libero-config'), TOKENIZERS_PARALLELISM='false',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4)
    import smolvla_libero_tools as sim
    import smolvla_training_workbench as wb
    # The existing configure routine verifies ordinary LIBERO source/import and config.
    sim.configure(argparse.Namespace(workdir=args.runtime, sim_fps=20, max_steps=280), wb)
    from corrector_data_workbench import cached_policy_snapshot, load_complete_policy
    from j36m.model import make_model
    from j36m.training import source_hash
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    models, hashes = {}, {}
    for name, checkpoint in [('j', args.j_checkpoint), ('c0', args.c0_checkpoint)]:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        protocol = saved['protocol']
        if (protocol['source'] != source_hash() or protocol['variant'] != {'j': 'J', 'c0': 'C0'}[name]
                or protocol['timebase'] != {'data_hz': 10, 'k': 10, 'online_hz': 20, 'due_steps': 20}):
            raise ValueError('checkpoint source/variant/timebase mismatch')
        model = make_model(protocol['variant'], saved['input_rms'])
        model.load_state_dict(saved['model'], strict=True)
        models[name] = model.to('cuda').eval().requires_grad_(False)
        hashes[name] = digest(checkpoint)
        config = protocol['config']
        if name == 'j':
            training_config = config
        elif config != training_config:
            raise ValueError('J/C0 training data differ')
    snapshot = Path(cached_policy_snapshot(False))
    if digest(snapshot / 'model.safetensors') != training_config['policy_weight_sha256']:
        raise ValueError('policy weight SHA differs from training')
    cfg = SmolVLAConfig.from_pretrained(snapshot, local_files_only=True)
    if (cfg.chunk_size, cfg.num_steps) != (50, 10):
        raise ValueError('unexpected frozen VLA chunk/denoise protocol')
    cfg.n_action_steps = 50
    policy = load_complete_policy(SmolVLAPolicy.from_pretrained, snapshot, cfg, 'cuda').to('cuda').eval().requires_grad_(False)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=snapshot,
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    return policy, pre, post, models, hashes


def template(name, sha):
    layers = ('block_02', 'block_08', 'connector') if name == 'j' else ('connector',)
    return {'status': 'collecting_normal_only', 'checkpoint_sha256': sha,
            'weights': {k: 1 / len(layers) for k in layers}, 'norm_floor': {k: 1e-8 for k in layers},
            'normal_history': [.01, .02] * 16, 'state_settings': {}}


def run_phase(args, phase, holder):
    import numpy as np
    import torch
    import smolvla_libero_tools as sim
    from j36m.online import JointOnlineCorrector
    from libero.libero import benchmark
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.libero import get_task_init_states
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from studies.policy_generalization.corrector.batch_evaluation import make_plan, run_plan
    policy, pre, post, models, hashes = holder
    suite = benchmark.get_benchmark_dict()['libero_object']()
    manifest = read(args.core / 'studies/policy_generalization/corrector/j36m/manifest70.json')
    task_text = {r['task_id']: r['task'] for r in manifest['records']}
    data_to_suite = {}
    for task_id, text in task_text.items():
        matches = [i for i, t in enumerate(suite.tasks) if ' '.join(t.language.lower().split()) == ' '.join(text.lower().split())]
        if len(matches) != 1: raise ValueError('dataset/task language mapping ambiguous')
        data_to_suite[str(task_id)] = matches[0]
    if set(data_to_suite.values()) != set(range(10)): raise ValueError('Object10 coverage mismatch')
    cases = phase_cases(phase)
    root = args.output / phase
    cal = {k: (read(args.output / f'{k}_calibration.json') if phase in ('forced', 'pilot', 'formal') else template(k, hashes[k])) for k in hashes}
    conditions = {f'libero_object/task_{t}/init_{i}': {'task': t, 'init': i, 'language': suite.tasks[t].language} for t, i, _, _ in cases}
    # make_plan is reused for stable case identities; only the declared cases survive.
    method_cfg = {m: {'n_action_steps': 20 if m == 'H20' else 50, 'uses_corrector': m[0] in ('j', 'c'),
                     'corrector_sha256': hashes.get(m.split('_')[0]), 'video_policy': 'all',
                     'calibration_sha256': digest(args.output / f'{m.split("_")[0]}_calibration.json') if phase in ('forced', 'pilot', 'formal') and m[0] in ('j', 'c') else None} for _, _, _, m in cases}
    protocol = {'domain': 'ordinary_libero_object', 'phase': phase, 'source_sha256': digest(__file__),
                'core_manifest_sha256': BASE_MANIFEST, 'checkpoints': hashes,
                'fps': 20, 'data_fps': 10, 'k_step': 10, 'due_steps': 20, 'chunk_size': 50, 'denoise_steps': 10,
                'horizon': 60 if phase in ('forced', 'passive') else 280, 'conditions': conditions,
                'forced_event': phase == 'forced', 'performance_scope': phase in ('pilot', 'formal'),
                'guidance_eta': 1., 'guidance_every': 3}
    protocol['dataset_task_to_suite_id'] = data_to_suite
    plan = make_plan(list(conditions), sorted({s for _, _, s, _ in cases}), method_cfg, protocol)
    wanted = {(f'libero_object/task_{t}/init_{i}', s, m) for t, i, s, m in cases}
    plan['cases'] = [c for c in plan['cases'] if (c['condition'], c['seed'], c['method']) in wanted]

    def callback(case, directory):
        guard(args)
        chosen = conditions[case['condition']]
        states = get_task_init_states(suite, chosen['task'], is_libero_plus=False)
        if not 0 <= chosen['init'] < len(states):
            raise ValueError('initial state out of bounds')
        method = case['method']
        policy.config.n_action_steps = case['config']['n_action_steps']
        policy.reset()
        if policy._queues['action'] or policy._queues['action'].maxlen != policy.config.n_action_steps:
            raise ValueError('stale or incorrectly sized policy action cache')
        cameras = list(policy.config.image_features)
        cfg = LiberoEnv(task='libero_object', task_ids=[chosen['task']], fps=20,
            episode_length=protocol['horizon'], observation_height=256, observation_width=256,
            control_mode='relative', is_libero_plus=False,
            camera_name_mapping={'agentview_image': cameras[0].split('.')[-1], 'robot0_eye_in_hand_image': cameras[1].split('.')[-1]})
        env = sim.make_vector(cfg)
        raw = env.envs[0].unwrapped
        raw.init_state_id = chosen['init']
        envpre, envpost = cfg.get_env_processors()
        runtime = None
        if method.startswith('j_'):
            runtime = JointOnlineCorrector(policy, models['j'], cal['j'], method, args.runtime / 'vla-corrector-reference', directory,
                                           checkpoint_sha=hashes['j'], force_step=20 if phase == 'forced' else -1)
        elif method.startswith('c0_'):
            runtime = connector_class()(policy, models['c0'].model, cal['c0'], method,
                args.runtime / 'vla-corrector-reference', directory, 20 if phase == 'forced' else -1)
        submitted, normalized, chunk, first = [], [], [], {}
        versions = {n: p._version for name, model in models.items() for n, p in ((name + '/' + k, v) for k, v in model.named_parameters())}
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        def observe(batch):
            value = pre(batch)
            if not first:
                torch.save(torch.as_tensor(raw._env.get_sim_state()).clone(), directory / 'initial_sim_state.pt')
                torch.save({k: v.detach().cpu() for k, v in value.items() if isinstance(v, torch.Tensor)}, directory / 'first_policy_input.pt')
                first['done'] = True
            return value

        original_step, original_chunk, original_noise = raw.step, policy._get_action_chunk, policy.model.sample_noise
        def step(action):
            submitted.append(np.asarray(action).copy().tolist())
            return original_step(action)
        def generate(*a, **kw):
            torch.cuda.synchronize(); tick = time.perf_counter()
            value = original_chunk(*a, **kw)
            torch.cuda.synchronize(); chunk.append({'step': len(submitted), 'seconds': time.perf_counter() - tick})
            if value.shape[1] != 50:
                raise ValueError('VLA did not generate original 50-action chunk')
            return value
        def noise(*a, **kw):
            value = original_noise(*a, **kw)
            if not (directory / 'first_noise.pt').exists():
                torch.save(value.detach().cpu(), directory / 'first_noise.pt')
            return value
        try:
            set_seed(case['seed'])
            with ExitStack() as stack:
                stack.enter_context(patch.object(raw, 'step', step))
                stack.enter_context(patch.object(policy, '_get_action_chunk', generate))
                stack.enter_context(patch.object(policy.model, 'sample_noise', noise))
                if runtime:
                    stack.enter_context(runtime.installed())
                original_select = policy.select_action
                def select(*a, **kw):
                    value = original_select(*a, **kw)
                    normalized.append(value.detach().cpu().tolist())
                    return value
                stack.enter_context(patch.object(policy, 'select_action', select))
                info = eval_policy(env=env, policy=policy, env_preprocessor=envpre, env_postprocessor=envpost,
                    preprocessor=observe, postprocessor=post, n_episodes=1, max_episodes_rendered=1,
                    videos_dir=directory / 'videos', start_seed=case['seed'])
            values = [s for task in info['per_task'] for s in task['metrics']['successes']]
            if len(values) != 1 or type(values[0]) is not bool:
                raise ValueError('expected exactly one real boolean success')
            videos = [sim.verify_video(v, 20) for v in info.get('video_paths', [])]
            if len(videos) != 1 or videos[0]['frames'] != len(submitted) or len(normalized) != len(submitted):
                raise ValueError('action/control/video count mismatch')
            if any(p._version != versions[name + '/' + n] or p.grad is not None for name, model in models.items() for n, p in model.named_parameters()):
                raise ValueError('frozen corrector weights mutated')
            write(directory / 'actions.json', {'submitted_env_actions': submitted, 'normalized_policy_actions': normalized,
                  'recording_site': 'LiberoEnv.step input, before internal controller transform'})
            write(directory / 'eval_info.json', info)
            # Exact first state/input/noise checks across genuinely matching condition/seed.
            for other in (root / 'attempts').glob('*/*/actual_protocol.json'):
                old = read(other)
                if (old['case']['condition'], old['case']['seed']) == (case['condition'], case['seed']):
                    for filename in ('initial_sim_state.pt', 'first_noise.pt', 'first_policy_input.pt'):
                        a = torch.load(directory / filename, weights_only=True)
                        b = torch.load(other.parent / filename, weights_only=True)
                        if isinstance(a, dict):
                            if a.keys() != b.keys(): raise ValueError('paired input keys differ')
                            for k in a: torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
                        else: torch.testing.assert_close(a, b, atol=0, rtol=0)
                    if phase == 'passive' or phase == 'calibration':
                        if read(other.parent / 'actions.json')['submitted_env_actions'] != submitted:
                            raise ValueError('monitor-only actions differ from matched H50 trajectory')
            write(directory / 'actual_protocol.json', {**protocol, 'case': case, 'camera_keys': cameras,
                'task_language': raw.task_description, 'video': videos[0], 'chunks': chunk,
                'video_alignment': 'frame count/order checked; recording phase not inferred from FPS',
                'frozen_parameters_unchanged': True})
            return {'success': values[0], 'env_steps': len(submitted), 'chunk_calls': len(chunk),
                    'policy_s': sum(c['seconds'] for c in chunk), 'monitor_s': runtime.monitor_s if runtime else 0.,
                    'event_replans': runtime.events if runtime else 0, 'guidance_calls': runtime.guidance_calls if runtime else 0,
                    'cancelled_condition_predictions': runtime.cancelled_predictions if runtime else 0,
                    'peak_torch_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'episode_wall_s': time.perf_counter() - start, 'video_paths': info['video_paths'],
                    'action_continuity_mean_l2': float(np.linalg.norm(np.diff(np.asarray(submitted), axis=0), axis=-1).mean()) if len(submitted) > 1 else None}
        finally:
            env.close()
    run_plan(root, plan, {c['method']: callback for c in plan['cases']}, max_seconds=max(1, args.deadline - time.time()),
             min_free_gib=8, stop_file=args.output / 'STOP')
    execution = read(root / 'execution.json')
    if execution['status'] != 'completed':
        raise RuntimeError('phase paused: ' + str(execution))
    records = [read(p) for p in (root / 'trials').glob('*.json')]
    if phase == 'calibration':
        for name in hashes:
            trials = []
            for record in records:
                if record['case']['method'] != name + '_monitor' or not record['result']['success']:
                    continue
                # Locate by the stable attempt directory rather than assuming a video nesting level.
                directory = root / 'attempts' / record['case']['id'] / f'attempt_{record["attempt"]:03d}'
                rows = [json.loads(line) for line in (directory / 'monitor_steps.jsonl').read_text().splitlines()]
                trials.append({'condition': record['case']['condition'], 'init': conditions[record['case']['condition']]['init'],
                    'seed': record['case']['seed'], 'normal_success': True, 'checkpoint_sha256': hashes[name], 'rows': rows})
            layers = ('block_02', 'block_08', 'connector') if name == 'j' else ('connector',)
            target = args.output / f'{name}_calibration.json'
            result = fit_normal(trials, hashes[name], digest(args.core / 'studies/policy_generalization/corrector/j36m/data70.json'), layers)
            if target.exists() and read(target) != result: raise ValueError('frozen calibration differs')
            write(target, result)
    if phase == 'forced':
        traces = {}
        for record in records:
            method = record['case']['method']
            directory = root / 'attempts' / record['case']['id'] / f'attempt_{record["attempt"]:03d}'
            rows = [json.loads(line) for line in (directory / 'monitor_steps.jsonl').read_text().splitlines()]
            trigger = next((r for r in rows if r.get('interrupted') and r['step'] == 20), None)
            if trigger is None: raise ValueError('forced event not observed at due step20')
            traces[method] = trigger
            calls = record['result']['guidance_calls']
            if (method.endswith('ogg') and calls < 1) or (method.endswith('lvm') and calls != 0):
                raise ValueError('OGG toggle interface failed')
        for a, b in [('j_lvm', 'j_lvm_ogg'), ('c0_lvm', 'c0_ogg')]:
            for key in ('score', 'threshold', 'layer_scores'):
                if traces[a][key] != traces[b][key]: raise ValueError('pre-fork detector differs: ' + key)
    write(args.output / (phase + '_receipt.json'), {'status': 'completed', 'phase': phase,
          'performance_scope': phase in ('pilot', 'formal'), 'execution': execution, 'checkpoint_sha256': hashes})


def guard(args):
    import shutil
    if (args.output / 'STOP').exists() or time.time() >= args.deadline:
        raise RuntimeError('STOP or wall budget; complete trial preserved')
    for path, floor in [('/root', 8), ('/root/autodl-tmp', 5)]:
        if shutil.disk_usage(path).free < floor * 2**30:
            raise RuntimeError('disk free below floor: ' + path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['run', 'status', 'stop'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core', type=Path, default=Path('/root/autodl-tmp/corrector/dev/j36m-35ecdd7488bf'))
    p.add_argument('--runtime', type=Path, default=Path('/root/autodl-tmp/corrector/runtime'))
    p.add_argument('--training', type=Path, default=Path('/root/autodl-tmp/corrector/j36m_longrun_20261113_001/training'))
    p.add_argument('--phase', choices=['pilot', 'formal'], default='pilot')
    p.add_argument('--max-hours', type=float, default=4.)
    args = p.parse_args()
    if args.mode == 'status':
        print(json.dumps(read(args.output / 'status.json'), ensure_ascii=False, indent=2)); return
    if args.mode == 'stop':
        (args.output / 'STOP').touch(); return
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 12:
        raise ValueError('budget 0..12 hours')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.deadline = time.time() + args.max_hours * 3600
        args.j_checkpoint, args.c0_checkpoint = args.training / 'J_s42/best.pt', args.training / 'C0_s42/best.pt'
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: (args.output / 'STOP').touch())
        state = {'status': 'running', 'pid': os.getpid(), 'phase': 'preflight', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(args.output / 'status.json', state)
        try:
            guard(args)
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if gpu: raise RuntimeError('GPU already occupied: ' + gpu)
            holder = initialize(args)
            for phase in ['passive', 'calibration', 'forced', 'pilot'] + (['formal'] if args.phase == 'formal' else []):
                state.update(phase=phase); write(args.output / 'status.json', state)
                run_phase(args, phase, holder)
            state.update(status='completed', phase='pilot_completed' if args.phase == 'pilot' else 'formal_completed')
        except BaseException as exc:
            state.update(status='paused' if (args.output / 'STOP').exists() or time.time() >= args.deadline else 'error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write(args.output / 'status.json', state)


if __name__ == '__main__':
    main()
