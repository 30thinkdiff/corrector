"""纯SmolVLA执行窗口Pilot：固定完整动作块，仅执行前H步；不导入Corrector。

复用LIBERO环境、官方eval_policy及逐回合持久化；冻结实验源码不原地修改。
"""
import argparse
from contextlib import ExitStack
import csv
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from batch_evaluation import make_plan, run_plan, write_json, hash_file
except ModuleNotFoundError as error:
    if error.name != 'batch_evaluation':
        raise
    sys.path.insert(0, str(HERE.parent))
    from studies.policy_generalization.corrector.batch_evaluation import make_plan, run_plan, write_json, hash_file
import smolvla_training_workbench as wb
import smolvla_libero_tools as sim
from smolvla_baseline_benchmark import POLICY_ID, POLICY_SHA, select_initial_state

HORIZONS = (50, 20, 10, 5, 2, 1)
BASE = Path('/root/autodl-tmp/corrector')
EXPECTED_VERSIONS = {'torch': '2.11.0+cu128', 'lerobot': '0.6.1',
                     'transformers': '5.5.4', 'mujoco': '3.7.0', 'robosuite': '1.4.1'}


def read(path):
    return json.loads(Path(path).read_text())


def old_processes(proc=Path('/proc')):
    """从argv识别旧调度器及实验；忽略自身，不靠易匹配到检查命令的pgrep。"""
    found = []
    for path in proc.glob('[0-9]*/cmdline'):
        try:
            pid = int(path.parent.name)
            if pid == os.getpid():
                continue
            argv = path.read_bytes().decode(errors='replace').split('\0')
            scripts = [Path(a).name for a in argv if a.endswith('.py')]
            if any(re.match(r'(corrector_|smolvla_|lerobot_).*\.py$', a) for a in scripts):
                found.append({'pid': pid, 'scripts': scripts})
        except (OSError, ValueError):
            continue
    return found


def prior_training(base):
    records = []
    for folder, variants in [('night_20261109_001', ['mlp', 'transformer-h1', 'transformer-h3']),
                             ('action_followup_20261109_001/experiments',
                              ['transformer-a10-h1', 'transformer-a10-h3'])]:
        for variant in variants:
            for seed in (42, 43, 44):
                run = base/'object10'/folder/'runs'/f'{variant}_s{seed}'
                status = read(run/'training_status.json')
                if status.get('status') != 'completed':
                    raise RuntimeError(f'旧训练未完成：{run}，{status.get("status")}；不启动')
                records.append({'run': str(run), 'status': status['status'],
                                'epochs': status.get('epochs_completed')})
    return records


def preflight(base=BASE):
    """不加载模型；每次启动均检查，不把历史空闲快照当作当前空闲证明。"""
    processes = old_processes()
    if processes:
        raise RuntimeError('发现旧/其他实验进程，不启动：' + json.dumps(processes))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in gpu.split(',')]
    if len(gpu.splitlines()) != 1 or len(fields) != 4 or '4090' not in fields[0]:
        raise RuntimeError('GPU配置与单4090协议不一致：' + gpu)
    if apps or float(fields[1]) > 128 or float(fields[3]) > 5:
        raise RuntimeError('GPU非空闲，不启动：' + gpu + ' / ' + apps)
    free = shutil.disk_usage(base).free/2**30
    if free < 5:
        raise RuntimeError(f'数据盘剩余{free:.2f}GiB，小于5GiB，不启动')
    for folder, expected in [('lerobot', wb.LEROBOT_SHA), ('libero-original', sim.LIBERO_SHA)]:
        actual = subprocess.check_output(['git', '-C', str(base/'runtime'/folder),
                                          'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError(f'{folder}版本冲突，不启动：{actual}')
        dirty = subprocess.check_output(['git', '-C', str(base/'runtime'/folder), 'diff',
                                         '--name-only', 'HEAD'], text=True).strip()
        if dirty:
            raise RuntimeError(f'{folder}存在已跟踪源码修改，不启动：{dirty}')
    return {'gpu': gpu, 'compute_processes': apps, 'free_gib': free,
            'prior_training': prior_training(base)}


class QueueAudit:
    """验证实际select_action缓存消费与当前观测触发；完整块长度始终为50。"""
    def __init__(self, policy, horizon, directory, warmup=2):
        self.policy, self.h, self.directory = policy, horizon, directory
        self.steps = self.calls = self.samples = self.denoise = 0
        self.times, self.trace, self.actions = [], [], []
        self.expected = []
        self.warmup = warmup
        self.warm = False
        self.warmup_s = 0.
        self.first_noise_sha = None
        self.original_chunk = policy._get_action_chunk
        self.original_select = policy.select_action
        self.original_sample = policy.model.sample_actions
        self.original_denoise = policy.model.denoise_step
        self.active = False

    def sample(self, *a, **kw):
        if self.active:
            self.samples += 1
        return self.original_sample(*a, **kw)

    def denoise_step(self, *a, **kw):
        if self.active:
            self.denoise += 1
        return self.original_denoise(*a, **kw)

    def chunk(self, batch, *a, **kw):
        import torch
        if not self.active or len(self.policy._queues['action']) != 0:
            raise RuntimeError('动作生成必须来自当前select_action且旧动作队列为空')
        torch.cuda.synchronize()
        began = time.perf_counter()
        out = self.original_chunk(batch, *a, **kw)
        torch.cuda.synchronize()
        self.times.append(time.perf_counter()-began)
        if out.ndim != 3 or tuple(out.shape[:2]) != (1, 50):
            raise RuntimeError(f'实际生成非完整50动作块：{tuple(out.shape)}')
        self.expected = list(out[:, :self.h].transpose(0, 1).unbind(0))
        self.calls += 1
        return out

    def select(self, batch, *a, **kw):
        import torch
        if not self.warm:
            # 同一首观测做两次完整热身；保存Torch RNG，排除在正式调用/计时之外。
            began = time.perf_counter()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                for _ in range(self.warmup):
                    self.original_chunk(batch)
                torch.cuda.synchronize()
            self.warmup_s = time.perf_counter()-began
            self.policy.reset()
            torch.cuda.reset_peak_memory_stats()
            self.warm = True
        before = len(self.policy._queues['action'])
        should_infer = self.steps % self.h == 0
        if (before == 0) != should_infer:
            raise RuntimeError(f'step{self.steps}旧队列残留/提前耗尽：{before}')
        if should_infer and self.expected:
            raise RuntimeError('新推理前仍有未消费的上一块前H动作')
        calls_before = self.calls
        self.active = True
        try:
            out = self.original_select(batch, *a, **kw)
        finally:
            self.active = False
        if self.calls-calls_before != int(should_infer):
            raise RuntimeError('实际动作块调用与执行窗口不一致')
        if not self.expected or not torch.equal(out, self.expected.pop(0)):
            raise RuntimeError('执行动作不是本块对应的前H动作')
        after = len(self.policy._queues['action'])
        if after != len(self.expected) or self.policy._queues['action'].maxlen != self.h:
            raise RuntimeError('动作缓存长度不一致')
        self.trace.append({'step': self.steps, 'queue_before': before, 'queue_after': after,
                           'fresh_inference': should_infer, 'chunk_index': self.calls-1})
        self.steps += 1
        return out

    def noise(self, original, *a, **kw):
        value = original(*a, **kw)
        if self.active and self.first_noise_sha is None:
            self.first_noise_sha = hashlib.sha256(value.detach().cpu().float().numpy().tobytes()).hexdigest()
        return value

    def check(self):
        if self.calls != math.ceil(self.steps/self.h) or self.samples != self.calls:
            raise RuntimeError('实际sample_actions次数不匹配，不计为有效回合')
        if self.denoise != self.calls*self.policy.config.num_steps:
            raise RuntimeError('实际去噪前向次数与固定num_steps不匹配')


def continuity(actions, horizon):
    """环境真正收到的相对控制向量；分开平移/旋转/夹爪，不冒称物理jerk。"""
    import numpy as np
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
        raise RuntimeError('环境实际动作须为有限的[N,7]')
    delta = np.diff(a, axis=0)
    boundary = np.arange(1, len(a)) % horizon == 0
    result = {}
    for label, mask in [('all', np.ones(len(delta), dtype=bool)), ('boundary', boundary), ('within', ~boundary)]:
        result[label+'_n'] = int(mask.sum())
        for name, part in [('translation', slice(0, 3)), ('rotation', slice(3, 6))]:
            values = np.linalg.norm(delta[mask, part], axis=1)
            result[label+'_'+name+'_delta_mean'] = float(values.mean()) if len(values) else None
            result[label+'_'+name+'_delta_p95'] = float(np.percentile(values, 95)) if len(values) else None
    result['gripper_sign_switches'] = int(np.sum((a[1:, 6] > 0) != (a[:-1, 6] > 0)))
    return result


def pilot_summary(output):
    rows = []
    records = [read(p) for p in sorted((output/'trials').glob('*.json'))]
    for h in HORIZONS:
        subset = [r for r in records if r['case']['method'] == f'H{h}' and r['status'] == 'completed']
        results = [r['result'] for r in subset]
        row = {'method': f'H{h}', 'completed': len(results), 'planned': 4,
               'successes': sum(r['success'] for r in results),
               'success_rate': sum(r['success'] for r in results)/len(results) if results else None}
        for key in ['env_steps', 'chunk_calls', 'sample_actions_calls', 'denoise_forward_calls',
                    'policy_s', 'rollout_wall_s', 'peak_torch_allocated_gib', 'peak_torch_reserved_gib']:
            row['mean_'+key] = sum(r[key] for r in results)/len(results) if results else None
        for key in ['all_translation_delta_mean', 'boundary_translation_delta_mean',
                    'all_rotation_delta_mean', 'boundary_rotation_delta_mean', 'gripper_sign_switches']:
            values = [r['continuity'][key] for r in results if r['continuity'][key] is not None]
            row['mean_'+key] = sum(values)/len(values) if values else None
        row['mean_episode_wall_s'] = sum(r['wall_s'] for r in subset)/len(subset) if subset else None
        rows.append(row)
    # 初始状态、首输入和首噪声必须在同一case的各方法之间完全配对。
    groups = {}
    for record in records:
        if record['status'] == 'completed':
            key = (record['case']['condition'], record['case']['seed'])
            groups.setdefault(key, set()).add(tuple(record['result'][k] for k in
                ['initial_state_sha256', 'first_input_sha256', 'first_noise_sha256']))
    if any(len(values) != 1 for values in groups.values()):
        raise RuntimeError('方法间初始状态/首输入/首噪声不一致，停止并报告')
    output.joinpath('analysis').mkdir(exist_ok=True)
    with (output/'analysis/pilot_overall.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'analysis/pilot_summary.json', {'methods': rows, 'paired_initial_conditions_verified': True,
        'scope': 'failure-selected two-task pilot, four episodes/method; not Object10-wide success rate',
        'continuity_units': 'environment relative control-vector differences; not physical jerk',
        'timing': 'synchronized full chunk generation; rollout includes preprocessing/physics/rendering; episode wall includes setup/artifacts',
        'automatic_expansion': False})
    return rows


def run(args):
    lockfile = (args.base/'.horizon-pilot.lock').open('a')
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('已有Pilot持有GPU启动锁，不重复运行')
    if args.output.exists() and not args.resume:
        raise RuntimeError('输出目录已存在；不覆盖。续跑须显式--resume且计划一致')
    if args.resume and not (args.output/'plan.json').is_file():
        raise RuntimeError('没有可续跑的冻结Pilot计划')
    # 新GPU任务前的最终检查，状态/status命令不调用这里。
    audit = preflight(args.base)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'preflight.json', audit)
    import torch
    from importlib.metadata import version
    actual = {name: version(name) for name in EXPECTED_VERSIONS}
    if actual != EXPECTED_VERSIONS or not torch.cuda.is_available():
        raise RuntimeError('运行库/CUDA配置不一致，不启动：' + json.dumps(actual))
    sim.configure(argparse.Namespace(workdir=args.base/'runtime', sim_fps=20, max_steps=280), wb)
    # configure复用原环境路径；强制全程只读本地模型缓存，禁止慢网络隐式补下载。
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from libero.libero import benchmark
    snapshot = Path(snapshot_download(POLICY_ID, revision=POLICY_SHA, local_files_only=True, token=False))
    cfg = SmolVLAConfig.from_pretrained(snapshot)
    if cfg.chunk_size != 50 or cfg.n_action_steps != 50 or cfg.rtc_config is not None:
        raise RuntimeError('缓存基础模型配置非原50步/无RTC，不启动')
    if cfg.num_steps != 10 or cfg.adapt_to_pi_aloha:
        raise RuntimeError('基础模型去噪/动作适配配置与原LIBERO协议不一致')
    cfg.device, cfg.pretrained_path, cfg.load_vlm_weights = 'cuda', snapshot, False
    suite = benchmark.get_benchmark_dict()['libero_object']()
    conditions = {f'libero_object/task_{t}/init_{i}': {'task_id': t, 'initial_state_id': i,
                  'language': suite.tasks[t].language} for t in (1, 9) for i in (0, 2)}
    sources = {p.name: hash_file(p) for p in HERE.glob('*.py')}
    protocol = {'policy_id': POLICY_ID, 'policy_revision': POLICY_SHA, 'chunk_size': 50,
        'num_steps': cfg.num_steps, 'fps': 20, 'max_env_steps': 280, 'suite': 'libero_object',
        'conditions': conditions, 'runtime': actual, 'source_sha256': sources,
        'lerobot_revision': wb.LEROBOT_SHA, 'libero_revision': sim.LIBERO_SHA,
        'policy_source_sha256': hash_file(Path(inspect.getfile(SmolVLAPolicy))),
        'config_sha256': hash_file(snapshot/'config.json'), 'warmup_chunks_per_episode': 2,
        'weight_sha256': {p.name: hash_file(p) for p in snapshot.glob('*.safetensors')},
        'seed': 1000, 'corrector': False, 'lvm': False, 'ogg': False,
        'task_selection': 'historical cream-cheese H50 failures vs H20 recovery; orange-juice H20 failures',
        'automatic_expansion': False}
    plan = make_plan(list(conditions), [1000],
                     {f'H{h}': {'n_action_steps': h, 'uses_corrector': False, 'video_policy': 'all'}
                      for h in HORIZONS}, protocol)
    if args.resume and read(args.output/'plan.json') != plan:
        raise RuntimeError('配置/源码与旧Pilot不一致，拒绝续跑')
    holder = {}

    def run_case(case, directory):
        if not holder:
            policy = SmolVLAPolicy.from_pretrained(snapshot, config=cfg).to('cuda').eval()
            policy.requires_grad_(False)
            pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=snapshot,
                preprocessor_overrides={'device_processor': {'device': 'cuda'}})
            holder.update(policy=policy, pre=pre, post=post)
        policy, pre, post = holder['policy'], holder['pre'], holder['post']
        h = case['config']['n_action_steps']
        policy.config.n_action_steps = h
        policy.reset()
        if policy.config.chunk_size != 50 or policy._queues['action'].maxlen != h:
            raise RuntimeError('动作块配置或缓存容量不一致')
        chosen = conditions[case['condition']]
        cameras = list(policy.config.image_features)
        if len(cameras) != 2:
            raise RuntimeError('相机配置不一致')
        env_cfg = LiberoEnv(task='libero_object', task_ids=[chosen['task_id']], fps=20,
            episode_length=280, observation_height=256, observation_width=256,
            control_mode='relative', is_libero_plus=False,
            camera_name_mapping={'agentview_image': cameras[0].removeprefix('observation.images.'),
                                 'robot0_eye_in_hand_image': cameras[1].removeprefix('observation.images.')})
        env = sim.make_vector(env_cfg)
        env_pre, env_post = env_cfg.get_env_processors()
        monitor = QueueAudit(policy, h, directory)
        first = {}
        initial = {}
        def observe(batch):
            value = pre(batch)
            if not first:
                tensor = torch.as_tensor(env.envs[0].unwrapped._env.get_sim_state()).cpu().numpy()
                initial['sha'] = hashlib.sha256(tensor.tobytes()).hexdigest()
                dig = hashlib.sha256()
                for key in sorted(value):
                    if isinstance(value[key], torch.Tensor):
                        v = value[key].detach().cpu().contiguous()
                        dig.update(str((key, str(v.dtype), tuple(v.shape))).encode())
                        dig.update(v.view(torch.uint8).numpy().tobytes())
                first['sha'] = dig.hexdigest()
            return value
        original_step = env.step
        def step(action):
            # numpy action即官方env_postprocessor之后、实际传入物理环境的7维控制。
            if len(action) != 1:
                raise RuntimeError('只允许单环境Pilot')
            monitor.actions.append(action[0].tolist())
            return original_step(action)
        original_noise = policy.model.sample_noise
        try:
            select_initial_state(env, chosen['initial_state_id'])
            set_seed(case['seed'])
            torch.cuda.reset_peak_memory_stats()
            with ExitStack() as stack:
                for obj, name, method in [(policy, '_get_action_chunk', monitor.chunk),
                    (policy, 'select_action', monitor.select),
                    (policy.model, 'sample_actions', monitor.sample),
                    (policy.model, 'denoise_step', monitor.denoise_step), (env, 'step', step),
                    (policy.model, 'sample_noise', lambda *a, **kw: monitor.noise(original_noise, *a, **kw))]:
                    stack.enter_context(patch.object(obj, name, method))
                started = time.perf_counter()
                info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                    preprocessor=observe, postprocessor=post, n_episodes=1, max_episodes_rendered=1,
                    videos_dir=directory/'videos', start_seed=case['seed'])
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
            monitor.check()
            if len(monitor.actions) != monitor.steps:
                raise RuntimeError('推理/动作实际执行步数不一致')
            videos = [sim.verify_video(p, 20) for p in info.get('video_paths', [])]
            if len(videos) != 1 or videos[0]['frames'] != monitor.steps:
                raise RuntimeError('录像缺失或帧数与动作数不同')
            write_json(directory/'eval_info.json', info)
            write_json(directory/'queue_trace.json', monitor.trace)
            write_json(directory/'executed_actions.json', monitor.actions)
            write_json(directory/'actual_protocol.json', {**chosen, 'horizon': h, **protocol})
            result = {'success': bool(info['per_episode'][0]['success']), 'env_steps': monitor.steps,
                'chunk_calls': monitor.calls, 'sample_actions_calls': monitor.samples,
                'denoise_forward_calls': monitor.denoise, 'policy_s': sum(monitor.times),
                'chunk_latency_s': monitor.times, 'rollout_wall_s': elapsed-monitor.warmup_s,
                'rollout_including_warmup_s': elapsed, 'warmup_s': monitor.warmup_s,
                'peak_torch_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                'peak_torch_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                'continuity': continuity(monitor.actions, h), 'queue_verified': True,
                'initial_state_sha256': initial['sha'], 'first_input_sha256': first['sha'],
                'first_noise_sha256': monitor.first_noise_sha,
                'video_paths': info.get('video_paths', [])}
            # 在完成标记前对照已有方法，错配立即停止，不把混合条件计入有效结果。
            for path in (args.output/'trials').glob('*.json'):
                old = read(path)
                if old['status'] == 'completed' and old['case']['condition'] == case['condition']:
                    for key in ['initial_state_sha256', 'first_input_sha256', 'first_noise_sha256']:
                        if old['result'][key] != result[key]:
                            raise RuntimeError(f'方法间{key}不一致，不启动后续回合')
            print(json.dumps({'case': case['condition'], 'method': case['method'],
                              'success': result['success'], 'steps': monitor.steps,
                              'chunk_calls': monitor.calls}, ensure_ascii=False), flush=True)
            return result
        finally:
            env.close()

    stop = args.output/'STOP'
    def request_pause(signum, frame):
        stop.touch()
        print('收到暂停请求；当前回合结束后暂停。', flush=True)
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    try:
        run_plan(args.output, plan, {f'H{h}': run_case for h in HORIZONS},
                 max_seconds=3*3600, min_free_gib=5, stop_file=stop)
    finally:
        pilot_summary(args.output)
    print(json.dumps({'execution': read(args.output/'execution.json'), 'pilot_only': True}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'run', 'status', 'pause'])
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, default=BASE/'object10/horizon_pilot_20261111_001')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.mode == 'preflight':
        print(json.dumps(preflight(args.base), ensure_ascii=False, indent=2))
    elif args.mode == 'status':
        for name in ['execution.json', 'analysis/pilot_summary.json', 'launch.json']:
            if (args.output/name).is_file():
                print(json.dumps({name: read(args.output/name)}, ensure_ascii=False, indent=2))
    elif args.mode == 'pause':
        if not args.output.is_dir():
            raise RuntimeError('Pilot目录不存在')
        (args.output/'STOP').touch()
        print('已请求在当前回合结束后暂停；未发送强制终止信号。')
    else:
        run(args)


if __name__ == '__main__':
    main()
