"""夜间有限扩展：原单动作LVM/OGG及固定A10规则；五模型同轨迹预测诊断。"""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
import resource
import shutil
from pathlib import Path
import signal
import subprocess
import time
from unittest.mock import patch

from smolvla_horizon_pilot import BASE,EXPECTED_VERSIONS,preflight,read
from smolvla_baseline_benchmark import POLICY_ID,POLICY_SHA,select_initial_state
import smolvla_training_workbench as wb
import smolvla_libero_tools as sim
try:
    from batch_evaluation import make_plan,run_plan,write_json
except ModuleNotFoundError:
    from studies.policy_generalization.corrector.batch_evaluation import make_plan,run_plan,write_json

HERE=Path(__file__).resolve().parent
OUTPUT=Path('/root/corrector_overnight_20261111/results')
BRANCH='experiment/overnight-generalization-20261111'
METHODS=['H50','H10','planned_lvm','executed_lvm','single_lvm','single_ogg']


def validate_calibration(base,calibration):
    source=calibration['source'];root=base/'object10'/source['output_name']
    if read(root/'pipeline_status.json')['status']!='completed':raise RuntimeError('被动校准来源不完整')
    rows=[]
    for path in (root/'trials').glob('*.json'):
        record=read(path);case=record['case']
        if case['condition']==source['condition'] and case['seed']==source['seed']:
            if record['status']!='completed' or not record['result']['success']:raise RuntimeError('校准参考不是成功轨迹')
            trace=root/'attempts'/case['id']/f"attempt_{record['attempt']:03d}"/'monitor_steps.jsonl'
            rows=[json.loads(l) for l in trace.read_text().splitlines()]
    actual=[r for r in rows if r['step']>=24 and r['planned_score'] is not None]
    if [r['step'] for r in actual]!=source['due_steps'] or [r['planned_score'] for r in actual]!=source['scores']:
        raise RuntimeError('阈值来源与原始被动轨迹不符，不启动测试')
    if not all(r['planned_score']==r['executed_score'] for r in actual):raise RuntimeError('校准共享评分不一致')
    values=sorted(source['scores']);i=(len(values)-1)*.95;lo=int(i);hi=min(lo+1,len(values)-1)
    q=values[lo]*(hi-i)+values[hi]*(i-lo)
    if q!=calibration['rule']['threshold'] or not calibration['threshold_frozen_before_new_tests']:
        raise RuntimeError('阈值非预登记参考分位数')


def summary(root):
    import csv,statistics
    records=[read(p) for p in (root/'trials').glob('*.json')]
    rows=[]
    for method in METHODS:
        sub=[r for r in records if r['status']=='completed' and r['case']['method']==method]
        results=[r['result'] for r in sub]
        row={'method':method,'planned':sum(c['method']==method for c in read(root/'plan.json')['cases']),'completed':len(sub),'successes':sum(r['success'] for r in results),
             'errors':sum(r['status']=='error' and r['case']['method']==method for r in records),
             'success_rate':sum(r['success'] for r in results)/len(sub) if sub else None}
        for key in ['env_steps','chunk_calls','policy_s','monitor_s','event_replans','peak_torch_allocated_gib']:
            row['mean_'+key]=statistics.mean(r[key] for r in results) if results else None
        row['mean_episode_wall_s']=statistics.mean(r['wall_s'] for r in sub) if sub else None
        eligible=sum(r['eligible_steps'] for r in results);scored=sum(r['scored_steps'] for r in results)
        row['score_coverage']=scored/eligible if eligible else None
        rows.append(row)
    (root/'analysis').mkdir(exist_ok=True)
    with (root/'analysis/closed_loop_overall.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(root/'analysis/closed_loop_summary.json',{'methods':rows,'scope':'two tasks, new initial states/seeds; exploratory closed-loop test',
        'threshold':'frozen successful-trajectory reference quantile, not independently anomaly-calibrated',
        'ogg':False,'automatic_expansion':False})


def run(args):
    lock=(args.base/'.consistency-closed-loop.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.output.exists() and not args.resume:raise RuntimeError('输出目录已存在，拒绝覆盖')
    if args.resume and not (args.output/'plan.json').is_file():raise RuntimeError('没有原计划可续跑')
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(32768,hard) if hard!=-1 else 32768,hard))
    parent=int(os.environ.get('CORRECTOR_OVERNIGHT_SUPERVISOR_PID','0'))
    original_processes=preflight.__globals__['old_processes']
    if parent:
        argv=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
        if parent!=os.getppid() or str(HERE/'corrector_overnight_queue.py').encode() not in argv:
            raise RuntimeError('不接受未知父进程绕过检查')
    with patch.dict(preflight.__globals__,old_processes=lambda:[p for p in original_processes() if p['pid']!=parent]):
        audit=preflight(args.base)
    if read(args.base/'object10/consistency_closed_loop_20261111_001/pipeline_status.json')['status']!='completed':
        raise RuntimeError('上一轮48回合未完成，不启动')
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline<=datetime.now(timezone.utc):raise RuntimeError('截止时间已到或没有时区')
    if subprocess.check_output(['git','-C',str(args.base),'branch','--show-current'],text=True).strip()!=BRANCH:
        raise RuntimeError('不在独立闭环实验分支')
    calibration=read(HERE/'calibration_rule.json');validate_calibration(args.base,calibration)
    args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'preflight.json',audit)
    write_json(args.output/'pipeline_status.json',{'status':'running','phase':'loading_frozen_models_and_validating_inputs'})
    def pause(*a):(args.output/'STOP').touch()
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    import numpy as np
    import torch
    from importlib.metadata import version
    from corrector_hybrid import load_frozen,feature_evidence,digest
    from corrector_hybrid_pilot import role_paths
    from corrector_executed_monitor import lazy_controller_capture
    from corrector_consistency_control import ConsistencyLVM
    actual={k:version(k) for k in EXPECTED_VERSIONS}
    if actual!=EXPECTED_VERSIONS:raise RuntimeError('运行库配置变化')
    sim.configure(argparse.Namespace(workdir=args.base/'runtime',sim_fps=20,max_steps=280),wb)
    source=args.base/'runtime/vla-corrector-reference';setting=role_paths(args.base)['a10']
    from corrector_overnight_metrics import TrajectoryDiagnostics
    from corrector_online_runtime import OnlineCorrector
    models={};protocols={};weights={}
    for item in read(HERE/'overnight_models.json'):
        loaded,mp,meta=load_frozen(Path(item['checkpoint']),Path(item['factory']),source,'cuda',item['sha'])
        if mp['h_window']!=item['h_window'] or mp.get('action_steps',1)!=item['action_steps']:
            raise RuntimeError('五模型输入协议与固定清单不一致')
        models[item['name']]=loaded;protocols[item['name']]=mp;weights[item['name']]=meta
    model=models['transformer-a10-h1'];predictor_protocol=protocols['transformer-a10-h1'];weight=weights['transformer-a10-h1']
    common=['encoder_policy_revision','source_sha','data_manifest_sha256','fps','k_step','prediction_seconds','anchor_history','records']
    for name,mp in protocols.items():
        for key in common:
            if mp[key]!=predictor_protocol[key]:raise RuntimeError(f'跨模型协议不兼容{name}:{key}；禁止缩放拼接')
        if mp['model']['action_dim']!=7 or mp['model']['token_dim']!=960:raise RuntimeError('跨模型维度不一致')
    write_json(args.output/'model_protocols.json',{'models':protocols,'weights':weights})
    # 先用此前真实采集的一段历史验证五模型输入/输出，不补造图像或动作。
    from corrector_overnight_metrics import model_inputs,prediction_metrics
    captures=sorted((args.base/'object10/executed_consistency_pilot_20261111_002/attempts').glob('*/*/feature_capture.json'))
    if not captures:raise RuntimeError('缺少真实采集窗口，不能做五模型预检')
    capture=read(captures[0]);commands=read(captures[0].parent/'controller_commands.json')
    if capture['shape'][1:]!=[128,960] or digest(Path(capture['path']))!=capture['sha256']:
        raise RuntimeError('真实采集特征的shape或SHA不符')
    frames=torch.from_numpy(np.fromfile(capture['path'],dtype=np.float32,count=25*128*960).reshape(25,128,960).copy())
    actions=torch.tensor([r['processor_normalized_action'] for r in commands[:25]],dtype=torch.float32)
    probes=[]
    with torch.no_grad():
        for name,m in models.items():
            mp=protocols[name];v,a,target=model_inputs(frames,actions,[4],mp['h_window'],mp.get('action_steps',1))
            x=v if getattr(m,'uses_history',False) else v[:,0]
            predicted=m(x.cuda(),a.cuda());metrics=prediction_metrics(predicted,target.cuda())
            probes.append({'model':name,'input_visual':list(x.shape),'input_action':list(a.shape),
                'output':list(predicted.shape),'mse':float(metrics['mse'][0]),'source':str(captures[0]),
                'origin':4,'due':24,'real_executed_action_offsets':list(range(4,24,2))})
    write_json(args.output/'shape_probe.json',probes)
    evidence=feature_evidence(args.base/'object10',predictor_protocol,predictor_protocol)
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from libero.libero import benchmark
    snapshot=Path(snapshot_download(POLICY_ID,revision=POLICY_SHA,local_files_only=True,token=False))
    cfg=SmolVLAConfig.from_pretrained(snapshot)
    if cfg.chunk_size!=50 or cfg.n_action_steps!=50 or cfg.num_steps!=10 or cfg.rtc_config is not None:
        raise RuntimeError('基础策略配置不同')
    cfg.device,cfg.pretrained_path,cfg.load_vlm_weights='cuda',snapshot,False
    policy=SmolVLAPolicy.from_pretrained(snapshot,config=cfg).to('cuda').eval().requires_grad_(False)
    overrides={'device_processor':{'device':'cuda'}}
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,preprocessor_overrides=overrides)
    action_pre,action_post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,preprocessor_overrides=overrides)
    cameras=list(policy.config.image_features)
    if cameras!=evidence['shared_feature_metadata']['camera_keys']:raise RuntimeError('相机顺序不一致')
    suite=benchmark.get_benchmark_dict()['libero_object']()
    tasks=[1,9,0,2,3,4,5,6,7,8]
    for task in tasks:
        if len(suite.get_task_init_states(task))<50:raise RuntimeError('任务不足50个真实初始状态，停止')
    conditions={f'libero_object/task_{t}/init_{i}':{'task_id':t,'initial_state_id':i,'language':suite.tasks[t].language}
                for i in range(10,50) for t in tasks}
    settings={m:{'action_steps':10 if m=='H10' else 50,'video_policy':'none',
                 'ogg':m=='single_ogg','phase':'core' if m.startswith('single_') else 'expanded'} for m in METHODS}
    protocol={'policy_id':POLICY_ID,'policy_revision':POLICY_SHA,'checkpoint':weight,'all_weights':weights,'calibration':calibration,
        'rule_sha256':digest(HERE/'calibration_rule.json'),'conditions':conditions,'seeds':[1004,1005],
        'control_fps':20,'data_fps':10,'prediction_seconds':1,'horizon_steps':20,'action_offsets':list(range(0,20,2)),
        'max_steps':280,'chunk_size':50,'num_steps':10,'score_start':24,
        'single_control':'original frozen T-H1/seed42 robust LVM; OGG eta1 every3 denoise steps',
        'a10_control':'same pre-registered shared rule as 48-case pilot, no OGG',
        'diagnostics':'5 seed42 predictors, same actual trajectory; deferred batches; origins4,6,...; early due24..64',
        'timing_limit':'rollout includes passive shared encoder/capture; diagnostic batches timed separately; not native baseline latency',
        'max_retained_video_bytes':int(1.5*2**30),'max_failed_videos_per_task_method':2,
        'branch':BRANCH,'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},
        'test_result_tuning':False,'automatic_expansion':False}
    plan=make_plan(list(conditions),[1004,1005],settings,protocol)
    plan['cases']=[c for c in plan['cases'] if not c['method'].startswith('single_') or conditions[c['condition']]['initial_state_id']<15]
    if len(plan['cases'])!=3400:raise RuntimeError('夜间计划数量异常')
    if args.resume and read(args.output/'plan.json')!=plan:raise RuntimeError('续跑源码/规则/协议不同')
    write_json(args.output/'protocol.json',protocol)
    retained={};retained_bytes=0;sampled=set()
    signatures={}
    for p in (args.output/'trials').glob('*.json'):
        old=read(p)
        if old['status']=='completed':
            key=(old['case']['condition'],old['case']['seed']);signatures[key]=old['result']['initial_signature']
            result=old['result'];tag=(protocol['conditions'][old['case']['condition']]['task_id'],old['case']['method'])
            if result['video_paths']:
                retained[tag]=retained.get(tag,0)+len(result['video_paths']);retained_bytes+=sum(Path(v).stat().st_size for v in result['video_paths'])
            sampled.add(old['case']['method'])

    def callback(case,directory):
        nonlocal retained_bytes
        chosen=conditions.get(case['condition']) or case['_smoke_condition'];mode=case['method'];first={};noise={};raw_batch={};chunk_times=[]
        policy.config.n_action_steps=case['config']['action_steps'];policy.reset()
        if policy._queues['action'].maxlen!=case['config']['action_steps']:raise RuntimeError('动作队列容量错误')
        env_cfg=LiberoEnv(task='libero_object',task_ids=[chosen['task_id']],fps=20,episode_length=280,
            observation_height=256,observation_width=256,control_mode='relative',is_libero_plus=False,
            camera_name_mapping={'agentview_image':cameras[0].removeprefix('observation.images.'),
                                 'robot0_eye_in_hand_image':cameras[1].removeprefix('observation.images.')})
        env=sim.make_vector(env_cfg);env_pre,env_post=env_cfg.get_env_processors();unwrapped=env.envs[0].unwrapped
        def normalize(raw):
            batch={k:v.detach().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in raw_batch.items()}
            batch['action']=torch.tensor(raw.copy())[None];normalized=action_pre(batch)['action'].reshape(1,7).float()
            if not np.allclose(action_post(normalized).detach().cpu().numpy().reshape(7),raw,rtol=1e-5,atol=1e-5):
                raise RuntimeError('实际动作处理器往返不一致')
            return normalized
        if mode in ['planned_lvm','executed_lvm']:
            runtime=ConsistencyLVM(policy,model,predictor_protocol,source,directory,mode,calibration['rule'],normalize)
        elif mode in ['single_lvm','single_ogg']:
            runtime=OnlineCorrector(policy,models['transformer-h1'],protocols['transformer-h1'],
                'ogg' if mode=='single_ogg' else 'lvm',source,directory,fps=20,implementation='reference',trace=True)
        else:runtime=None
        diagnostics=TrajectoryDiagnostics(models,protocols,directory)
        encoder=OnlineCorrector(policy,model,predictor_protocol,'monitor',source,directory/'encoder',trace=False)
        shared_capture=[0.];normalization=[0.];normal_cache={};selected=[None]
        class Capture:
            in_reset=False;outer_action=None
            def controller_step(self,original,action,*a,**kw):
                if self.in_reset:return original(action,*a,**kw)
                raw=np.asarray(action,dtype=np.float32)
                if self.outer_action is None or not np.array_equal(raw,self.outer_action):raise RuntimeError('真实命令与外层提交不同')
                normalized=normalize_once(raw)
                diagnostics.submit(normalized,selected[0],policy._queues['action'],len(chunk_times))
                if mode in ['planned_lvm','executed_lvm']:
                    runtime.outer_action=self.outer_action
                    return runtime.controller_step(original,action,*a,**kw)
                return original(action,*a,**kw)
        capture=Capture()
        def normalize_once(raw):
            if 'action' not in normal_cache:
                began=time.perf_counter();normal_cache['action']=normalize(raw);normalization[0]+=time.perf_counter()-began
            return normal_cache['action']
        if mode in ['planned_lvm','executed_lvm']:runtime.normalizer=normalize_once
        def cached_encode(batch):
            if diagnostics.last_z is None or runtime.step!=len(diagnostics.actions):raise RuntimeError('共享视觉时序异常')
            return diagnostics.last_z
        selected_steps=0;commands=[]
        def observe(batch):
            raw_batch.clear();raw_batch.update(batch);value=pre(batch)
            normal_cache.clear()
            with torch.inference_mode(False),torch.no_grad():
                clean={k:v.clone() if isinstance(v,torch.Tensor) and v.is_inference() else v for k,v in value.items()}
                began=time.perf_counter();z=encoder.encode(clean);diagnostics.observe(z);torch.cuda.synchronize();shared_capture[0]+=time.perf_counter()-began
            if not first:
                first['state']=hashlib.sha256(torch.as_tensor(unwrapped._env.get_sim_state()).cpu().numpy().tobytes()).hexdigest()
                h=hashlib.sha256()
                for k in sorted(value):
                    if isinstance(value[k],torch.Tensor):
                        v=value[k].detach().cpu().contiguous();h.update(str((k,str(v.dtype),tuple(v.shape))).encode());h.update(v.view(torch.uint8).numpy().tobytes())
                first['input']=h.hexdigest()
            return value
        original_chunk=policy._get_action_chunk;original_noise=policy.model.sample_noise;original_select=policy.select_action;original_step=env.step
        def chunk(*a,**kw):
            if policy._queues['action']:raise RuntimeError('非空旧队列下错误调用新推理')
            torch.cuda.synchronize();began=time.perf_counter();out=original_chunk(*a,**kw);torch.cuda.synchronize()
            chunk_times.append(time.perf_counter()-began)
            if out.shape!=(1,50,7):raise RuntimeError('未生成原完整50动作块')
            return out
        def sample_noise(*a,**kw):
            out=original_noise(*a,**kw)
            if not noise:noise['sha']=hashlib.sha256(out.detach().cpu().float().numpy().tobytes()).hexdigest()
            return out
        def baseline_select(*a,**kw):
            nonlocal selected_steps
            h=case['config']['action_steps']
            if (not policy._queues['action'])!=(selected_steps%h==0):raise RuntimeError('纯VLA旧动作缓存消费错误')
            out=original_select(*a,**kw);selected_steps+=1;return out
        def step(action):
            commands.append(action[0].tolist())
            capture.outer_action=np.array(action[0],dtype=np.float32,copy=True)
            return original_step(action)
        try:
            torch.cuda.reset_peak_memory_stats();select_initial_state(env,chosen['initial_state_id']);set_seed(case['seed'])
            with ExitStack() as stack:
                for obj,name,fn in [(policy,'_get_action_chunk',chunk),(policy.model,'sample_noise',sample_noise),(env,'step',step)]:
                    stack.enter_context(patch.object(obj,name,fn))
                stack.enter_context(lazy_controller_capture(unwrapped,capture))
                if runtime:
                    stack.enter_context(patch.object(runtime,'encode',cached_encode));stack.enter_context(runtime.installed())
                else:stack.enter_context(patch.object(policy,'select_action',baseline_select))
                actual_select=policy.select_action
                def select(*a,**kw):
                    out=actual_select(*a,**kw);selected[0]=out.detach().clone();return out
                stack.enter_context(patch.object(policy,'select_action',select))
                rollout_began=time.perf_counter()
                info=eval_policy(env=env,policy=policy,env_preprocessor=env_pre,env_postprocessor=env_post,
                    preprocessor=observe,postprocessor=post,n_episodes=1,max_episodes_rendered=1,
                    videos_dir=directory/'videos',start_seed=case['seed'])
            rollout_seconds=time.perf_counter()-rollout_began
            n=runtime.step if runtime else selected_steps
            videos=[sim.verify_video(p,20) for p in info.get('video_paths',[])]
            if len(commands)!=n or len(videos)!=1 or videos[0]['frames']!=n:raise RuntimeError('视频/动作步数不一致')
            if not runtime and len(chunk_times)!=math.ceil(n/case['config']['action_steps']):raise RuntimeError('纯VLA实际调用计数不符')
            trace=[]
            if mode in ['planned_lvm','executed_lvm']:
                if runtime.submissions!=n or runtime.selected is not None or not all(c['returned'] for c in runtime.raw_commands):
                    raise RuntimeError('真实动作历史提交不完整')
                if runtime.guidance_calls or runtime.active:raise RuntimeError('A10普通重规划禁止OGG')
                write_json(directory/'controller_commands.json',runtime.raw_commands)
                trace=[json.loads(line) for line in (directory/'monitor_steps.jsonl').read_text().splitlines()]
            if mode in ['single_lvm','single_ogg']:
                trace=[json.loads(line) for line in (directory/'monitor_steps.jsonl').read_text().splitlines()]
                if mode=='single_lvm' and runtime.guidance_calls:raise RuntimeError('LVM意外启用OGG')
            if len(diagnostics.actions)!=n:raise RuntimeError('诊断未记录全部真实控制命令')
            signature=[first['state'],first['input'],noise['sha']];key=(case['condition'],case['seed'])
            if key in signatures and signatures[key]!=signature:raise RuntimeError('四方法初始状态/输入/噪声不一致')
            signatures[key]=signature
            write_json(directory/'eval_info.json',info);write_json(directory/'executed_actions.json',commands)
            write_json(directory/'actual_protocol.json',{'case':case,'protocol_sha256':digest(args.output/'protocol.json')})
            diagnostic_began=time.perf_counter();prediction_summary=diagnostics.evaluate(audit_sample=mode not in sampled)
            diagnostic_seconds=time.perf_counter()-diagnostic_began;sampled.add(mode)
            success=bool(info['per_episode'][0]['success']);tag=(chosen['task_id'],mode)
            paths=info.get('video_paths',[]);video_size=sum(Path(v).stat().st_size for v in paths)
            keep=not success and retained.get(tag,0)<2 and retained_bytes+video_size<=int(1.5*2**30)
            if keep:retained[tag]=retained.get(tag,0)+len(paths);retained_bytes+=video_size
            else:
                for video in paths:
                    path=Path(video).resolve();path.relative_to(directory.resolve());path.unlink()
                paths=[]
            # Only this newly generated episode's video is eligible for retention/disposal.
            write_json(directory/'video_retention.json',{'retained':keep,'verified':videos,'policy':'max2 failures/task/method, total1.5GiB; old results untouched'})
            return {'success':bool(info['per_episode'][0]['success']),'env_steps':n,'chunk_calls':len(chunk_times),
                'policy_s':sum(chunk_times),'monitor_s':runtime.monitor_s if runtime else 0.,
                'event_replans':runtime.events if runtime else 0,'guidance_calls':runtime.guidance_calls if runtime else 0,
                'scored_steps':sum(r.get('score') is not None for r in trace),'eligible_steps':max(0,n-24) if runtime else 0,
                'planned_skips':runtime.sequence_skips if runtime else 0,
                'planned_cancellations':runtime.cancelled_predictions if runtime else 0,
                'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'initial_signature':signature,'video_paths':paths,'rollout_wall_s':rollout_seconds,
                'shared_capture_s':shared_capture[0],'normalization_s':normalization[0],
                'diagnostic_seconds':diagnostic_seconds,'prediction_summary':prediction_summary,
                'diagnostic_windows_path':str(directory/'prediction_windows.jsonl.gz')}
        finally:env.close()

    # 额外视觉/批量诊断不得改变控制。重跑两条已保存基线，逐动作及首签名对照，另存技术检查不计入3400回合。
    smoke_path=args.output/'interface_smoke.json'
    if not smoke_path.exists():
        write_json(args.output/'pipeline_status.json',{'status':'running','phase':'known_baseline_action_parity_checks'})
        smoke=[];prior=args.base/'object10/consistency_closed_loop_20261111_001'
        previous=[read(p) for p in (prior/'trials').glob('*.json')]
        for method in ['H50','H10']:
            if (args.output/'STOP').exists() or datetime.now(timezone.utc)>=deadline:
                write_json(args.output/'pipeline_status.json',{'status':'paused','reason':'stop_or_deadline_during_checks'});return
            old=next(r for r in previous if r['case']['condition']=='libero_object/task_1/init_3' and r['case']['seed']==1002 and r['case']['method']==method)
            case={'condition':'libero_object/task_1/init_3','seed':1002,'method':method,'id':'interface_'+method,
                'config':settings[method],'_smoke_condition':{'task_id':1,'initial_state_id':3}}
            directory=args.output/'interface_checks'/method
            if directory.exists():
                # 未完成的旧检查保留，下一次显式恢复使用新的attempt。
                index=1
                while directory.with_name(method+f'_retry{index}').exists():index+=1
                directory=directory.with_name(method+f'_retry{index}')
            directory.mkdir(parents=True);result=callback(case,directory)
            expected=read(prior/'attempts'/old['case']['id']/f"attempt_{old['attempt']:03d}"/'executed_actions.json')
            if read(directory/'executed_actions.json')!=expected or any(result[k]!=old['result'][k] for k in ['success','env_steps','chunk_calls','initial_signature']):
                raise RuntimeError('新增观测/诊断改变已核验纯VLA控制命令，不进入夜间扩展')
            smoke.append({'method':method,'exact_actions_equal':True,'success':result['success'],'env_steps':result['env_steps']})
        write_json(smoke_path,smoke)
    if (args.output/'STOP').exists() or datetime.now(timezone.utc)>=deadline:
        write_json(args.output/'pipeline_status.json',{'status':'paused','reason':'stop_or_deadline_after_checks'});return
    write_json(args.output/'pipeline_status.json',{'status':'running','phase':'expanded_closed_loop_and_prediction_audit','planned':len(plan['cases'])})
    try:run_plan(args.output,plan,{m:callback for m in METHODS},max_seconds=max(0.01,(deadline-datetime.now(timezone.utc)).total_seconds()),min_free_gib=5,stop_file=args.output/'STOP')
    finally:
        if (args.output/'trials').is_dir():summary(args.output)
    state=read(args.output/'execution.json')
    if any(digest(Path(meta['checkpoint']))!=meta['checkpoint_sha256'] for meta in weights.values()):raise RuntimeError('冻结权重变化')
    write_json(args.output/'pipeline_status.json',{'status':state['status'],'completed':state['completed'],
        'planned':len(plan['cases']),'checkpoint_sha_unchanged':True,'automatic_expansion':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--resume',action='store_true');p.add_argument('--deadline',required=True);args=p.parse_args()
    try:run(args)
    except BaseException as error:
        if args.output.is_dir():write_json(args.output/'pipeline_status.json',{'status':'error','error':f'{type(error).__name__}: {error}'})
        raise


if __name__=='__main__':main()
