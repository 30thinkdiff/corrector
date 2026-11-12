"""Replay完整配对诊断样本库；冻结15种配置，成功/失败都保留，回合边界续接。"""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

from corrector_casebank_plan import cases,STUDY
from smolvla_horizon_pilot import BASE,EXPECTED_VERSIONS,preflight,read
from smolvla_baseline_benchmark import POLICY_ID,POLICY_SHA,select_initial_state
from batch_evaluation import run_plan,write_json
import smolvla_training_workbench as wb
import smolvla_libero_tools as sim

HERE=Path(__file__).resolve().parent
BRANCH='experiment/replay-casebank-20261112-v3'
RECORDER_SHA='fddec10c709401dd863814091d525fcd54d41cc6ccf51f7ef8300459cf69b873'


def decode_video_pts(path):
    # 服务器已有PyAV（原verify_video也使用它），不要求额外ffprobe可执行文件。
    import av
    with av.open(str(path)) as container:
        frames=[]
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:raise RuntimeError('视频帧没有真实PTS，禁止按FPS伪造')
            frames.append(float(frame.pts*frame.time_base))
    return frames


def resources(root):
    available=None
    for group in [Path('/sys/fs/cgroup')]:
        try:
            limit=(group/'memory.max').read_text().strip();used=int((group/'memory.current').read_text())
            if limit!='max':available=(int(limit)-used)/2**30
        except (OSError,ValueError):pass
    return {'system_free_gib':shutil.disk_usage(root).free/2**30,'data_free_gib':shutil.disk_usage(BASE).free/2**30,'available_ram_gib':available}


def summarize(root):
    from collections import defaultdict
    records=[read(p) for p in (root/'trials').glob('*.json')];groups=defaultdict(dict)
    for r in records:
        if r['status']=='completed':groups[(r['case']['condition'],r['case']['seed'])][r['case']['method']]=r
    patterns=defaultdict(int);complete=0
    for g in groups.values():
        if all(m in g for m in ['H50','H20','mlp_ogg']):
            patterns[''.join('1' if g[m]['result']['success'] else '0' for m in ['H50','H20','mlp_ogg'])]+=1
        complete+=len(g)==15
    write_json(root/'analysis/casebank_summary.json',{'completed':sum(r['status']=='completed' for r in records),
        'complete_15_method_groups':complete,'observed_patterns':dict(patterns),
        'scope':'outcome-selected diagnostic bank; confirmation seeds separate; not unbiased success-rate benchmark'})


def run(args):
    if not 0<args.hours<=12:raise ValueError('本次执行预算须为0～12小时')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.output.parent/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    limit=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(32768,limit[1]),limit[1]))
    audit=preflight(BASE)
    if subprocess.check_output(['git','-C',str(BASE),'branch','--show-current'],text=True).strip()!=BRANCH:raise RuntimeError('实验分支不一致')
    if (args.output/'plan.json').exists() and not args.resume:raise RuntimeError('结果已存在，不覆盖')
    if (args.output/'STOP').exists():raise RuntimeError('存在STOP，先明确核对，不自动清除')
    selection=read(HERE/'selection.json');settings=selection['methods'];model_list=read(HERE/'models.json')
    write_json(args.output/'pipeline_status.json',{'status':'running','phase':'preflight_and_real_cache_gradient_parity'})
    write_json(args.output/'preflight.json',audit)
    def stop(*unused):(args.output/'STOP').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    import numpy as np
    import torch
    from importlib.metadata import version
    from corrector_hybrid import load_frozen,feature_evidence,digest
    from corrector_casebank_capture import AuditedOnline,AuditedHybrid
    from corrector_online_runtime import OnlineCorrector
    from corrector_consistency_control import ConsistencyLVM
    from corrector_executed_monitor import lazy_controller_capture
    from corrector_overnight_eval import validate_calibration
    if {k:version(k) for k in EXPECTED_VERSIONS}!=EXPECTED_VERSIONS:raise RuntimeError('依赖版本改变，不临时升级或适配')
    source=BASE/'runtime/vla-corrector-reference';models={};protocols={};weights={}
    for item in model_list:
        m,p,w=load_frozen(Path(item['checkpoint']),Path(item['factory']),source,'cuda',item['sha'])
        if p['seed']!=42 or p['h_window']!=item['h_window'] or p.get('action_steps',1)!=item['action_steps']:raise RuntimeError('模型输入协议与固定清单不一致')
        models[item['name']]=m;protocols[item['name']]=p;weights[item['name']]=w
    common=['encoder_policy_revision','source_sha','data_manifest_sha256','fps','k_step','prediction_seconds','anchor_history','records']
    reference=protocols['mlp']
    for name,p in protocols.items():
        if any(p.get(k)!=reference.get(k) for k in common) or p['model']['action_dim']!=7 or p['model']['token_dim']!=960:raise RuntimeError('跨模型特征/动作/时间协议不兼容：'+name)
    evidence=feature_evidence(BASE/'object10',reference,reference)
    calibration=read(HERE/'calibration_rule.json');validate_calibration(BASE,calibration)
    recorder=BASE/'runtime/lerobot/src/lerobot/scripts/lerobot_eval.py'
    if digest(recorder)!=RECORDER_SHA:raise RuntimeError('录制源码SHA不符，不能保证帧与动作相位')
    (args.output/'recording').mkdir(exist_ok=True)
    (args.output/'recording/lerobot_eval.py').write_bytes(recorder.read_bytes())
    write_json(args.output/'recording/evidence.json',{'git_revision':wb.LEROBOT_SHA,'source_sha256':RECORDER_SHA,'tracked_file_clean':True})
    # 用真实训练缓存验证原算法与记录包装的梯度/输出相等，覆盖H3，禁止H1填充成H3。
    ep=reference['records'][0];folder=BASE/'object10/features'/f"episode_{ep['episode']}"
    for name in ['z_q.npy','z_scale.npy','actions.npy']:
        if digest(folder/name)!=ep['files'][name]:raise RuntimeError('真实探针缓存SHA不符')
    q=np.load(folder/'z_q.npy',mmap_mode='r');scale=np.load(folder/'z_scale.npy',mmap_mode='r');actions=np.load(folder/'actions.npy',mmap_mode='r')
    visual=torch.tensor(np.asarray(q[:13],np.float32)*np.asarray(scale[:13],np.float32)[:,:,None],device='cuda')
    probes=[]
    for name,model in models.items():
        p=protocols[name];h=p['h_window'];ctx=visual[3-h:3][None];z=visual[2][None];target=(visual[12]-visual[2])[None]
        condition=torch.tensor(np.array(actions[2:12] if p.get('action_steps',1)==10 else actions[2]),device='cuda')[None].float()
        with torch.no_grad():out=model(ctx if getattr(model,'uses_history',False) else z,condition)
        if out.shape!=z.shape or not torch.isfinite(out).all():raise RuntimeError('真实输入shape探针失败：'+name)
        x=torch.zeros(1,50,32,device='cuda');offsets=list(range(0,20,2)) if p.get('action_steps',1)==10 else [0]
        x[:,offsets,:7]=condition if len(offsets)>1 else condition[:,None];v=torch.zeros_like(x);t=torch.tensor([.5],device='cuda')
        stub=SimpleNamespace(config=SimpleNamespace(rtc_config=None),select_action=lambda *a:None,model=SimpleNamespace(denoise_step=lambda *a:v))
        original=OnlineCorrector(stub,model,p,'ogg',source,args.output/'probes'/name/'original')
        recorded=AuditedOnline(stub,model,p,'ogg',source,args.output/'probes'/name/'recorded',audit_tag=name)
        for runtime in [original,recorded]:runtime.z_now=z;runtime.context_now=ctx;runtime.target_now=target;runtime.active=True
        recorded.last_event='probe'
        expected=original.denoise_step(None,None,x,t);actual=recorded.denoise_step(None,None,x,t)
        if not torch.isfinite(actual).all() or any(p.grad is not None for p in model.parameters()):raise RuntimeError('原OGG记录出现非有限输出/参数梯度：'+name)
        rows=[json.loads(l) for l in (recorded.directory/'monitor_steps.jsonl').read_text().splitlines()]
        if not rows[-1].get('original_update_formula_exact_equal') or rows[-1]['gradient_l2']<=0:raise RuntimeError('同次实际梯度恒等式/真实梯度探针失败')
        probes.append({'model':name,'visual_shape':list(ctx.shape),'action_shape':list(condition.shape),'output_shape':list(out.shape),
                       'recording_output_exact_equal':torch.equal(expected,actual),'max_abs_output_difference':float((expected-actual).abs().max()),
                       'same_call_original_update_exact_equal':True,'original_guidance_function_reused':True,
                       'limit':'separate AMP-BF16 backward calls may vary; gate uses captured same-call original gradient, no precision/parameter changes'})
    write_json(args.output/'shape_gradient_parity.json',probes)
    del q,scale,actions,visual
    sim.configure(argparse.Namespace(workdir=BASE/'runtime',sim_fps=20,max_steps=280),wb)
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from libero.libero import benchmark
    snapshot=Path(snapshot_download(POLICY_ID,revision=POLICY_SHA,local_files_only=True,token=False));cfg=SmolVLAConfig.from_pretrained(snapshot)
    if cfg.chunk_size!=50 or cfg.n_action_steps!=50 or cfg.num_steps!=10 or cfg.rtc_config is not None:raise RuntimeError('基础VLA配置改变')
    cfg.device,cfg.pretrained_path,cfg.load_vlm_weights='cuda',snapshot,False
    policy=SmolVLAPolicy.from_pretrained(snapshot,config=cfg).to('cuda').eval().requires_grad_(False)
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    action_pre,action_post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    cameras=list(policy.config.image_features)
    if cameras!=evidence['shared_feature_metadata']['camera_keys']:raise RuntimeError('相机顺序不一致')
    suite=benchmark.get_benchmark_dict()['libero_object']()
    conditions={c:{'task_id':int(c.split('/task_')[1].split('/')[0]),'initial_state_id':int(c.split('/init_')[1])} for c in {a['condition'] for a in selection['anchors']}}
    for c in conditions.values():c['language']=suite.tasks[c['task_id']].language
    protocol={'study_id':STUDY,'experiment_label':'Replay完整成败组合样本库','policy_id':POLICY_ID,'policy_revision':POLICY_SHA,
        'lerobot_revision':wb.LEROBOT_SHA,'all_weights':weights,'methods':settings,'conditions':conditions,'selection':selection,
        'control_fps':20,'data_fps':10,'prediction_seconds':1,'horizon_steps':20,'score_start':24,'visual_shape':[128,960],
        'chunk_size':50,'num_steps':10,'calibration':calibration,'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},
        'video_policy':'all','training_seed':42,'recording_schema':'replay_casebank_v1',
        'timing_limit':'policy_s includes original OGG and diagnostic writing; audit_s separately recorded; do not sum nested clocks',
        'outcome_selection_bias':True,'automatic_expansion':False,'artifact_budget_gib':4}
    plan={'protocol':protocol,'cases':cases(selection)}
    if (args.output/'plan.json').exists() and read(args.output/'plan.json')!=plan:raise RuntimeError('续跑计划/源码/权重不一致')
    write_json(args.output/'protocol.json',protocol)
    signatures={};written=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    for p in (args.output/'trials').glob('*.json'):
        r=read(p)
        if r['status']=='completed':signatures[r['case']['condition'],r['case']['seed']]=r['result']['initial_signature']

    def callback(case,directory,plain=False):
        nonlocal written
        setting=case['config'];chosen=conditions.get(case['condition']) or case.get('_condition');model_name=setting.get('model')
        policy.config.n_action_steps=setting['action_steps'];policy.reset()
        if policy._queues['action'].maxlen!=setting['action_steps']:raise RuntimeError('动作缓存长度未正确清理')
        env_cfg=LiberoEnv(task='libero_object',task_ids=[chosen['task_id']],fps=20,episode_length=280,observation_height=256,observation_width=256,
            control_mode='relative',is_libero_plus=False,camera_name_mapping={'agentview_image':cameras[0].removeprefix('observation.images.'),'robot0_eye_in_hand_image':cameras[1].removeprefix('observation.images.')})
        env=sim.make_vector(env_cfg);env_pre,env_post=env_cfg.get_env_processors();unwrapped=env.envs[0].unwrapped
        first={};raw_batch={};chunks=[];commands=[];queue=[];selected=[None];noise={};runtime=None
        def normalize(raw):
            batch={k:v.detach().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in raw_batch.items()};batch['action']=torch.tensor(raw.copy())[None]
            value=action_pre(batch)['action'].reshape(1,7).float()
            if not np.allclose(action_post(value).detach().cpu().numpy().reshape(7),raw,rtol=1e-5,atol=1e-5):raise RuntimeError('真实控制命令归一化往返失败')
            return value
        if setting['family']=='original':
            cls=OnlineCorrector if plain else AuditedOnline
            runtime=cls(policy,models[model_name],protocols[model_name],setting['mode'],source,directory,**({} if plain else {'audit_tag':case['method']}))
        elif setting['family']=='hybrid':runtime=AuditedHybrid(policy,models[model_name],protocols[model_name],models[setting['guide']],protocols[setting['guide']],source,directory,label=case['method'],audit_tag=case['method'])
        elif setting['family']=='consistency':runtime=ConsistencyLVM(policy,models[model_name],protocols[model_name],source,directory,setting['mode'],calibration['rule'],normalize)
        class Capture:
            in_reset=False;outer_action=None
            def controller_step(self,original,action,*a,**kw):
                if self.in_reset:return original(action,*a,**kw)
                raw=np.asarray(action,dtype=np.float32)
                if self.outer_action is None or not np.array_equal(raw,self.outer_action):raise RuntimeError('真实控制提交与外层动作不一致')
                step=len(commands);normalized=normalize(raw)
                if selected[0] is None:raise RuntimeError('无相应策略选择的真实动作')
                commands.append({'step':step,'raw_controller_action':raw.tolist(),'processor_normalized_action':normalized.detach().cpu().reshape(-1).tolist(),'returned':False})
                if setting['family']=='consistency':runtime.outer_action=raw;out=runtime.controller_step(original,action,*a,**kw)
                else:out=original(action,*a,**kw)
                commands[-1]['returned']=True;selected[0]=None;return out
        capture=Capture();original_chunk=policy._get_action_chunk;original_noise=policy.model.sample_noise;original_step=env.step
        def chunk(*a,**kw):
            if policy._queues['action']:raise RuntimeError('新推理前未清理旧动作队列')
            torch.cuda.synchronize();began=time.perf_counter();out=original_chunk(*a,**kw);torch.cuda.synchronize()
            if out.shape!=(1,50,7):raise RuntimeError('未生成原50步动作块')
            chunks.append({'call':len(chunks)+1,'step':len(commands),'seconds':time.perf_counter()-began,'shape':list(out.shape)})
            folder=directory/'chunks';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'chunk_{len(chunks):04d}.npz',normalized_actions=out.detach().cpu().float().numpy())
            return out
        def sample_noise(*a,**kw):
            out=original_noise(*a,**kw)
            if not noise:noise['sha']=hashlib.sha256(out.detach().cpu().float().numpy().tobytes()).hexdigest()
            return out
        def observe(batch):
            raw_batch.clear();raw_batch.update(batch);value=pre(batch)
            if not first:
                first['state']=hashlib.sha256(torch.as_tensor(unwrapped._env.get_sim_state()).cpu().numpy().tobytes()).hexdigest();h=hashlib.sha256()
                for k in sorted(value):
                    if isinstance(value[k],torch.Tensor):
                        v=value[k].detach().cpu().contiguous();h.update(str((k,str(v.dtype),tuple(v.shape))).encode());h.update(v.view(torch.uint8).numpy().tobytes())
                first['input']=h.hexdigest()
            return value
        def step(action):capture.outer_action=np.array(action[0],dtype=np.float32,copy=True);return original_step(action)
        try:
            torch.cuda.reset_peak_memory_stats();select_initial_state(env,chosen['initial_state_id']);set_seed(case['seed'])
            with ExitStack() as stack:
                for obj,name,fn in [(policy,'_get_action_chunk',chunk),(policy.model,'sample_noise',sample_noise),(env,'step',step)]:stack.enter_context(patch.object(obj,name,fn))
                stack.enter_context(lazy_controller_capture(unwrapped,capture))
                if runtime:stack.enter_context(runtime.installed())
                actual_select=policy.select_action
                def select(*a,**kw):
                    index=len(commands);before=len(policy._queues['action']);calls=len(chunks)
                    if not runtime and (not before)!=(index%setting['action_steps']==0):raise RuntimeError('纯VLA队列消费错误')
                    out=actual_select(*a,**kw);selected[0]=out.detach().clone()
                    queue.append({'step':index,'queue_before':before,'queue_after':len(policy._queues['action']),'chunk_calls_before':calls,'chunk_calls_after':len(chunks),'plan_id':len(chunks),'selected_normalized_action':out.detach().cpu().reshape(-1).tolist()});return out
                stack.enter_context(patch.object(policy,'select_action',select));began=time.perf_counter()
                info=eval_policy(env=env,policy=policy,env_preprocessor=env_pre,env_postprocessor=env_post,preprocessor=observe,postprocessor=post,
                    n_episodes=1,max_episodes_rendered=1,videos_dir=directory/'videos',start_seed=case['seed'])
            wall=time.perf_counter()-began;n=len(commands);video=info.get('video_paths',[])
            if not n or not all(c['returned'] for c in commands) or len(queue)!=n or len(video)!=1 or sim.verify_video(video[0],20)['frames']!=n:raise RuntimeError('视频、队列、真实控制记录不完整')
            if runtime and (runtime.step!=n or runtime.active):raise RuntimeError('监测/控制时钟不一致')
            if not runtime and len(chunks)!=math.ceil(n/setting['action_steps']):raise RuntimeError('纯VLA调用次数异常')
            signature=[first['state'],first['input'],noise['sha']];key=(case['condition'],case['seed'])
            if key in signatures and signatures[key]!=signature:raise RuntimeError('配对状态/输入/首次噪声不一致')
            signatures[key]=signature
            pts=decode_video_pts(video[0])
            if len(pts)!=n or abs(pts[0])>1e-8 or any(a>=b for a,b in zip(pts,pts[1:])):raise RuntimeError('实际视频PTS不符')
            with (directory/'frame_map.jsonl').open('w') as f:
                for i,t in enumerate(pts):f.write(json.dumps({'frame_index':i,'pts_seconds':t,'env_step':i,'phase':'before_action'})+'\n')
            write_json(directory/'executed_actions.json',[c['raw_controller_action'] for c in commands]);write_json(directory/'controller_commands.json',commands)
            write_json(directory/'queue_trace.json',queue);write_json(directory/'chunk_calls.json',chunks);write_json(directory/'eval_info.json',info)
            if runtime and hasattr(runtime,'target_files'):write_json(directory/'targets.json',runtime.target_files);write_json(directory/'tensor_manifest.json',runtime.tensor_files)
            write_json(directory/'actual_protocol.json',{'case':case,'protocol_sha256':digest(args.output/'protocol.json'),'fps':20,'visual_shape':[128,960],
                'monitor_horizon_steps':20 if runtime else None,'monitor_first_due_step':24 if runtime else None,
                'predictor_protocol':{k:protocols[model_name][k] for k in ['model_type','model','h_window','anchor_history','fps','prediction_seconds','action_steps'] if k in protocols[model_name]} if model_name else None,
                'detector':weights.get(model_name),'guide':weights.get(setting.get('guide',model_name)) if setting.get('mode')=='ogg' else None,
                'recording_source_sha256':RECORDER_SHA,'recording_phase':'frame n before submitted action n; final post-action frame not retained'})
            trace=[json.loads(l) for l in (directory/'monitor_steps.jsonl').read_text().splitlines()] if (directory/'monitor_steps.jsonl').exists() else []
            guides=[r for r in trace if 'guidance_loss' in r or 'ogg_loss' in r]
            if runtime and len(guides)!=runtime.guidance_calls:raise RuntimeError('OGG计数与逐步记录不一致')
            result={'success':bool(info['per_episode'][0]['success']),'env_steps':n,'chunk_calls':len(chunks),'policy_s':sum(c['seconds'] for c in chunks),
                'monitor_s':runtime.monitor_s if runtime else 0.,'guidance_calls':runtime.guidance_calls if runtime else 0,'event_replans':runtime.events if runtime else 0,
                'guidance_s':getattr(runtime,'guide_seconds',0.),
                'scored_steps':sum(r.get('score') is not None for r in trace),'eligible_steps':max(0,n-24) if runtime else 0,
                'sequence_prediction_skips':runtime.sequence_skips if runtime else 0,'cancelled_condition_predictions':runtime.cancelled_predictions if runtime else 0,
                'audit_s':getattr(runtime,'audit_s',0.),'rollout_wall_s':wall,'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'video_paths':video,'initial_signature':signature,'initial_state_sha256':signature[0],'first_input_sha256':signature[1],'first_noise_sha256':signature[2],
                'capture_schema':'replay_casebank_v1','data_completeness':{'video':True,'frame_map':True,'raw_and_normalized_commands':True,'queue':True,'chunks':True,'ogg_arrays':'all recorded calls' if guides and not plain else 'not applicable or plain sentinel'}}
            size=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file());written+=size
            r=resources(args.output)
            if written>4*2**30 or r['system_free_gib']<5 or r['data_free_gib']<3 or r['available_ram_gib'] is not None and r['available_ram_gib']<6:
                write_json(args.output/'resource_pause.json',{'written_bytes':written,**r});(args.output/'STOP').touch()
            print(json.dumps({'condition':case['condition'],'seed':case['seed'],'method':case['method'],'success':result['success'],'env_steps':n},ensure_ascii=False),flush=True)
            return result
        finally:env.close()

    # 采集开/关的控制等价性先在独立目录验证，技术回合不计入样本库。
    if not (args.output/'control_parity.json').exists():
        setting=settings['transformer-h1_ogg'];case={'id':'control_parity','condition':'libero_object/task_1/init_0','seed':1000,'method':'transformer-h1_ogg','config':setting,'_condition':{'task_id':1,'initial_state_id':0}}
        parity_results={}
        for mode in ['plain','recorded']:
            directory=args.output/'interface_checks'/mode
            if directory.exists():raise RuntimeError('未完成技术检查现场已存在，需审查，不覆盖重试')
            directory.mkdir(parents=True);parity_results[mode]=callback(case,directory,plain=mode=='plain')
        left=read(args.output/'interface_checks/plain/executed_actions.json');right=read(args.output/'interface_checks/recorded/executed_actions.json')
        if len(left)!=len(right) or not np.allclose(np.asarray(left),np.asarray(right),rtol=1e-6,atol=1e-7) or parity_results['plain']['success']!=parity_results['recorded']['success']:raise RuntimeError('采集包装改变实际控制命令/结局，正式队列未启动')
        write_json(args.output/'control_parity.json',{'exact_executed_actions_equal':np.array_equal(np.asarray(left),np.asarray(right)),
            'max_abs_command_difference':float(np.max(np.abs(np.asarray(left)-np.asarray(right)))),'rtol':1e-6,'atol':1e-7,
            'condition':case['condition'],'seed':1000,'steps':len(left),'same_outcome':True,'excluded_from_formal_trials':True})
    write_json(args.output/'pipeline_status.json',{'status':'running','phase':'complete_casebank_trials','planned':len(plan['cases'])})
    run_plan(args.output,plan,{m:callback for m in settings},max_seconds=args.hours*3600,min_free_gib=5,stop_file=args.output/'STOP')
    summarize(args.output);execution=read(args.output/'execution.json')
    for w in weights.values():
        if digest(Path(w['checkpoint']))!=w['checkpoint_sha256'] or digest(Path(w['factory']))!=w['factory_sha256']:raise RuntimeError('实验期间冻结权重/工厂改变')
    write_json(args.output/'pipeline_status.json',{'status':execution['status'],'phase':'finished','completed':execution['completed'],'planned':execution['planned'],'checkpoint_sha_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--hours',type=float,default=8);p.add_argument('--resume',action='store_true');args=p.parse_args()
    try:run(args)
    except BaseException as e:
        if args.output.exists():write_json(args.output/'pipeline_status.json',{'status':'error','error':type(e).__name__+': '+str(e)})
        raise
