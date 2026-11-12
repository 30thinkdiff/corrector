"""冻结A1/A10混合OGG：兼容性→真实缓存梯度检查→强制接口检查→12回合自然Pilot。"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time
from unittest.mock import patch

from smolvla_horizon_pilot import BASE, EXPECTED_VERSIONS, preflight, read
from smolvla_baseline_benchmark import POLICY_ID, POLICY_SHA, select_initial_state
try:
    from batch_evaluation import make_plan, run_plan, write_json
except ModuleNotFoundError:
    from studies.policy_generalization.corrector.batch_evaluation import make_plan, run_plan, write_json
import smolvla_training_workbench as wb
import smolvla_libero_tools as sim

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE/'object10/hybrid_pilot_20261111_001'


def role_paths(base):
    return {
        'single': {'checkpoint':base/'object10/night_20261109_001/runs/transformer-h1_s42/best_model.pt',
                   'factory':base/'night-code-1e80242789/corrector_models.py',
                   'sha':'db5f037acb24a2f0954bd9f7e5debd6974a43117411ebf157ec23fe68a02c7b3'},
        'a10': {'checkpoint':base/'object10/action_followup_20261109_001/experiments/runs/transformer-a10-h1_s42/best_model.pt',
                'factory':base/'action-code-9274415dd6/corrector_models.py',
                'sha':'c7185ff584d2ed224249fce506cb11c9069cc8446010010de78dcac284f5336f'}}


def paired_signatures(root):
    groups = {}
    for path in (root/'trials').glob('*.json'):
        r = read(path)
        if r['status'] != 'completed':
            continue
        key = (r['case']['condition'], r['case']['seed'])
        signature = tuple(r['result'][name] for name in ['initial_state_sha256','first_input_sha256','first_noise_sha256'])
        groups.setdefault(key,set()).add(signature)
    if any(len(v) != 1 for v in groups.values()):
        raise RuntimeError('A/B/C首状态/输入/噪声不一致，不继续评测')


def summarize(root):
    import statistics
    rows = []
    records = [read(p) for p in (root/'trials').glob('*.json')]
    for name in ['A','B','C']:
        subset = [r for r in records if r['status']=='completed' and r['case']['method']==name]
        results = [r['result'] for r in subset]
        row = {'configuration':name, 'planned':4, 'completed':len(results),
               'successes':sum(r['success'] for r in results),
               'success_rate':sum(r['success'] for r in results)/len(results) if results else None}
        for key in ['env_steps','chunk_calls','policy_s','monitor_s','guidance_s','event_replans','guidance_calls',
                    'score_eligible_steps','scored_steps','sequence_prediction_skips',
                    'cancelled_condition_predictions','peak_torch_allocated_gib']:
            row['mean_'+key] = statistics.mean(r[key] for r in results) if results else None
        eligible = sum(r['score_eligible_steps'] for r in results)
        row['score_coverage'] = sum(r['scored_steps'] for r in results)/eligible if eligible else None
        row['mean_episode_wall_s'] = statistics.mean(r['wall_s'] for r in subset) if subset else None
        rows.append(row)
    import csv
    root.joinpath('analysis').mkdir(exist_ok=True)
    with (root/'analysis/hybrid_overall.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    write_json(root/'analysis/hybrid_summary.json',{'methods':rows,'pilot_only':True,
        'causal_limit':'B-C changes detector coverage, scores and target; does not identify coverage alone',
        'guide_comparison_limit':'A-B share detector policy but subsequent trajectories/events may diverge',
        'scope':'12 natural episodes; separate 3 forced interface checks excluded'})


def run(args):
    lock = (args.base/'.hybrid-pilot.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.output.exists():
        raise RuntimeError('独立Pilot目录已存在，不覆盖/自动续跑')
    if read(args.base/'object10/horizon_pilot_20261111_001/execution.json')['status'] != 'completed':
        raise RuntimeError('上一个horizon Pilot尚未确认完成，不启动')
    audit = preflight(args.base)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'preflight.json',audit)
    def pause(signum, frame):
        (args.output/'STOP').touch()
        print('请求在回合边界暂停，不启动下一阶段。',flush=True)
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    import torch
    import numpy as np
    from importlib.metadata import version
    from corrector_hybrid import load_frozen, compatible, feature_evidence, guidance_update, HybridCorrector, digest
    actual={key:version(key) for key in EXPECTED_VERSIONS}
    if actual != EXPECTED_VERSIONS:
        raise RuntimeError('运行库版本与冻结协议不一致')
    sim.configure(argparse.Namespace(workdir=args.base/'runtime',sim_fps=20,max_steps=280),wb)
    paths = role_paths(args.base)
    for role, prefix in [('single','single'),('a10','a10')]:
        for key in ['checkpoint','factory']:
            override=getattr(args,prefix+'_'+key,None)
            if override is not None:paths[role][key]=override
    source = args.base/'runtime/vla-corrector-reference'
    models, protocols, weights = {}, {}, {}
    for name, setting in paths.items():
        models[name],protocols[name],weights[name] = load_frozen(setting['checkpoint'],setting['factory'],source,'cuda',setting['sha'])
        if protocols[name]['seed']!=42:
            raise RuntimeError('本Pilot固定训练seed42；其他训练seed须另建协议')
        write_json(args.output/('frozen_'+name+'_protocol.json'),protocols[name])
    compatibility=compatible(protocols['single'],protocols['a10'])
    compatibility.update(feature_evidence(args.base/'object10',protocols['single'],protocols['a10']))
    if compatibility['shared_feature_metadata']['policy_revision'] != POLICY_SHA:
        raise RuntimeError('预测器编码器不是当前冻结VLA')
    write_json(args.output/'compatibility.json',compatibility)
    # 使用原缓存实际shape与正常尺度，不拼接两种模型、不修改数据定义。
    ep=protocols['single']['records'][0]
    feature=args.base/'object10/features'/f"episode_{ep['episode']}"
    for name in ['z_q.npy','z_scale.npy','actions.npy']:
        if digest(feature/name)!=ep['files'][name]:
            raise RuntimeError('梯度检查缓存与冻结SHA不符')
    q=np.load(feature/'z_q.npy',mmap_mode='r');scale=np.load(feature/'z_scale.npy',mmap_mode='r')
    actions=np.load(feature/'actions.npy',mmap_mode='r')
    z=torch.tensor(np.asarray(q[0],dtype=np.float32)*np.asarray(scale[0],dtype=np.float32)[:,None],device='cuda')[None]
    future=torch.tensor(np.asarray(q[10],dtype=np.float32)*np.asarray(scale[10],dtype=np.float32)[:,None],device='cuda')[None]
    target=future-z
    probe={}
    for name,model in models.items():
        condition=torch.tensor(np.array(actions[:10] if name=='a10' else actions[0],dtype=np.float32),device='cuda')[None]
        with torch.no_grad():
            pred=model(z[:,None],condition)
        if pred.shape!=z.shape or not torch.isfinite(pred).all():
            raise RuntimeError('缓存输入/输出shape检查失败')
        x=torch.zeros(1,50,32,device='cuda')
        offsets=[2*i for i in range(10)] if name=='a10' else [0]
        x[:,offsets,:7]=condition if name=='a10' else condition[:,None]
        guided,stats=guidance_update(model,z,x,torch.zeros_like(x),torch.tensor([.5],device='cuda'),target,offsets)
        if stats['gradient_l2']<=0 or not torch.isfinite(guided).all() or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('输入梯度不存在/非有限或冻结参数产生梯度，不启动闭环')
        support=[i for i,v in enumerate(stats['per_action_clean_update_l2'][0]) if v>1e-10]
        if not set(support).issubset(offsets) or not support:
            raise RuntimeError('梯度更新位置与动作条件不一致')
        probe[name]={'input_visual':list(z[:,None].shape),'input_action':list(condition.shape),
                     'output':list(pred.shape),'target':list(target.shape),'support':support,**stats}
    write_json(args.output/'shape_gradient_probe.json',probe)
    print(json.dumps({'phase':'shape_gradient_passed','compatibility':compatibility['compatible']},ensure_ascii=False),flush=True)
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
        raise RuntimeError('基础VLA配置不匹配')
    cfg.device,cfg.pretrained_path,cfg.load_vlm_weights='cuda',snapshot,False
    policy=SmolVLAPolicy.from_pretrained(snapshot,config=cfg).to('cuda').eval().requires_grad_(False)
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,
        preprocessor_overrides={'device_processor':{'device':'cuda'}})
    cameras=list(policy.config.image_features)
    if cameras!=compatibility['shared_feature_metadata']['camera_keys']:
        raise RuntimeError('实际VLA相机顺序与特征缓存不同')
    suite=benchmark.get_benchmark_dict()['libero_object']()
    conditions={f'libero_object/task_{t}/init_{i}':{'task_id':t,'initial_state_id':i,'language':suite.tasks[t].language}
                for t in [1,9] for i in [0,2]}
    configs={'A':{'detector':'single','guide':'single','video_policy':'all'},
             'B':{'detector':'single','guide':'a10','video_policy':'all'},
             'C':{'detector':'a10','guide':'a10','video_policy':'all'}}
    protocol={'policy_id':POLICY_ID,'policy_revision':POLICY_SHA,'weights':weights,
        'configuration':configs,'fps':20,'data_fps':10,'prediction_seconds':1,'due_steps':20,
        'action_steps':50,'chunk_size':50,'num_steps':10,'eta':1.,'guidance_every':3,
        'guidance_indices':[0,3,6,9],'seed':1000,'training_seed':42,'conditions':conditions,
        'threshold':'original robust_threshold_step defaults, cooldown=0',
        'warmup_steps':4,'target':'detector_expected_delta - observed_delta',
        'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},'runtime':actual,
        'both_models_resident_for_all_configurations':True,'automatic_expansion':False}
    import subprocess
    protocol['experiment_branch']=subprocess.check_output(['git','-C',str(args.base),'branch','--show-current'],text=True).strip()
    if protocol['experiment_branch']!='experiment/hybrid-corrector-pilot-20261111':
        raise RuntimeError('远程未在独立实验分支，停止')
    write_json(args.output/'protocol.json',protocol)

    def callback(case,directory,force=False):
        chosen=conditions[case['condition']]
        env_cfg=LiberoEnv(task='libero_object',task_ids=[chosen['task_id']],fps=20,
            episode_length=60 if force else 280,observation_height=256,observation_width=256,
            control_mode='relative',is_libero_plus=False,
            camera_name_mapping={'agentview_image':cameras[0].removeprefix('observation.images.'),
                                 'robot0_eye_in_hand_image':cameras[1].removeprefix('observation.images.')})
        env=sim.make_vector(env_cfg);env_pre,env_post=env_cfg.get_env_processors()
        setting=configs[case['method']]
        runtime=HybridCorrector(policy,models[setting['detector']],protocols[setting['detector']],
            models[setting['guide']],protocols[setting['guide']],source,directory,
            label=case['method'],force_step=24 if force else -1)
        from dataclasses import asdict
        write_json(directory/'initial_threshold_state.json',asdict(runtime.state))
        torch.cuda.reset_peak_memory_stats()
        chunk_times=[];first={};noise={};executed=[]
        original_chunk=policy._get_action_chunk
        def chunk(*a,**kw):
            if policy._queues['action']:
                raise RuntimeError('新块推理前存在旧队列')
            torch.cuda.synchronize();start=time.perf_counter();out=original_chunk(*a,**kw);torch.cuda.synchronize()
            chunk_times.append(time.perf_counter()-start)
            if out.shape!=(1,50,7):
                raise RuntimeError('生成的实际动作块shape改变')
            return out
        original_noise=policy.model.sample_noise
        def sample_noise(*a,**kw):
            import hashlib
            value=original_noise(*a,**kw)
            if not noise:
                noise['sha']=hashlib.sha256(value.detach().cpu().float().numpy().tobytes()).hexdigest()
            return value
        def observe(batch):
            import hashlib
            value=pre(batch)
            if not first:
                state=torch.as_tensor(env.envs[0].unwrapped._env.get_sim_state()).cpu().numpy()
                first['state']=hashlib.sha256(state.tobytes()).hexdigest()
                h=hashlib.sha256()
                for key in sorted(value):
                    if isinstance(value[key],torch.Tensor):
                        v=value[key].detach().cpu().contiguous()
                        h.update(str((key,str(v.dtype),tuple(v.shape))).encode());h.update(v.view(torch.uint8).numpy().tobytes())
                first['input']=h.hexdigest()
            return value
        original_step=env.step
        def step(action):
            executed.append(action[0].tolist());return original_step(action)
        try:
            policy.reset();select_initial_state(env,chosen['initial_state_id']);set_seed(case['seed'])
            with ExitStack() as stack:
                for obj,name,fn in [(policy,'_get_action_chunk',chunk),(policy.model,'sample_noise',sample_noise),(env,'step',step)]:
                    stack.enter_context(patch.object(obj,name,fn))
                stack.enter_context(runtime.installed())
                info=eval_policy(env=env,policy=policy,env_preprocessor=env_pre,env_postprocessor=env_post,
                    preprocessor=observe,postprocessor=post,n_episodes=1,max_episodes_rendered=1,
                    videos_dir=directory/'videos',start_seed=case['seed'])
            videos=[sim.verify_video(p,20) for p in info.get('video_paths',[])]
            if len(videos)!=1 or videos[0]['frames']!=runtime.step or len(executed)!=runtime.step:
                raise RuntimeError('视频/控制步数不匹配')
            trace=[json.loads(line) for line in (directory/'monitor_steps.jsonl').read_text().splitlines()]
            updates=[r for r in trace if r.get('type')=='hybrid_guidance']
            scores=[r for r in trace if 'score' in r and r['score'] is not None]
            if len(updates)!=runtime.guidance_calls or len(updates)!=4*runtime.events:
                raise RuntimeError('事件/引导调用次数不匹配')
            if any(r['event'] not in {e['event'] for e in runtime.target_files} for r in updates):
                raise RuntimeError('引导调用没有检测器目标')
            if force and (not runtime.events or not updates or max(r['gradient_l2'] for r in updates)<=0):
                raise RuntimeError('强制控制接口检查没有有效目标/输入梯度')
            write_json(directory/'eval_info.json',info)
            write_json(directory/'executed_actions.json',executed)
            write_json(directory/'targets.json',runtime.target_files)
            write_json(directory/'actual_protocol.json',{'case':case,'protocol':protocol,'forced_check':force,
                'detector':weights[setting['detector']],'guide':weights[setting['guide']],
                'camera_keys':cameras,'source_history_stride':runtime.history_stride,
                'k':runtime.k,'warmup_steps':runtime.warmup_steps})
            result={'success':bool(info['per_episode'][0]['success']),'env_steps':runtime.step,
                'chunk_calls':len(chunk_times),'policy_s':sum(chunk_times),'chunk_latency_s':chunk_times,
                'monitor_s':runtime.monitor_s,'guidance_s':runtime.guide_s,
                'event_replans':runtime.events,'guidance_calls':runtime.guidance_calls,
                'sequence_prediction_skips':runtime.sequence_skips,'cancelled_condition_predictions':runtime.cancelled_predictions,
                'score_eligible_steps':max(0,runtime.step-(runtime.warmup_steps+runtime.k)),
                'scored_steps':len(scores),'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'video_paths':info.get('video_paths',[]),'initial_state_sha256':first['state'],
                'first_input_sha256':first['input'],'first_noise_sha256':noise['sha'],
                'max_gradient_l2':max((r['gradient_l2'] for r in updates),default=None),
                'mean_ogg_loss':sum(r['ogg_loss'] for r in updates)/len(updates) if updates else None,
                'mean_clean_action_update_l2':sum(r['clean_action_update_l2'] for r in updates)/len(updates) if updates else None}
            # 跨配置检查在run_plan写completed后立即于下个callback之前执行。
            paired_signatures(args.output/('forced_check' if force else 'paired'))
            for path in (args.output/('forced_check' if force else 'paired')/'trials').glob('*.json'):
                old=read(path)
                if old['status']=='completed' and old['case']['condition']==case['condition']:
                    if any(old['result'][key]!=result[key] for key in ['initial_state_sha256','first_input_sha256','first_noise_sha256']):
                        raise RuntimeError('配对初始条件不一致')
            print(json.dumps({'phase':'forced_check' if force else 'paired','method':case['method'],
                'condition':case['condition'],'success':result['success'],'events':runtime.events,
                'scored_steps':len(scores),'guide_calls':runtime.guidance_calls},ensure_ascii=False),flush=True)
            return result
        finally:
            env.close()

    def run_stage(name,selected,force):
        stage=args.output/name
        plan=make_plan(selected,[1000],configs,{**protocol,'max_steps':60 if force else 280,'forced_check':force})
        write_json(args.output/'pipeline_status.json',{'status':'running','phase':name})
        run_plan(stage,plan,{m:(lambda c,d:callback(c,d,force)) for m in configs},
                 max_seconds=3600,min_free_gib=5,stop_file=args.output/'STOP')
        paired_signatures(stage)
        if read(stage/'execution.json')['status']!='completed':
            write_json(args.output/'pipeline_status.json',{'status':'paused','phase':name});return False
        return True

    try:
        if not run_stage('forced_check',['libero_object/task_1/init_0'],True):return
        if (args.output/'STOP').exists():
            write_json(args.output/'pipeline_status.json',{'status':'paused','phase':'before_paired'});return
        if not run_stage('paired',list(conditions),False):return
        summarize(args.output/'paired')
        for item in weights.values():
            if digest(Path(item['checkpoint']))!=item['checkpoint_sha256'] or digest(Path(item['factory']))!=item['factory_sha256']:
                raise RuntimeError('冻结权重/工厂在实验期间改变')
        write_json(args.output/'pipeline_status.json',{'status':'completed','phase':'pilot_complete',
            'forced_checks':3,'paired_trials':12,'automatic_expansion':False,'checkpoint_sha_unchanged':True})
    finally:
        if (args.output/'paired/trials').is_dir():
            summarize(args.output/'paired')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    for name in ['single-checkpoint','single-factory','a10-checkpoint','a10-factory']:
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    try:run(args)
    except BaseException as error:
        if args.output.is_dir():
            write_json(args.output/'pipeline_status.json',{'status':'error','error':f'{type(error).__name__}: {error}'})
        raise


if __name__=='__main__':main()
