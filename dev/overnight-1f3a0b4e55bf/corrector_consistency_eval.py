"""48回合完整任务：H50/H10/规划LVM/已执行LVM，预冻结单一触发规则，无OGG。"""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
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
OUTPUT=BASE/'object10/consistency_closed_loop_20261111_001'
BRANCH='experiment/executed-closed-loop-pilot-20261111'
METHODS=['H50','H10','planned_lvm','executed_lvm']


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
        row={'method':method,'planned':12,'completed':len(sub),'successes':sum(r['success'] for r in results),
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
    audit=preflight(args.base)
    if subprocess.check_output(['git','-C',str(args.base),'branch','--show-current'],text=True).strip()!=BRANCH:
        raise RuntimeError('不在独立闭环实验分支')
    calibration=read(HERE/'calibration_rule.json');validate_calibration(args.base,calibration)
    args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'preflight.json',audit)
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
    model,predictor_protocol,weight=load_frozen(setting['checkpoint'],setting['factory'],source,'cuda',setting['sha'])
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
    conditions={f'libero_object/task_{t}/init_{i}':{'task_id':t,'initial_state_id':i,'language':suite.tasks[t].language}
                for t in [1,9] for i in [3,4,5]}
    settings={m:{'action_steps':10 if m=='H10' else 50,'video_policy':'all','ogg':False} for m in METHODS}
    protocol={'policy_id':POLICY_ID,'policy_revision':POLICY_SHA,'checkpoint':weight,'calibration':calibration,
        'rule_sha256':digest(HERE/'calibration_rule.json'),'conditions':conditions,'seeds':[1002,1003],
        'control_fps':20,'data_fps':10,'prediction_seconds':1,'horizon_steps':20,'action_offsets':list(range(0,20,2)),
        'max_steps':280,'chunk_size':50,'num_steps':10,'score_start':24,'ogg':False,'runtime':actual,
        'branch':BRANCH,'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},
        'test_result_tuning':False,'automatic_expansion':False}
    plan=make_plan(list(conditions),[1002,1003],settings,protocol)
    if args.resume and read(args.output/'plan.json')!=plan:raise RuntimeError('续跑源码/规则/协议不同')
    write_json(args.output/'protocol.json',protocol)
    signatures={}
    for p in (args.output/'trials').glob('*.json'):
        old=read(p)
        if old['status']=='completed':
            key=(old['case']['condition'],old['case']['seed']);signatures[key]=old['result']['initial_signature']

    def callback(case,directory):
        chosen=conditions[case['condition']];mode=case['method'];first={};noise={};raw_batch={};chunk_times=[]
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
        runtime=ConsistencyLVM(policy,model,predictor_protocol,source,directory,mode,calibration['rule'],normalize) if mode.endswith('_lvm') else None
        selected_steps=0;commands=[]
        def observe(batch):
            raw_batch.clear();raw_batch.update(batch);value=pre(batch)
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
            if runtime:runtime.outer_action=np.array(action[0],dtype=np.float32,copy=True)
            return original_step(action)
        try:
            torch.cuda.reset_peak_memory_stats();select_initial_state(env,chosen['initial_state_id']);set_seed(case['seed'])
            with ExitStack() as stack:
                for obj,name,fn in [(policy,'_get_action_chunk',chunk),(policy.model,'sample_noise',sample_noise),(env,'step',step)]:
                    stack.enter_context(patch.object(obj,name,fn))
                if runtime:
                    stack.enter_context(lazy_controller_capture(unwrapped,runtime));stack.enter_context(runtime.installed())
                else:stack.enter_context(patch.object(policy,'select_action',baseline_select))
                info=eval_policy(env=env,policy=policy,env_preprocessor=env_pre,env_postprocessor=env_post,
                    preprocessor=observe,postprocessor=post,n_episodes=1,max_episodes_rendered=1,
                    videos_dir=directory/'videos',start_seed=case['seed'])
            n=runtime.step if runtime else selected_steps
            videos=[sim.verify_video(p,20) for p in info.get('video_paths',[])]
            if len(commands)!=n or len(videos)!=1 or videos[0]['frames']!=n:raise RuntimeError('视频/动作步数不一致')
            if not runtime and len(chunk_times)!=math.ceil(n/case['config']['action_steps']):raise RuntimeError('纯VLA实际调用计数不符')
            trace=[]
            if runtime:
                if runtime.submissions!=n or runtime.selected is not None or not all(c['returned'] for c in runtime.raw_commands):
                    raise RuntimeError('真实动作历史提交不完整')
                if runtime.guidance_calls or runtime.active:raise RuntimeError('本轮禁止OGG')
                write_json(directory/'controller_commands.json',runtime.raw_commands)
                trace=[json.loads(line) for line in (directory/'monitor_steps.jsonl').read_text().splitlines()]
            signature=[first['state'],first['input'],noise['sha']];key=(case['condition'],case['seed'])
            if key in signatures and signatures[key]!=signature:raise RuntimeError('四方法初始状态/输入/噪声不一致')
            signatures[key]=signature
            write_json(directory/'eval_info.json',info);write_json(directory/'executed_actions.json',commands)
            write_json(directory/'actual_protocol.json',{'case':case,'protocol':protocol})
            return {'success':bool(info['per_episode'][0]['success']),'env_steps':n,'chunk_calls':len(chunk_times),
                'policy_s':sum(chunk_times),'monitor_s':runtime.monitor_s if runtime else 0.,
                'event_replans':runtime.events if runtime else 0,'guidance_calls':0,
                'scored_steps':sum(r['score'] is not None for r in trace),'eligible_steps':max(0,n-24) if runtime else 0,
                'planned_skips':runtime.sequence_skips if runtime else 0,
                'planned_cancellations':runtime.cancelled_predictions if runtime else 0,
                'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'initial_signature':signature,'video_paths':info.get('video_paths',[])}
        finally:env.close()

    write_json(args.output/'pipeline_status.json',{'status':'running','phase':'paired_closed_loop','planned':48})
    try:run_plan(args.output,plan,{m:callback for m in METHODS},max_seconds=2*3600,min_free_gib=5,stop_file=args.output/'STOP')
    finally:
        if (args.output/'trials').is_dir():summary(args.output)
    state=read(args.output/'execution.json')
    if digest(Path(weight['checkpoint']))!=weight['checkpoint_sha256']:raise RuntimeError('冻结权重变化')
    write_json(args.output/'pipeline_status.json',{'status':state['status'],'completed':state['completed'],
        'planned':48,'checkpoint_sha_unchanged':True,'ogg_calls':0,'automatic_expansion':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--resume',action='store_true');args=p.parse_args()
    try:run(args)
    except BaseException as error:
        if args.output.is_dir():write_json(args.output/'pipeline_status.json',{'status':'error','error':f'{type(error).__name__}: {error}'})
        raise


if __name__=='__main__':main()
