"""A10-H1被动到期一致性Pilot：采集真实控制器提交/视觉，逐步保持原H50动作不变。"""
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

from smolvla_horizon_pilot import BASE, EXPECTED_VERSIONS, preflight, read
from smolvla_baseline_benchmark import POLICY_ID, POLICY_SHA, select_initial_state
import smolvla_training_workbench as wb
import smolvla_libero_tools as sim
try:
    from batch_evaluation import make_plan, run_plan, write_json
except ModuleNotFoundError:
    from studies.policy_generalization.corrector.batch_evaluation import make_plan, run_plan, write_json

HERE=Path(__file__).resolve().parent
OUTPUT=BASE/'object10/executed_consistency_pilot_20261111_002'
BRANCH='experiment/executed-consistency-pilot-20261111-v2'


def run(args):
    lock=(args.base/'.executed-consistency-pilot.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.output.exists():raise RuntimeError('新输出目录已存在，不覆盖/自动续跑')
    if read(args.base/'object10/hybrid_pilot_20261111_001/pipeline_status.json').get('status')!='completed':
        raise RuntimeError('前序混合Pilot未确认完成')
    audit=preflight(args.base)
    branch=subprocess.check_output(['git','-C',str(args.base),'branch','--show-current'],text=True).strip()
    if branch!=BRANCH:raise RuntimeError('不在独立被动监测实验分支')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'preflight.json',audit)
    def pause(*a):
        (args.output/'STOP').touch();print('已请求在当前回合结束后暂停。',flush=True)
    signal.signal(signal.SIGINT,pause);signal.signal(signal.SIGTERM,pause)
    import numpy as np
    import torch
    from importlib.metadata import version
    from corrector_hybrid import load_frozen,feature_evidence,digest
    from corrector_hybrid_pilot import role_paths
    from corrector_executed_monitor import PassiveExecutedMonitor,lazy_controller_capture
    actual={k:version(k) for k in EXPECTED_VERSIONS}
    if actual!=EXPECTED_VERSIONS:raise RuntimeError('运行库版本不一致')
    sim.configure(argparse.Namespace(workdir=args.base/'runtime',sim_fps=20,max_steps=280),wb)
    setting=role_paths(args.base)['a10'];source=args.base/'runtime/vla-corrector-reference'
    model,predictor_protocol,weight=load_frozen(setting['checkpoint'],setting['factory'],source,'cuda',setting['sha'])
    if (predictor_protocol['model_type']!='transformer-action' or predictor_protocol['h_window']!=1 or
        predictor_protocol['action_steps']!=10 or predictor_protocol['fps']!=10 or
        predictor_protocol['k_step']!=10 or predictor_protocol['prediction_seconds']!=1):
        raise RuntimeError('冻结A10-H1时基/形状协议不兼容')
    evidence=feature_evidence(args.base/'object10',predictor_protocol,predictor_protocol)
    write_json(args.output/'compatibility.json',evidence)
    write_json(args.output/'frozen_a10_protocol.json',predictor_protocol)
    baseline=args.base/'object10/horizon_pilot_20261111_001'
    old_plan=read(baseline/'plan.json')
    if read(baseline/'execution.json')['status']!='completed' or old_plan['protocol']['policy_revision']!=POLICY_SHA:
        raise RuntimeError('没有可对照的已完成纯H50轨迹')
    refs={}
    for path in (baseline/'trials').glob('*.json'):
        r=read(path);c=r['case']
        if c['method']=='H50' and c['seed']==1000:
            if r['status']!='completed':raise RuntimeError('纯H50参考回合不完整')
            p=baseline/'attempts'/c['id']/f"attempt_{r['attempt']:03d}"
            refs[c['condition']]={'record':r,'actions':read(p/'executed_actions.json')}
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
        raise RuntimeError('VLA执行配置与纯H50不一致')
    cfg.device,cfg.pretrained_path,cfg.load_vlm_weights='cuda',snapshot,False
    policy=SmolVLAPolicy.from_pretrained(snapshot,config=cfg).to('cuda').eval().requires_grad_(False)
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,
        preprocessor_overrides={'device_processor':{'device':'cuda'}})
    action_pre,action_post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,
        preprocessor_overrides={'device_processor':{'device':'cuda'}})
    cameras=list(policy.config.image_features)
    if cameras!=evidence['shared_feature_metadata']['camera_keys']:raise RuntimeError('相机顺序不同')
    suite=benchmark.get_benchmark_dict()['libero_object']()
    conditions={f'libero_object/task_{t}/init_{i}':{'task_id':t,'initial_state_id':i,'language':suite.tasks[t].language}
        for t in [1,9] for i in [0,2]}
    if set(refs)!=set(conditions):raise RuntimeError('纯H50参考任务/初始状态不一致')
    protocol={'scope':'passive due-consistency, not future prediction or closed-loop control',
        'policy_id':POLICY_ID,'policy_revision':POLICY_SHA,'checkpoint':weight,
        'data_fps':10,'control_fps':20,'horizon_steps':20,'prediction_seconds':1,
        'action_offsets':list(range(0,20,2)),'visual_shape':[1,128,960],
        'executed_first_due':20,'planned_first_due':24,'common_evaluation_start':24,
        'action_steps':50,'chunk_size':50,'num_steps':10,'seed':1000,
        'conditions':conditions,'branch':branch,'runtime':actual,
        'thresholds_applied':False,'interruptions_allowed':False,'ogg_allowed':False,'retraining':False,
        'normalization':'actual raw controller command -> same frozen policy preprocessor as training',
        'control_capture':'underlying simulator step, excluding reset settling; outer vector input checked equal',
        'clock':'Z_t before u_t, after u_(t-1) returns; sampled history t-20+[0,2,...,18]',
        'baseline_root':str(baseline),'automatic_expansion':False,
        'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},
        'annotation_status':'no independent per-window anomaly labels; outcome-stratified scores are not normal/anomaly truth'}
    write_json(args.output/'protocol.json',protocol)
    plan=make_plan(list(conditions),[1000],{'passive_dual_score':{'video_policy':'all'}},protocol)

    def callback(case,directory):
        chosen=conditions[case['condition']];reference=refs[case['condition']]
        env_cfg=LiberoEnv(task='libero_object',task_ids=[chosen['task_id']],fps=20,episode_length=280,
            observation_height=256,observation_width=256,control_mode='relative',is_libero_plus=False,
            camera_name_mapping={'agentview_image':cameras[0].removeprefix('observation.images.'),
                                 'robot0_eye_in_hand_image':cameras[1].removeprefix('observation.images.')})
        env=sim.make_vector(env_cfg);env_pre,env_post=env_cfg.get_env_processors()
        first={};noise={};raw_batch={};chunk_times=[];normalization_errors=[]
        def normalize(raw):
            batch={k:(v.detach().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in raw_batch.items()}
            batch['action']=torch.tensor(raw.copy(),dtype=torch.float32)[None]
            normalized=action_pre(batch)['action'].reshape(1,7).float()
            roundtrip=action_post(normalized).detach().cpu().numpy().reshape(7)
            err=float(np.max(np.abs(roundtrip-raw)));normalization_errors.append(err)
            if not np.allclose(roundtrip,raw,rtol=1e-5,atol=1e-5):
                raise RuntimeError('控制器动作与冻结预处理器不能原样往返，停止并报告')
            return normalized
        runtime=PassiveExecutedMonitor(policy,model,predictor_protocol,source,directory,normalize,reference['actions'])
        unwrapped=env.envs[0].unwrapped
        original_env_step=env.step
        def outer_step(action):
            if action.shape!=(1,7):raise RuntimeError('外层控制动作shape错误')
            runtime.outer_action=np.array(action[0],dtype=np.float32,copy=True)
            return original_env_step(action)
        def observe(batch):
            raw_batch.clear();raw_batch.update(batch)
            value=pre(batch)
            if not first:
                state=torch.as_tensor(unwrapped._env.get_sim_state()).cpu().numpy()
                first['state']=hashlib.sha256(state.tobytes()).hexdigest();h=hashlib.sha256()
                for key in sorted(value):
                    if isinstance(value[key],torch.Tensor):
                        v=value[key].detach().cpu().contiguous()
                        h.update(str((key,str(v.dtype),tuple(v.shape))).encode());h.update(v.view(torch.uint8).numpy().tobytes())
                first['input']=h.hexdigest()
            return value
        original_chunk=policy._get_action_chunk;original_noise=policy.model.sample_noise
        def chunk(*a,**kw):
            if policy._queues['action']:raise RuntimeError('非空旧队列下重新生成动作块')
            torch.cuda.synchronize();began=time.perf_counter();out=original_chunk(*a,**kw);torch.cuda.synchronize()
            chunk_times.append(time.perf_counter()-began)
            return out
        def sample_noise(*a,**kw):
            value=original_noise(*a,**kw)
            if not noise:noise['sha']=hashlib.sha256(value.detach().cpu().float().numpy().tobytes()).hexdigest()
            return value
        try:
            torch.cuda.reset_peak_memory_stats();select_initial_state(env,chosen['initial_state_id']);set_seed(case['seed'])
            with ExitStack() as stack:
                stack.enter_context(lazy_controller_capture(unwrapped,runtime))
                for obj,name,fn in [(env,'step',outer_step),
                    (policy,'_get_action_chunk',chunk),(policy.model,'sample_noise',sample_noise)]:
                    stack.enter_context(patch.object(obj,name,fn))
                stack.enter_context(runtime.installed())
                info=eval_policy(env=env,policy=policy,env_preprocessor=env_pre,env_postprocessor=env_post,
                    preprocessor=observe,postprocessor=post,n_episodes=1,max_episodes_rendered=1,
                    videos_dir=directory/'videos',start_seed=case['seed'])
            runtime.close()
            videos=[sim.verify_video(p,20) for p in info.get('video_paths',[])]
            if len(videos)!=1 or videos[0]['frames']!=runtime.step:raise RuntimeError('视频与控制计数不一致')
            old=reference['record']['result'];success=bool(info['per_episode'][0]['success'])
            if success!=old['success'] or runtime.step!=old['env_steps'] or len(chunk_times)!=old['chunk_calls']:
                raise RuntimeError('被动采集改变了纯H50成败/步数/调用')
            for key,value in [('initial_state_sha256',first['state']),('first_input_sha256',first['input']),('first_noise_sha256',noise['sha'])]:
                if value!=old[key]:raise RuntimeError('被动采集与纯H50初始条件不一致：'+key)
            trace=[json.loads(line) for line in (directory/'monitor_steps.jsonl').read_text().splitlines()]
            if runtime.events or runtime.guidance_calls or any(r.get('trigger') or r.get('interrupted') for r in trace):
                raise RuntimeError('被动监测意外干预控制')
            write_json(directory/'eval_info.json',info)
            write_json(directory/'actual_protocol.json',{'case':case,'protocol':protocol,'normalizer_max_roundtrip_error':max(normalization_errors)})
            return {'success':success,'env_steps':runtime.step,'chunk_calls':len(chunk_times),'policy_s':sum(chunk_times),
                'events':0,'guidance_calls':0,'baseline_unchanged':True,
                'planned_scored_steps':sum(r['planned_score'] is not None and r['step']>=24 for r in trace),
                'executed_scored_steps':sum(r['executed_score'] is not None and r['step']>=24 for r in trace),
                'common_eligible_steps':max(0,runtime.step-24),
                'planned_skips':runtime.sequence_skips,'planned_cancellations':runtime.cancelled_predictions,
                'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'video_paths':info.get('video_paths',[])}
        finally:
            if not runtime.feature_file.closed:runtime.feature_file.close()
            env.close()

    write_json(args.output/'pipeline_status.json',{'status':'running','phase':'passive_collection_and_scoring'})
    run_plan(args.output,plan,{'passive_dual_score':callback},max_seconds=3600,min_free_gib=5,stop_file=args.output/'STOP')
    state=read(args.output/'execution.json')
    if state['status']=='completed':
        if digest(Path(weight['checkpoint']))!=weight['checkpoint_sha256']:raise RuntimeError('冻结权重变化')
        write_json(args.output/'pipeline_status.json',{'status':'completed','trials':4,'checkpoint_sha_unchanged':True,
            'controller_actions_equal_to_baseline':True,'interruptions':0,'ogg_calls':0,'automatic_expansion':False})
    else:write_json(args.output/'pipeline_status.json',{'status':'paused','reason':state.get('reason')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=OUTPUT);args=p.parse_args()
    try:run(args)
    except BaseException as error:
        if args.output.is_dir():write_json(args.output/'pipeline_status.json',{'status':'error','error':f'{type(error).__name__}: {error}'})
        raise


if __name__=='__main__':main()
