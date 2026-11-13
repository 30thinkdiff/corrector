"""Historical-protocol Object10 reuse and isolated 17-method Plus robustness.

Source-only adapter: frozen weights/core/packages remain read-only. No training.
"""
import argparse,copy,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time
import shutil
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import corrector_j36m_object_eval as obj

BASE=Path('/root/autodl-tmp/corrector')
OLD=BASE/'object10/night_20261109_001/evaluation/mlp_s42/05d55ba9452142be/paired'
PLUS_SHA='4976dc30028e805ff8094b55501d532c48fec182'
HERE=Path(__file__).resolve().parent

def guard(args):
    if (args.output/'STOP').exists() or time.time()>=args.deadline:raise RuntimeError('STOP or wall budget')
    for path in ['/root','/root/autodl-tmp']:
        if shutil.disk_usage(path).free<5*2**30:raise RuntimeError('disk free below 5GiB: '+path)


def registry():
    methods={'H50':{'steps':50},'H20':{'steps':20}}
    for name in ['mlp','transformer-h1','transformer-h3','transformer-a10-h1','transformer-a10-h3']:
        for mode in ['lvm','ogg']:methods[name+'_'+mode]={'steps':50,'model':name,'mode':mode}
    methods['hybrid_t1_a10_ogg']={'steps':50,'model':'transformer-h1','guide':'transformer-a10-h1','mode':'ogg'}
    for mode in ['planned_lvm','executed_lvm']:methods[mode]={'steps':50,'model':'transformer-a10-h1','consistency':mode}
    for mode in ['lvm','ogg']:methods['j_'+mode]={'steps':50,'joint':True,'mode':mode}
    return methods


def cases(mode,conditions):
    if mode=='bridge':return [('libero_object/task_1/init_0',1000,m) for m in ['H50','mlp_lvm','mlp_ogg']]+[('libero_object/task_9/init_0',1000,'H20')]
    if mode=='ordinary':return [(f'libero_object/task_{t}/init_{i}',s,m) for t in range(10) for i in range(3) for s in [1000,1001] for m in ['j_lvm','j_ogg']]
    if mode=='plus_smoke':return [('plus/default',9102,m) for m in registry()]
    if mode=='plus':return [(c,s,m) for c in conditions for s in [1000,1001,1002] for m in registry()]
    raise ValueError(mode)


def frozen_models():
    rows={}
    for name in ['mlp','transformer-h1','transformer-h3','transformer-a10-h1','transformer-a10-h3']:
        action='a10' in name
        run=BASE/('object10/action_followup_20261109_001/experiments/runs' if action else 'object10/night_20261109_001/runs')/(name+'_s42')
        rows[name]={'checkpoint':str(run/'best_model.pt'),'factory':str(BASE/('action-code-9274415dd6' if action else 'night-code-1e80242789')/'corrector_models.py')}
    return rows


def dynamic_joint_class():
    import torch
    import corrector_online_runtime as legacy
    from j36m.online import JointOnlineCorrector,unpack
    class DynamicJoint(obj.joint_class()):
        def __init__(self,policy,model,cal,method,source,directory,encoder,force=-1):
            super().__init__(policy,model,cal,'j_lvm_ogg' if method=='j_ogg' else 'j_lvm',source,directory,
                             checkpoint_sha=cal['checkpoint_sha256'],force_step=force)
            State,step,_=legacy.monitor_api(source)
            self.state=State(cooldown_steps=0)
            self.threshold_step=lambda score,state: step(score,state) if self.last_score is not None else ((state.dynamic_threshold[-1] if state.dynamic_threshold else 0.),False)
            self.warmup_steps=4 # same common anchor3 used by historical MLP/H1/H3.
            self.encoder=encoder;self.saved_targets=0
        def encode(self,batch):
            return JointOnlineCorrector.encode(SimpleNamespace(policy=self.encoder),batch)
        def log(self,row):
            row=dict(row,method=self.archive_method,plan_id=self.plan_id)
            if 'score' in row:
                row.update(layer_scores=self.layer_details,score=self.last_score,scoring_valid=self.last_score is not None,
                           prediction_crossed_replan=self.step in self.cross_replan_due)
                self.cross_replan_due.discard(self.step)
                if self.step>=24 and self.last_score is None:row['prediction_scoring_skipped']='no_reliable_layer' if self.layer_details else 'no_due_prediction'
            if row.get('interrupted'):
                targets=unpack(self.target_now)
                row['correction_targets']={k:{'shape':list(v.shape),'rms':float(v.float().square().mean().sqrt())} for k,v in targets.items()}
                if self.saved_targets<1:
                    name=f'joint_target_{self.step:06d}.pt';torch.save({k:v.detach().cpu() for k,v in targets.items()},self.directory/name)
                    row['correction_target_file']=name;self.saved_targets+=1
                else:row['tensor_retention']='summary only after first event; resource bounded, computation unchanged'
            legacy.OnlineCorrector.log(self,row)
    return DynamicJoint


def load(args):
    obj.verify_core(args.core)
    sys.path[:0]=[str(args.core/'scripts'),str(args.core),str(HERE/'expanded_helpers')]
    import torch
    import corrector_data_workbench as archive
    import smolvla_libero_tools as sim
    def fp32(loader,path,cfg,device,**kw):
        cfg.load_vlm_weights=False;cfg.device=device;cfg.pretrained_path=Path(path)
        return loader(path,config=cfg,local_files_only=True,strict=True)
    def plus_configure(*unused):
        os.environ['LIBERO_CONFIG_PATH']=str(args.plus_config)
        sys.path.insert(0,str(args.plus_source))
        import libero.libero as lib
        if not Path(lib.__file__).resolve().is_relative_to(args.plus_source.resolve()):raise ValueError('wrong LIBERO import')
        if subprocess.check_output(['git','-C',str(args.plus_source),'rev-parse','HEAD'],text=True).strip()!=PLUS_SHA:raise ValueError('Plus source SHA')
    with ExitStack() as stack:
        stack.enter_context(patch.object(archive,'load_complete_policy',fp32))
        if args.mode=='plus':stack.enter_context(patch.object(sim,'configure',plus_configure))
        holder=obj.initialize(args)
    policy,pre,post,models,hashes=holder
    encoder=copy.deepcopy(policy).eval().requires_grad_(False)
    encoder.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    if next(policy.model.vlm_with_expert.vlm.parameters()).dtype!=torch.float32:raise ValueError('historical VLA must remain FP32')
    from corrector_hybrid import load_frozen
    legacy={};protocols={};weights={}
    for name,item in frozen_models().items():
        if args.mode=='ordinary' and name!='mlp':continue
        model,protocol,weight=load_frozen(Path(item['checkpoint']),Path(item['factory']),args.runtime/'vla-corrector-reference','cuda')
        if protocol['prediction_seconds']!=1 or protocol['anchor_history']!=3:raise ValueError('legacy timing/anchor changed')
        legacy[name]=model;protocols[name]=protocol;weights[name]=weight
    cal=obj.read(args.calibration)
    if cal['checkpoint_sha256']!=hashes['j']:raise ValueError('J layer calibration checkpoint mismatch')
    return SimpleNamespace(policy=policy,pre=pre,post=post,joint=models['j'],hashes=hashes,encoder=encoder,
                           legacy=legacy,protocols=protocols,weights=weights,cal=cal)


def ordinary_conditions():
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_object']()
    return {f'libero_object/task_{t}/init_{i}':{'task_id':t,'init':i,'language':suite.tasks[t].language} for t in range(10) for i in range(3)}


def probe_plus(args,conditions):
    import numpy as np
    from libero.libero import benchmark
    from lerobot.envs.configs import LiberoEnv
    import smolvla_libero_tools as sim
    results=[]
    for name,c in conditions.items():
        guard(args)
        slot=benchmark.libero_task_map['libero_object'][c['task_id']];task=benchmark.task_maps['libero_object'][slot]
        if task.name!=c['task_name']:raise ValueError('Plus task name/index mismatch')
        kw={'language':c['language']}
        if c.get('default_task'):kw.update(name=c['default_task'],bddl_file=c['default_task']+'.bddl',init_states_file=c['default_task']+'.pruned_init')
        with patch.dict(benchmark.task_maps['libero_object'],{slot:task._replace(**kw)}):
            cfg=LiberoEnv(task='libero_object',task_ids=[c['task_id']],fps=20,episode_length=60,
                observation_height=256,observation_width=256,control_mode='relative',is_libero_plus=True,hard_reset=True,init_states=True,
                camera_name_mapping={'agentview_image':'camera1','robot0_eye_in_hand_image':'camera2'})
            env=sim.make_vector(cfg)
            try:
                env.envs[0].unwrapped.init_state_id=0;obs,_=env.reset(seed=9003)
                env.step(np.array([[0,0,0,0,0,0,-1]],dtype=np.float32))
                results.append({'condition':name,'task_id':c['task_id'],'task_name':task.name,'reset_and_one_step':'passed'})
                obj.write(args.output/'environment_probe.json',{'status':'running','completed':len(results),'expected':len(conditions),'conditions':results})
            finally:env.close()
    obj.write(args.output/'environment_probe.json',{'status':'passed','completed':len(results),'expected':len(conditions),'conditions':results,'scope':'one-step environment/asset check, not success-rate trials'})


def run(args,mode,h,conditions):
    import numpy as np,torch
    from libero.libero import benchmark
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from lerobot.policies.factory import make_pre_post_processors
    from corrector_data_workbench import cached_policy_snapshot
    from corrector_online_runtime import OnlineCorrector
    from corrector_hybrid import HybridCorrector
    from corrector_consistency_control import ConsistencyLVM
    from corrector_executed_monitor import lazy_controller_capture
    from studies.policy_generalization.corrector.batch_evaluation import make_plan,run_plan
    import smolvla_libero_tools as sim
    class BoundedHybrid(HybridCorrector):
        def detector_target(self,expected,observed):
            if not self.target_files:return super().detector_target(expected,observed)
            self.last_event=f'event_{self.step:04d}'
            self.log({'step':self.step,'event':self.last_event,'tensor_retention':'summary only after first event'})
            return (expected-observed).detach()
    output=args.output/mode;spec=registry();wanted=cases(mode,conditions)
    selected={m:{**spec[m],'uses_corrector':m not in ['H50','H20'],'corrector_sha256':h.hashes['j'] if m.startswith('j_') else h.weights[spec[m]['model']]['checkpoint_sha256'] if 'model' in spec[m] else None,'video_policy':'all'} for _,_,m in wanted}
    protocol={'mode':mode,'source_sha256':obj.digest(__file__),'object_adapter_sha256':obj.digest(HERE/'corrector_j36m_object_eval.py'),
        'core_manifest':obj.BASE_MANIFEST,'policy_revision':'a24f1f380461b665158d52acd140152c79e8b36d','vla_dtype':'float32',
        'joint_visual_encoder_dtype':'bfloat16','joint_checkpoint_sha256':h.hashes['j'],'legacy_weights':h.weights,
        'fps':20,'data_fps':10,'due_steps':20,'first_due_step':24,'horizon':280,'chunk_size':50,'denoise':10,
        'threshold':'original robust dynamic history, window15/bootstrap15/cooldown0; no frozen-history wrapper',
        'joint_layer_calibration_sha256':obj.digest(args.calibration),'prediction_retention':'original single-action retained; legacy A10 keeps its own sequence validity rules',
        'conditions':conditions,'plus_revision':PLUS_SHA if args.mode=='plus' else None,'full_target_tensors_per_joint_trial':1}
    plan=make_plan(list(conditions),sorted({s for _,s,_ in wanted}),selected,protocol)
    wanted=set(wanted);plan['cases']=[c for c in plan['cases'] if (c['condition'],c['seed'],c['method']) in wanted]
    action_pre,action_post=make_pre_post_processors(policy_cfg=h.policy.config,pretrained_path=Path(cached_policy_snapshot(False)),preprocessor_overrides={'device_processor':{'device':'cuda'}})
    def callback(case,directory):
        guard(args)
        c=conditions[case['condition']];setting=spec[case['method']];policy=h.policy
        policy.config.n_action_steps=setting['steps'];policy.reset()
        if policy._queues['action'] or policy._queues['action'].maxlen!=setting['steps']:raise ValueError('stale queue')
        cams=list(policy.config.image_features)
        cfg=LiberoEnv(task='libero_object',task_ids=[c['task_id']],fps=20,episode_length=280,
            observation_height=256,observation_width=256,control_mode='relative',is_libero_plus=args.mode=='plus',
            hard_reset=True if args.mode=='plus' else False,init_states=True,
            camera_name_mapping={'agentview_image':cams[0].split('.')[-1],'robot0_eye_in_hand_image':cams[1].split('.')[-1]})
        stack=ExitStack()
        if args.mode=='plus':
            slot=benchmark.libero_task_map['libero_object'][c['task_id']];task=benchmark.task_maps['libero_object'][slot]
            overrides={'language':c['language']}
            if c.get('default_task'):overrides.update(name=c['default_task'],bddl_file=c['default_task']+'.bddl',init_states_file=c['default_task']+'.pruned_init')
            stack.enter_context(patch.dict(benchmark.task_maps['libero_object'],{slot:task._replace(**overrides)}))
        env=None
        try:
            env=sim.make_vector(cfg);raw=env.envs[0].unwrapped;raw.init_state_id=c['init'];envpre,envpost=cfg.get_env_processors()
            batch_raw={};commands=[];normal=[];chunks=[];queue=[];initial={};runtime=None
            def normalize(action):
                batch={k:v.detach().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in batch_raw.items()};batch['action']=torch.tensor(np.array(action,copy=True))[None]
                value=action_pre(batch)['action'].reshape(1,7).float()
                if not np.allclose(action_post(value).detach().cpu().numpy().reshape(7),action,rtol=1e-5,atol=1e-5):raise ValueError('normalization round-trip')
                return value
            if setting.get('joint'):
                runtime=dynamic_joint_class()(policy,h.joint,h.cal,case['method'],args.runtime/'vla-corrector-reference',directory,h.encoder)
            elif setting.get('consistency'):
                rule=obj.read(args.rule)['rule'];runtime=ConsistencyLVM(policy,h.legacy[setting['model']],h.protocols[setting['model']],args.runtime/'vla-corrector-reference',directory,setting['consistency'],rule,normalize)
            elif setting.get('guide'):
                runtime=BoundedHybrid(policy,h.legacy[setting['model']],h.protocols[setting['model']],h.legacy[setting['guide']],h.protocols[setting['guide']],args.runtime/'vla-corrector-reference',directory,label=case['method'])
            elif setting.get('model'):
                runtime=OnlineCorrector(policy,h.legacy[setting['model']],h.protocols[setting['model']],setting['mode'],args.runtime/'vla-corrector-reference',directory,fps=20,stride=1,eta=1.,guidance_every=3,cooldown=0)
            class Capture:
                in_reset=False;outer_action=None
                def controller_step(self,original,action,*a,**kw):
                    if self.in_reset:return original(action,*a,**kw)
                    value=np.asarray(action,dtype=np.float32)
                    if self.outer_action is None or not np.array_equal(value,self.outer_action):raise ValueError('submitted controller action mismatch')
                    commands.append(value.tolist())
                    if setting.get('consistency'):
                        runtime.outer_action=value;return runtime.controller_step(original,action,*a,**kw)
                    return original(action,*a,**kw)
            capture=Capture();original_env=env.step;original_chunk=policy._get_action_chunk;original_noise=policy.model.sample_noise
            def submit(action):capture.outer_action=np.asarray(action[0],dtype=np.float32).copy();return original_env(action)
            def chunk(*a,**kw):
                if policy._queues['action']:raise ValueError('inference called with nonempty action cache')
                torch.cuda.synchronize();t=time.perf_counter();value=original_chunk(*a,**kw);torch.cuda.synchronize()
                if value.shape!=(1,50,7):raise ValueError('expected original chunk50')
                chunks.append({'step':len(commands),'seconds':time.perf_counter()-t});return value
            def noise(*a,**kw):
                v=original_noise(*a,**kw)
                if not (directory/'first_noise.pt').exists():torch.save(v.detach().cpu(),directory/'first_noise.pt')
                return v
            def observe(batch):
                batch_raw.clear();batch_raw.update(batch);v=h.pre(batch)
                if not initial:
                    state=torch.as_tensor(raw._env.get_sim_state()).clone();torch.save(state,directory/'initial_sim_state.pt')
                    if c.get('historical_compat'):
                        reference=obj.read(args.core/'studies/policy_generalization/corrector/j36m/historical_plus_reference.json')
                        data=getattr(raw._env.sim.data,'_data',raw._env.sim.data)
                        for key,item in reference['state'].items():
                            actual=torch.as_tensor(np.asarray(raw._init_states[0]).copy()) if key=='selected_init' else torch.from_numpy(np.array(getattr(data,key),copy=True))
                            expected=torch.tensor(item['values'],dtype=getattr(torch,item['dtype'].removeprefix('torch.'))).reshape(item['shape'])
                            torch.testing.assert_close(actual,expected,rtol=0,atol=1e-8)
                    tensors={k:x.detach().cpu() for k,x in v.items() if isinstance(x,torch.Tensor)}
                    hashed=hashlib.sha256()
                    for k,x in sorted(tensors.items()):hashed.update(str((k,str(x.dtype),tuple(x.shape))).encode());hashed.update(x.contiguous().view(torch.uint8).numpy().tobytes())
                    initial.update(state_sha256=hashlib.sha256(state.numpy().tobytes()).hexdigest(),input_sha256=hashed.hexdigest())
                    # One full input per condition/seed, references + hashes for other methods.
                    audit=args.output/'input_audit'/hashlib.sha256((case['condition']+str(case['seed'])).encode()).hexdigest()[:20]
                    audit.mkdir(parents=True,exist_ok=True);ref=audit/'first_policy_input.pt'
                    if ref.exists():
                        previous=torch.load(ref,weights_only=True)
                        for k,x in tensors.items():torch.testing.assert_close(x,previous[k],rtol=0,atol=0)
                    else:torch.save(tensors,ref)
                    initial['input_file']=str(ref)
                return v
            torch.cuda.reset_peak_memory_stats();set_seed(case['seed']);started=time.perf_counter()
            versions={prefix+'/'+key:p._version for prefix,model in [('joint',h.joint),*h.legacy.items()] for key,p in model.named_parameters()}
            stack.enter_context(patch.object(env,'step',submit));stack.enter_context(patch.object(policy,'_get_action_chunk',chunk));stack.enter_context(patch.object(policy.model,'sample_noise',noise))
            stack.enter_context(lazy_controller_capture(raw,capture))
            if runtime:stack.enter_context(runtime.installed())
            selected=policy.select_action
            def select(*a,**kw):
                before=len(policy._queues['action']);v=selected(*a,**kw);normal.append(v.detach().cpu().reshape(-1).tolist())
                queue.append({'step':len(commands),'before':before,'after':len(policy._queues['action']),'chunk_calls':len(chunks)})
                return v
            stack.enter_context(patch.object(policy,'select_action',select))
            info=eval_policy(env=env,policy=policy,env_preprocessor=envpre,env_postprocessor=envpost,preprocessor=observe,postprocessor=h.post,
                n_episodes=1,max_episodes_rendered=1,videos_dir=directory/'videos',start_seed=case['seed'])
            stack.close();wall=time.perf_counter()-started;success=obj.one_success(info)
            if any(p._version!=versions[name+'/'+key] or p.grad is not None for name,model in [('joint',h.joint),*h.legacy.items()] for key,p in model.named_parameters()):raise ValueError('frozen predictor changed')
            videos=[sim.verify_video(p,20) for p in info['video_paths']]
            if len(videos)!=1 or videos[0]['frames']!=len(commands) or len(normal)!=len(commands):raise ValueError('action/video/observation timebase mismatch')
            obj.write(directory/'eval_info.json',info);obj.write(directory/'actions.json',{'submitted_env_actions':commands,'normalized_policy_actions':normal})
            obj.write(directory/'queue_trace.json',queue);obj.write(directory/'chunk_calls.json',chunks)
            initial['noise_sha256']=hashlib.sha256(torch.load(directory/'first_noise.pt',weights_only=True).contiguous().numpy().tobytes()).hexdigest()
            signature={k:initial[k] for k in ['state_sha256','input_sha256','noise_sha256']}
            sigdir=args.output/'paired_signatures';sigdir.mkdir(exist_ok=True)
            sigfile=sigdir/(hashlib.sha256((case['condition']+str(case['seed'])).encode()).hexdigest()[:20]+'.json')
            if sigfile.exists() and obj.read(sigfile)!=signature:raise ValueError('paired state/input/noise differ')
            if not sigfile.exists():obj.write(sigfile,signature)
            obj.write(directory/'actual_protocol.json',{'case':case,'first':initial,'protocol_sha256':obj.digest(output/'plan.json'),'video':videos[0],
                'recording_limit':'count/order verified; no FPS-only frame-phase claim','language':raw.task_description,
                'joint_predictor_input_dtype':'bfloat16 encoder outputs promoted to float32' if setting.get('joint') else None})
            return {'success':success,'env_steps':len(commands),'chunk_calls':len(chunks),'policy_s':sum(x['seconds'] for x in chunks),
                'monitor_s':runtime.monitor_s if runtime else 0.,'event_replans':runtime.events if runtime else 0,'guidance_calls':runtime.guidance_calls if runtime else 0,
                'cancelled_condition_predictions':runtime.cancelled_predictions if runtime else 0,'sequence_prediction_skips':runtime.sequence_skips if runtime else 0,
                'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'episode_wall_s':wall,'video_paths':info['video_paths']}
        finally:
            stack.close()
            if env:env.close()
    run_plan(output,plan,{m:callback for m in selected},max_seconds=max(1,args.deadline-time.time()),min_free_gib=5,stop_file=args.output/'STOP')
    execution=obj.read(output/'execution.json')
    if execution['status']!='completed':raise RuntimeError('paused phase: '+str(execution))
    if mode=='bridge':verify_bridge(output)
    obj.write(args.output/(mode+'_receipt.json'),{'status':'passed','execution':execution,'protocol':protocol})


def verify_bridge(output):
    import torch,av
    previous={}
    for p in (OLD/'trials').glob('*.json'):
        r=obj.read(p);c=r['case'];previous[c['condition'],c['seed'],c['method']]=r
    for p in (output/'trials').glob('*.json'):
        r=obj.read(p);c=r['case'];method=c['method'].removeprefix('mlp_');old=previous[c['condition'],c['seed'],method]
        for k in ['success','env_steps','chunk_calls','event_replans','guidance_calls']:
            if r['result'][k]!=old['result'][k]:raise ValueError('historical bridge result differs: '+c['method']+'/'+k)
        directory=output/'attempts'/c['id']/f'attempt_{r["attempt"]:03d}'
        olddir=OLD/'attempts'/old['case']['id']/f'attempt_{old["attempt"]:03d}'
        for filename in ['initial_sim_state.pt','first_noise.pt']:
            if (olddir/filename).exists():torch.testing.assert_close(torch.load(directory/filename,weights_only=True),torch.load(olddir/filename,weights_only=True),rtol=0,atol=0)
        if (olddir/'first_policy_input.pt').exists():
            current=torch.load(obj.read(directory/'actual_protocol.json')['first']['input_file'],weights_only=True);past=torch.load(olddir/'first_policy_input.pt',weights_only=True)
            for k,x in current.items():torch.testing.assert_close(x,past[k],rtol=0,atol=0)
        if old['video_artifacts']:
            def frames(path):
                h=hashlib.sha256()
                with av.open(str(path)) as container:
                    for frame in container.decode(video=0):h.update(frame.to_ndarray(format='rgb24').tobytes())
                return h.hexdigest()
            if frames(r['result']['video_paths'][0])!=frames(OLD/old['video_artifacts'][0]['path']):raise ValueError('historical bridge video frames differ')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['ordinary','plus']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--core',type=Path,default=BASE/'dev/j36m-35ecdd7488bf');p.add_argument('--runtime',type=Path,default=BASE/'runtime')
    p.add_argument('--training',type=Path,default=BASE/'j36m_longrun_20261113_001/training');p.add_argument('--calibration',type=Path,default=Path('/root/corrector_j36m_object_20261113_002/j_calibration.json'))
    p.add_argument('--plus-source',type=Path,default=BASE/'runtime/libero-plus-4976dc3');p.add_argument('--plus-config',type=Path,default=BASE/'plus_resources_20261113_001/config')
    p.add_argument('--conditions',type=Path,default=HERE/'plus_conditions.json');p.add_argument('--rule',type=Path,default=BASE/'dev/consistency-control-49cb16490eab/calibration_rule.json')
    p.add_argument('--hours',type=float,default=8);args=p.parse_args()
    if not 0<args.hours<=24:raise ValueError('budget 0..24h')
    args.output.mkdir(parents=True,exist_ok=True);lock=(args.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.deadline=time.time()+args.hours*3600;args.j_checkpoint=args.training/'J_s42/best.pt';args.c0_checkpoint=args.training/'C0_s42/best.pt'
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:(args.output/'STOP').touch())
    state={'status':'running','phase':'preflight','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};obj.write(args.output/'status.json',state)
    try:
        guard(args)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():raise ValueError('GPU occupied')
        h=load(args);conditions=ordinary_conditions() if args.mode=='ordinary' else obj.read(args.conditions)['conditions']
        if args.mode=='plus':
            state['phase']='environment_probe';obj.write(args.output/'status.json',state);probe_plus(args,conditions)
        for phase in ['bridge','ordinary'] if args.mode=='ordinary' else ['plus_smoke','plus']:
            state['phase']=phase;obj.write(args.output/'status.json',state);run(args,phase,h,conditions)
        state.update(status='completed',phase='completed')
    except BaseException as e:state.update(status='paused' if (args.output/'STOP').exists() or time.time()>=args.deadline else 'error',error=f'{type(e).__name__}: {e}');raise
    finally:obj.write(args.output/'status.json',state)


if __name__=='__main__':main()
