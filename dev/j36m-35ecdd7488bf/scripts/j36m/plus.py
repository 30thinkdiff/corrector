"""Strict LIBERO-Plus adapter, reuses batch_evaluation and legacy control runtime.

Only actual GPU evaluation can produce success rows. Source/assets must exist;
this module never substitutes ordinary Object10 or downloads dependencies.
"""
from contextlib import ExitStack
import hashlib,inspect,json,os,subprocess,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from corrector_data_workbench import digest,write_json,load_complete_policy,cached_policy_snapshot
from .model import Joint36M
from .online import JointOnlineCorrector

PLUS_SHA='4976dc30028e805ff8094b55501d532c48fec182'
POLICY_SHA='a24f1f380461b665158d52acd140152c79e8b36d'
INSTRUCTION='pick the alphabet soup and place it in the basket'
CONDITIONS={'default':653,'view15':666,'view5':656,'view10':661}
METHODS={'H50':50,'H20':20,'j_monitor':50,'j_lvm':50,'j_lvm_ogg':50}


def plus_plan(checkpoint_sha,calibration_sha,phase='development',monitor=False):
    if phase not in ('development','heldout','calibration','online-smoke'):raise ValueError('phase')
    conditions=['default','view15'] if phase=='development' else ['view5','view10'] if phase=='heldout' else ['default']
    methods={k:{'n_action_steps':v,'checkpoint_sha256':checkpoint_sha,'calibration_sha256':calibration_sha,'uses_corrector':k.startswith('j_'),
                    'corrector_sha256':checkpoint_sha if k.startswith('j_') else None}
             for k,v in METHODS.items() if k!='j_monitor' or monitor or phase in ('calibration','online-smoke')}
    if phase=='calibration':methods={'j_monitor':methods['j_monitor']}
    if phase=='online-smoke':methods={k:methods[k] for k in ('H50','H20','j_monitor','j_lvm','j_lvm_ogg')}
    seeds=[9000,9001] if phase=='calibration' else [1000] if phase=='online-smoke' else [1000,1001,1002]
    from studies.policy_generalization.corrector.batch_evaluation import make_plan
    return make_plan(conditions,seeds,methods,{'phase':phase,'policy_revision':POLICY_SHA,'plus_revision':PLUS_SHA,
        'language':INSTRUCTION,'fps':20,'chunk_size':50,'denoise_steps':10,'due_steps':20,
        'checkpoint_sha256':checkpoint_sha,'calibration_sha256':calibration_sha,'scope':'LIBERO-Plus camera diagnostics, not Object10 benchmark'})


def selected_task(benchmark,condition):
    task_id=CONDITIONS[condition];slot=benchmark.libero_task_map['libero_object'][task_id]
    original=benchmark.task_maps['libero_object'][slot];base=original.name.split('_view_',1)[0]
    if condition!='default':
        expected=base+f'_view_{condition.removeprefix("view")}_15_100_0_0_initstate_0'
        if original.name!=expected:raise ValueError('Plus task slot/name mismatch')
        selected=original._replace(language=INSTRUCTION)
    else:selected=original._replace(name=base,language=INSTRUCTION,bddl_file=base+'.bddl',init_states_file=base+'.pruned_init')
    return task_id,slot,original,selected


def run_plus(args,guard):
    if subprocess.check_output(['git','-C',str(args.plus_source),'rev-parse','HEAD'],text=True).strip()!=PLUS_SHA:
        raise ValueError('LIBERO-Plus source revision mismatch')
    os.environ.update(HF_HOME=str(args.runtime/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      LIBERO_CONFIG_PATH=str(args.plus_config),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    import sys
    sys.path.insert(0,str(args.plus_source));sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from libero.libero import benchmark
    import libero.libero as imported
    if not Path(imported.__file__).resolve().is_relative_to(args.plus_source.resolve()):raise ValueError('ordinary LIBERO imported instead of fixed Plus')
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    import smolvla_libero_tools as sim
    from studies.policy_generalization.corrector.batch_evaluation import run_plan
    import yaml
    reference_path=Path(__file__).resolve().parents[2]/'studies/policy_generalization/corrector/j36m/historical_plus_reference.json'
    historical=json.loads(reference_path.read_text())
    config_file=args.plus_config/'config.yaml'
    if not config_file.is_file():raise FileNotFoundError('independent Plus config.yaml missing')
    config_paths=yaml.safe_load(config_file.read_text());asset_root=Path(config_paths['assets'])
    marker=asset_root/'.lerobot_camera_assets.json'
    if not marker.is_file() or json.loads(marker.read_text())['revision']!=historical['asset_revision']:raise ValueError('validated historical Plus asset revision absent; do not substitute ordinary assets')
    bddl=args.plus_source/'libero/libero/bddl_files/libero_object'/(historical['base_task']+'.bddl')
    if digest(bddl)!=historical['base_bddl_sha256']:raise ValueError('historical base BDDL changed')
    if subprocess.check_output(['git','-C',str(args.plus_source),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('modified frozen Plus source')
    guard();snapshot=Path(cached_policy_snapshot(False))
    cfg=SmolVLAConfig.from_pretrained(snapshot,local_files_only=True);cfg.n_action_steps=50;cfg.device='cuda'
    policy=load_complete_policy(SmolVLAPolicy.from_pretrained,snapshot,cfg,'cuda').to('cuda').eval().requires_grad_(False)
    if (cfg.chunk_size,cfg.num_steps)!=(50,10):raise ValueError('frozen policy action/denoise protocol differs')
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved['protocol']['variant'] not in ('J','J_equal','JC'):raise ValueError('requires actual formal J checkpoint')
    from .training import source_hash
    if saved['protocol']['source']!=source_hash():raise ValueError('best checkpoint code protocol differs from deployed J runtime')
    joint=Joint36M(saved['input_rms']).to('cuda');joint.load_state_dict(saved['model']);joint.eval().requires_grad_(False)
    cal=json.loads(args.calibration.read_text());checkpoint_sha=digest(args.checkpoint)
    if cal['training_manifest_sha256']!=digest(args.config):raise ValueError('calibration/training manifest mismatch')
    if cal['checkpoint_sha256']!=checkpoint_sha:raise ValueError('calibration bound to another best checkpoint')
    if args.phase!='calibration' and cal['status']!='frozen':raise ValueError('independent normal calibration not frozen')
    if args.phase=='heldout':
        if not args.physical_reference or not args.physical_reference.is_file():raise ValueError('development physical reference required')
        if not args.development_receipt:raise ValueError('development receipt required')
        freeze=json.loads(args.development_receipt.read_text())
        if freeze['checkpoint_sha256']!=checkpoint_sha or freeze['calibration_sha256']!=digest(args.calibration) or freeze['status']!='completed':
            raise ValueError('heldout requires frozen completed development receipt; no tuning from retained views')
    plan=plus_plan(checkpoint_sha,digest(args.calibration),args.phase,monitor=True if args.phase in ('online-smoke','calibration') else args.include_monitor)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    def callback(case,directory):
        guard();torch.cuda.reset_peak_memory_stats();condition=case['condition'];task_id,slot,original,task=selected_task(benchmark,condition)
        policy.config.n_action_steps=METHODS[case['method']];policy.reset()
        if policy._queues['action'].maxlen!=METHODS[case['method']]:raise RuntimeError('stale action queue')
        camera_keys=list(policy.config.image_features)
        if len(camera_keys)!=2:raise ValueError('two trained camera streams required')
        camera_names=[k.removeprefix('observation.images.') for k in camera_keys]
        config=LiberoEnv(task='libero_object',task_ids=[task_id],fps=20,episode_length=280,
                  observation_height=256,observation_width=256,control_mode='relative',is_libero_plus=True,hard_reset=True,init_states=True,
                  camera_name_mapping={'agentview_image':camera_names[0],'robot0_eye_in_hand_image':camera_names[1]})
        parameter_versions={k:p._version for k,p in joint.named_parameters()}
        events=None;steps=0;chunk=[];raw_actions=[];norm_actions=[];first={};start=time.perf_counter()
        if case['method'].startswith('j_'):
            events=JointOnlineCorrector(policy,joint,cal,case['method'],args.source_root,directory,
                                       checkpoint_sha=checkpoint_sha,force_step=20 if args.phase=='online-smoke' else -1)
        with patch.dict(benchmark.task_maps['libero_object'],{slot:task}):
            env=sim.make_vector(config)
            try:
                raw=env.envs[0].unwrapped;raw.init_state_id=0
                envpre,envpost=config.get_env_processors();set_seed(case['seed'])
                def observe(batch):
                    if not first:
                        data=getattr(raw._env.sim.data,'_data',raw._env.sim.data)
                        state={k:torch.from_numpy(np.array(getattr(data,k),copy=True)) for k in ('qpos','qvel','act','mocap_pos','mocap_quat')}
                        state['selected_init']=torch.as_tensor(np.asarray(raw._init_states[0]).copy())
                        for key,expected in historical['state'].items():
                            tensor=torch.tensor(expected['values'],dtype=getattr(torch,expected['dtype'].removeprefix('torch.'))).reshape(expected['shape'])
                            torch.testing.assert_close(state[key],tensor,rtol=0,atol=1e-8)
                        if type(raw._env.robots[0].robot_model).__name__!=historical['robot'] or raw.num_steps_wait!=10:raise ValueError('historical robot/settling protocol changed')
                        reference=args.physical_reference if args.physical_reference else output/'physical_reference.pt'
                        if args.phase=='heldout' and not reference.is_file():raise ValueError('heldout needs development physical reference')
                        if reference.exists():
                            ref=torch.load(reference,weights_only=True)
                            for k,v in state.items():torch.testing.assert_close(v,ref[k],rtol=0,atol=1e-8)
                        else:torch.save(state,reference)
                        if raw.task_description!=INSTRUCTION or raw._env.language_instruction!=INSTRUCTION:raise ValueError('Plus effective language changed')
                        first['checked']=True;torch.save(state,directory/'initial_state.pt')
                    value=pre(batch)
                    if not (directory/'first_input.pt').exists():torch.save({k:v.detach().cpu() for k,v in value.items() if isinstance(v,torch.Tensor)},directory/'first_input.pt')
                    return value
                original_chunk=policy._get_action_chunk;original_noise=policy.model.sample_noise;original_step=raw.step
                def timed(*a,**kw):
                    torch.cuda.synchronize();t=time.perf_counter();value=original_chunk(*a,**kw);torch.cuda.synchronize();chunk.append(time.perf_counter()-t);return value
                def noise(*a,**kw):
                    value=original_noise(*a,**kw)
                    path=directory/'first_noise.pt'
                    if not path.exists():torch.save(value.detach().cpu(),path)
                    return value
                def step(action):
                    nonlocal steps
                    raw_actions.append(np.asarray(action).tolist());steps+=1;return original_step(action)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(policy,'_get_action_chunk',timed));stack.enter_context(patch.object(policy.model,'sample_noise',noise));stack.enter_context(patch.object(raw,'step',step))
                    if events:stack.enter_context(events.installed())
                    original_select=policy.select_action
                    def select(batch,*a,**kw):
                        tokens=batch['observation.language.tokens'];mask=batch['observation.language.attention_mask']
                        tokenizer=policy.model.vlm_with_expert.processor.tokenizer
                        text=tokenizer.decode(tokens[0][mask[0].bool()].tolist(),skip_special_tokens=True).strip()
                        if text!=INSTRUCTION:raise ValueError('actual policy tokens differ from frozen instruction')
                        value=original_select(batch,*a,**kw);norm_actions.append(value.detach().cpu().tolist());return value
                    stack.enter_context(patch.object(policy,'select_action',select))
                    info=eval_policy(env=env,policy=policy,env_preprocessor=envpre,env_postprocessor=envpost,
                        preprocessor=observe,postprocessor=post,n_episodes=1,max_episodes_rendered=1,
                        videos_dir=directory/'videos',start_seed=case['seed'])
                if parameter_versions!={k:p._version for k,p in joint.named_parameters()} or any(p.grad is not None for p in joint.parameters()):raise RuntimeError('frozen J weights changed')
                successes=[v for t in info['per_task'] for v in t['metrics']['successes']]
                if len(successes)!=1 or type(successes[0]) is not bool:raise ValueError('need actual per-trial success boolean')
                videos=[sim.verify_video(p,20) for p in info.get('video_paths',[])]
                if len(videos)!=1 or videos[0]['frames']!=steps:raise ValueError('one video frame per control step not verified')
                arrays=np.asarray(raw_actions);continuity=float(np.linalg.norm(np.diff(arrays,axis=0),axis=-1).mean()) if len(arrays)>1 else None
                write_json(directory/'actions.json',{'submitted_env_actions':raw_actions,'action_recording_site':'LiberoEnv.step input, before internal controller transformations','normalized_policy_actions':norm_actions})
                write_json(directory/'eval_info.json',info)
                for old in (output/'attempts').glob('*/*/actual_protocol.json'):
                    oldp=json.loads(old.read_text())
                    if oldp['case']['condition']==case['condition'] and oldp['case']['seed']==case['seed']:
                        other=old.parent
                        if (other/'first_noise.pt').exists():
                            torch.testing.assert_close(torch.load(directory/'first_noise.pt',weights_only=True),torch.load(other/'first_noise.pt',weights_only=True),rtol=0,atol=0)
                        if (other/'first_input.pt').exists():
                            aa=torch.load(directory/'first_input.pt',weights_only=True);bb=torch.load(other/'first_input.pt',weights_only=True)
                            if set(aa)!=set(bb):raise ValueError('paired input keys differ')
                            for key in aa:torch.testing.assert_close(aa[key],bb[key],rtol=0,atol=0)
                write_json(directory/'actual_protocol.json',{'case':case,'language':INSTRUCTION,'plus_revision':PLUS_SHA,'asset_revision':historical['asset_revision'],'policy_revision':POLICY_SHA,
                    'model_sha256':checkpoint_sha,'monitor_source_sha256':events.monitor_sha if events else None,'calibration_sha256':digest(args.calibration),'fps':20,'chunk_size':50,'num_steps':10,
                    'n_action_steps':METHODS[case['method']],'camera_keys':camera_keys,'video':videos[0],
                    'evaluator_source_sha256':digest(Path(inspect.getfile(eval_policy))),'video_phase':'source audit required; frame-count/order verified, no fabricated substep alignment','force_event':events is not None and args.phase=='online-smoke','physical_state_checked':True,'predictor_parameters_unchanged':True})
                return {'success':successes[0],'env_steps':steps,'chunk_calls':len(chunk),'policy_s':sum(chunk),
                        'monitor_s':events.monitor_s if events else 0.,'event_replans':events.events if events else 0,
                        'guidance_calls':events.guidance_calls if events else 0,'cancelled_condition_predictions':events.cancelled_predictions if events else 0,
                        'failure_stage':'not_annotated','action_continuity_units':'mixed submitted env action vector units; not physical velocity',
                        'action_continuity_mean_l2':continuity,
                        'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                        'episode_wall_s':time.perf_counter()-start,'video_paths':info.get('video_paths',[])}
            finally:
                env.close()
                if events:
                    events.pending.clear();events.history.clear();events.z_now=None;events.context_now=None;events.target_now=None
    output_floor=args.data_min_free_gib if output.stat().st_dev==Path('/root/autodl-tmp').stat().st_dev else args.system_min_free_gib
    run_plan(output,plan,{m:callback for m in {c['method'] for c in plan['cases']}},max_seconds=args.max_hours*3600,
             min_free_gib=output_floor,stop_file=args.stop_file)
    execution=json.loads((output/'execution.json').read_text())
    if execution['status']=='completed':
        if args.phase=='online-smoke':audit_online_smoke(output)
        write_json(output/'frozen_receipt.json',{'status':'completed','phase':args.phase,'checkpoint_sha256':checkpoint_sha,
                                               'calibration_sha256':digest(args.calibration)})
    return execution


def audit_online_smoke(output):
    trials=[json.loads(p.read_text()) for p in (output/'trials').glob('*.json')]
    rows={r['case']['method']:r for r in trials if r['status']=='completed'}
    if set(rows)!={'H50','H20','j_monitor','j_lvm','j_lvm_ogg'}:raise ValueError('complete five-method online smoke required')
    def directory(method):
        row=rows[method];return output/'attempts'/row['case']['id']/f'attempt_{row["attempt"]:03d}'
    raw={m:json.loads((directory(m)/'actions.json').read_text()) for m in rows}
    for field in ('submitted_env_actions','normalized_policy_actions'):
        if raw['H50'][field]!=raw['j_monitor'][field]:raise RuntimeError('passive monitor changed actual policy/environment actions')
    if rows['j_lvm']['result']['guidance_calls']!=0 or rows['j_lvm_ogg']['result']['guidance_calls']<=0:raise RuntimeError('LVM/OGG routing failed')
    traces={m:[json.loads(s) for s in (directory(m)/'monitor_steps.jsonl').read_text().splitlines()] for m in ('j_lvm','j_lvm_ogg')}
    first={m:next(r for r in rr if r.get('interrupted')) for m,rr in traces.items()}
    if any(r['step']!=20 or not r.get('debug_forced') for r in first.values()):raise RuntimeError('forced step20 lacked reliable prediction; do not claim online smoke passed')
    for key in ('step','score','threshold','trigger','layer_scores'):
        if first['j_lvm'][key]!=first['j_lvm_ogg'][key]:raise RuntimeError('pre-fork detector mismatch: '+key)
    write_json(output/'online_smoke_audit.json',{'status':'passed','passive_actions_equal_H50':True,
                'same_first_event':True,'lvm_guidance_calls':0,'ogg_guidance_calls':rows['j_lvm_ogg']['result']['guidance_calls']})
