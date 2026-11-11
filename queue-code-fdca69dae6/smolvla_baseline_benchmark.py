"""冻结公开SmolVLA的原始LIBERO基线及可选Corrector；H50/H20无需预测器权重。

顺序运行task×真实init_state_id×seed，一回合一视频/输入审计，断连按已完成回合续跑。
这是本项目公开checkpoint基线，不宣称复现Corrector论文的MetaWorld/SmolVLA数值。
"""
import argparse
from pathlib import Path
import sys
import time
from contextlib import ExitStack
import shutil
import inspect
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from studies.policy_generalization.corrector.batch_evaluation import make_plan, run_plan
except ModuleNotFoundError as error:
    if not error.name.startswith("studies"):
        raise
    from batch_evaluation import make_plan, run_plan  # Notebook将此模块同步到脚本同目录

POLICY_ID = "lerobot/smolvla_libero"
POLICY_SHA = "a24f1f380461b665158d52acd140152c79e8b36d"


def select_initial_state(env, index):
    """必须在reset前设置，禁止用%取模把越界请求悄悄变成已有状态。"""
    subenv = env.envs[0].unwrapped
    if type(index) is not int or not 0 <= index < len(subenv._init_states):
        raise ValueError("请求的真实初始状态编号越界")
    subenv.init_state_id = index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"], default="libero_object")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--initial-state-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument('--methods',nargs='+',choices=['H50','H20','monitor','lvm','ogg'])
    parser.add_argument('--corrector-checkpoint',type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--control-implementation',choices=['reference','student'],default='reference')
    parser.add_argument('--monitor-stride',type=int,default=1)
    parser.add_argument('--monitor-cooldown',type=int,default=0)
    parser.add_argument('--guidance-eta',type=float,default=1.)
    parser.add_argument('--guidance-every',type=int,default=3)
    parser.add_argument('--debug-force-step',type=int,default=-1,
                        help='只用于验证事件/OGG调用链；带强制事件的结果不能当正式性能')
    parser.add_argument('--video-policy',choices=['all','failures','none'],default='all')
    parser.add_argument('--audit-mode',choices=['all','failures','none'],default='all')
    parser.add_argument('--trace-mode',choices=['all','failures','none'],default='all')
    parser.add_argument('--save-event-tensors',action='store_true')
    parser.add_argument('--max-wall-hours',type=float,default=0.,help='本次调用预算；仅在完整回合边界暂停，0不限时')
    parser.add_argument('--min-free-gib',type=float,default=2.,help='新回合启动前的剩余磁盘下限')
    parser.add_argument('--stop-file',type=Path,help='出现此文件后在回合边界暂停，删除后可同计划续跑')
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.max_wall_hours,args.min_free_gib)):
        raise ValueError('时间预算及磁盘下限须为非负有限数')
    methods=args.methods or [f'H{args.action_steps}']
    uses_corrector=any(m in {'monitor','lvm','ogg'} for m in methods)
    if len(methods)!=len(set(methods)):raise ValueError('方法不可重复')
    if uses_corrector:
        if not args.corrector_checkpoint or not args.corrector_checkpoint.is_file() or not args.source_root:
            raise ValueError('Corrector模式需要已保存的checkpoint和source-root')
        from corrector_online_runtime import decisions, guided_velocity
        decisions(args.control_implementation)
    if not 1 <= args.action_steps <= 50:
        raise ValueError("action-steps须为1..50")
    if args.fps <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        raise ValueError("FPS与最大控制步数必须为正")
    import smolvla_training_workbench as wb
    import smolvla_libero_tools as sim
    args.workdir = args.workdir.absolute()
    sim_args = argparse.Namespace(workdir=args.workdir, sim_fps=args.fps, max_steps=args.max_steps or 280)
    sim.configure(sim_args, wb)  # 固定源码/配置/导入路径检查；此时还没有创建环境或加载VLA
    from libero.libero import benchmark
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, get_task_init_states
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if len(args.task_ids) != len(set(args.task_ids)) or len(args.initial_state_ids) != len(set(args.initial_state_ids)):
        raise ValueError("task/init_state编号不能重复")
    horizon = args.max_steps or TASK_SUITE_MAX_STEPS[args.suite]
    import torch
    from importlib.metadata import version
    runtime = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               **{name: version(name) for name in ["lerobot", "transformers", "mujoco", "robosuite"]}}
    conditions = {}
    for task_id in args.task_ids:
        if not 0 <= task_id < len(suite.tasks):
            raise ValueError(f"task_id越界：{task_id}")
        states = get_task_init_states(suite, task_id, is_libero_plus=False)
        for init_id in args.initial_state_ids:
            if not 0 <= init_id < len(states):
                raise ValueError(f"task{task_id}没有init_state_id={init_id}")
            name = f"{args.suite}/task_{task_id}/init_{init_id}"
            conditions[name] = {"task_id": task_id, "initial_state_id": init_id,
                                "language": suite.tasks[task_id].language}
    method = f"H{args.action_steps}"
    method_configs={m:{'n_action_steps':int(m[1:]) if m.startswith('H') else args.action_steps,
                       'uses_corrector':m in {'monitor','lvm','ogg'},'video_policy':args.video_policy,
                       'corrector_sha256':wb.digest(args.corrector_checkpoint) if m in {'monitor','lvm','ogg'} else None}
                    for m in methods}
    plan = make_plan(list(conditions), args.seeds,
                     method_configs,
                     {"scope": "frozen SmolVLA / original LIBERO; Corrector adaptation when selected, not a paper table reproduction",
                      "policy_id": POLICY_ID, "policy_revision": POLICY_SHA, "lerobot_revision": wb.LEROBOT_SHA,
                      "libero_revision": sim.LIBERO_SHA, "conditions": conditions,
                      "runtime": runtime,
                      "fps": args.fps, "horizon": horizon, "control_mode": "relative",
                      'corrector_settings':{'stride':args.monitor_stride,'eta':args.guidance_eta,
                          'guidance_every':args.guidance_every,'cooldown':args.monitor_cooldown,
                          'force_step':args.debug_force_step,'control_implementation':args.control_implementation},
                      'retention':{'video':args.video_policy,'audit':args.audit_mode,'trace':args.trace_mode},
                      'debug_forced_event':args.debug_force_step>=0})
    plan['protocol']['evaluation_entry_sha256']=wb.digest(Path(__file__))
    if uses_corrector:
        plan['protocol']['runtime_source_sha256']=wb.digest(Path(__file__).parent/'corrector_online_runtime.py')
        plan['protocol']['learner_core_sha256']=wb.digest(Path(inspect.getfile(guided_velocity)))
    holder = {}

    def load_policy():
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        if not torch.cuda.is_available():
            raise RuntimeError("完整策略基线要求CUDA；CPU只做入口/场景与汇总检查")
        snapshot = snapshot_download(POLICY_ID, revision=POLICY_SHA, token=False)
        cfg = SmolVLAConfig.from_pretrained(snapshot)
        cfg.device, cfg.n_action_steps, cfg.pretrained_path = "cuda", args.action_steps, Path(snapshot)
        cfg.load_vlm_weights = False  # 完整policy权重随后加载，不重复下载原始VLM权重
        policy = SmolVLAPolicy.from_pretrained(snapshot, config=cfg).to("cuda").eval()
        policy.requires_grad_(False)
        policy.reset()
        if policy._queues["action"].maxlen != args.action_steps:
            raise RuntimeError("实际动作队列容量与配置不一致")
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=snapshot,
                                            preprocessor_overrides={"device_processor": {"device": "cuda"}})
        holder.update(policy=policy, pre=pre, post=post)
        if uses_corrector:
            from corrector_online_runtime import load_predictor
            predictor,protocol=load_predictor(args.corrector_checkpoint,args.source_root,'cuda')
            holder.update(predictor=predictor,predictor_protocol=protocol)

    def run_case(case, directory):
        import torch
        from unittest.mock import patch
        from lerobot.envs.configs import LiberoEnv
        from lerobot.scripts.lerobot_eval import eval_policy
        from lerobot.utils.random_utils import set_seed
        if not holder:
            load_policy()  # 全部回合已完成时，不重新加载模型
        torch.cuda.reset_peak_memory_stats()
        policy, pre, post = holder["policy"], holder["pre"], holder["post"]
        policy.config.n_action_steps=case['config']['n_action_steps'];policy.reset()
        if policy._queues['action'].maxlen!=case['config']['n_action_steps']:raise RuntimeError('实际队列容量错误')
        chosen = conditions[case["condition"]]
        camera_keys = list(policy.config.image_features)
        if len(camera_keys) != 2:
            raise RuntimeError("本协议要求两路相机")
        camera_names = [k.removeprefix("observation.images.") for k in camera_keys]
        cfg = LiberoEnv(task=args.suite, task_ids=[chosen["task_id"]], fps=args.fps,
                        episode_length=horizon, observation_height=256, observation_width=256,
                        control_mode="relative", is_libero_plus=False,
                        camera_name_mapping={"agentview_image": camera_names[0], "robot0_eye_in_hand_image": camera_names[1]})
        env = sim.make_vector(cfg)
        env_pre, env_post = cfg.get_env_processors()
        steps, chunk_times = 0, []
        runtime=None
        if case['config']['uses_corrector']:
            from corrector_online_runtime import OnlineCorrector
            runtime=OnlineCorrector(policy,holder['predictor'],holder['predictor_protocol'],case['method'],
                args.source_root,directory,fps=args.fps,stride=args.monitor_stride,eta=args.guidance_eta,
                guidance_every=args.guidance_every,implementation=args.control_implementation,
                trace=args.trace_mode!='none',event_tensors=args.save_event_tensors,
                force_step=args.debug_force_step,cooldown=args.monitor_cooldown)
        try:
            select_initial_state(env, chosen["initial_state_id"])
            set_seed(case["seed"])
            def observe(batch):
                nonlocal steps
                if steps == 0 and args.audit_mode!='none':
                    state = torch.as_tensor(env.envs[0].unwrapped._env.get_sim_state()).clone()
                    torch.save(state, directory / "initial_sim_state.pt")
                    wb.save_inspection_images(batch, directory / "input_images")
                value = pre(batch)
                if steps == 0 and args.audit_mode!='none':
                    torch.save({k: v.detach().cpu() for k, v in value.items() if isinstance(v, torch.Tensor)},
                               directory / "first_policy_input.pt")
                steps += 1
                return value
            original = policy._get_action_chunk
            def timed(*a, **kw):
                torch.cuda.synchronize(); start = time.perf_counter()
                out = original(*a, **kw)
                torch.cuda.synchronize(); chunk_times.append(time.perf_counter() - start)
                return out
            original_noise = policy.model.sample_noise
            def record_noise(*a, **kw):
                noise = original_noise(*a, **kw)
                if args.audit_mode!='none' and not (directory / "first_noise.pt").exists():
                    torch.save(noise.detach().cpu(), directory / "first_noise.pt")
                return noise
            with ExitStack() as stack:
                stack.enter_context(patch.object(policy,'_get_action_chunk',timed))
                stack.enter_context(patch.object(policy.model,'sample_noise',record_noise))
                if runtime is not None:stack.enter_context(runtime.installed())
                info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                                   preprocessor=observe, postprocessor=post, n_episodes=1,
                                   max_episodes_rendered=0 if args.video_policy=='none' else 1,
                                   videos_dir=directory / "videos", start_seed=case["seed"])
            videos = [sim.verify_video(p, args.fps) for p in info.get("video_paths", [])]
            if args.video_policy!='none' and (len(videos) != 1 or videos[0]["frames"] != steps):
                raise RuntimeError("视频帧数与控制步数不一致")
            wb.write_json(directory / "eval_info.json", info)
            wb.write_json(directory / "actual_protocol.json", {**chosen, "case": case,
                          "camera_keys": camera_keys, "chunk_size": policy.config.chunk_size,
                          "n_action_steps": policy.config.n_action_steps, "num_steps": policy.config.num_steps,
                          "video": videos[0] if videos else None, "chunk_wall_s": chunk_times,
                          'monitor_source_sha256':runtime.monitor_sha if runtime else None,
                          'monitor_horizon_steps':runtime.k if runtime else None,
                          'predictor_protocol':holder.get('predictor_protocol') if runtime else None})
            success=bool(info['per_episode'][0]['success'])
            retained=list(info.get('video_paths',[]))
            if success and args.video_policy=='failures':
                for v in retained:Path(v).unlink()
                retained=[]
            if success and args.audit_mode=='failures':
                for name in ['first_policy_input.pt','initial_sim_state.pt','first_noise.pt']:(directory/name).unlink(missing_ok=True)
                if (directory/'input_images').exists():shutil.rmtree(directory/'input_images')
            if success and args.trace_mode=='failures':(directory/'monitor_steps.jsonl').unlink(missing_ok=True)
            wb.write_json(directory/'retention.json',{'success':success,'video_policy':args.video_policy,
                'retained_video_paths':retained,'audit_mode':args.audit_mode,'trace_mode':args.trace_mode})
            return {"success": success, "env_steps": steps,
                    "chunk_calls": len(chunk_times), "policy_s": sum(chunk_times),
                    'monitor_s':runtime.monitor_s if runtime else 0.,
                    "event_replans": runtime.events if runtime else 0,
                    'guidance_calls':runtime.guidance_calls if runtime else 0,
                    'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    "video_paths": retained, "initial_state_id": chosen["initial_state_id"]}
        finally:
            env.close()
    print(f"冻结计划：{len(plan['cases'])}回合；任务{args.task_ids}，真实初始状态{args.initial_state_ids}，seed{args.seeds}", flush=True)
    rows = run_plan(args.output, plan, {m: run_case for m in methods},
                    max_seconds=args.max_wall_hours*3600,min_free_gib=args.min_free_gib,stop_file=args.stop_file)
    print(rows)


if __name__ == "__main__":
    main()
