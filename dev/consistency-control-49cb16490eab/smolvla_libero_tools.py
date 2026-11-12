"""保存checkpoint的LIBERO闭环评测；复用官方rollout/处理器，不实现另一个控制循环。"""
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

LIBERO_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
TASK_LANGUAGE = "pick up the alphabet soup and place it in the basket"
SIM_PINS = ["robosuite==1.4.1", "bddl==1.0.1", "easydict==1.13", "mujoco==3.7.0",
            "future==1.0.0", "Wand==0.6.13", "scikit-image==0.25.2", "gym==0.26.2",
            "matplotlib==3.10.8", "numba==0.61.2", "hydra-core>=1.2,<1.4", "h5py>=3.11,<4", "cloudpickle>=2"]


def sim_environment(workdir, wb):
    env = wb.runtime_env(workdir)
    env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               LIBERO_CONFIG_PATH=str(workdir / "libero-config"),
               MPLCONFIGDIR=str(workdir / "matplotlib-cache"), NUMBA_CACHE_DIR=str(workdir / "numba-cache"))
    return env


def find_task(benchmark):
    """按原始任务语言定位，拒绝view后缀；数据集task_index不是suite内部task_id。"""
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    matches = [(i, task) for i, task in enumerate(suite.tasks)
               if " ".join(task.language.lower().split()) == TASK_LANGUAGE]
    if len(matches) != 1:
        raise RuntimeError("未唯一找到原始alphabet soup任务；不能用带view后缀的Plus任务代替。")
    return matches[0]


def register_source_root(site_dir, source):
    """LIBERO外层是namespace目录，旧setup.py的find_packages可能漏掉它。"""
    source = Path(source).resolve()
    if not (source / "libero/libero/__init__.py").is_file():
        raise FileNotFoundError(f"LIBERO源码布局不完整：{source}")
    target = Path(site_dir) / "libero_source_root.pth"
    target.write_text(str(source) + "\n")
    return target


def repair_entry(args, log, wb):
    """已有依赖安装成功时，只修复导入路径并复查渲染，不安装、不加载VLA。"""
    work = args.workdir
    python = work / "sim-venv/bin/python"
    source = work / "libero-original"
    if not python.is_file() or not (work / "libero-config/config.yaml").is_file():
        raise RuntimeError("没有已安装的仿真环境，请用prepare-sim完成首次准备。")
    site = Path(subprocess.check_output([str(python), "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    env = sim_environment(work, wb)
    register_source_root(site, source)
    log.say("已在独立sim-venv注册LIBERO源码根目录；不修改第三方源码或训练venv。")
    log.run([python, "-c", "import libero.libero as m; print('LIBERO实际导入文件:', m.__file__)"], env=env)
    version_check = ("from packaging.requirements import Requirement; from importlib.metadata import version; "
                     f"items=[Requirement(s) for s in {SIM_PINS!r}]; "
                     "bad=[str(r) for r in items if version(r.name) not in r.specifier]; "
                     "assert not bad, '依赖版本不符合本次协议，请运行prepare-sim: '+str(bad); "
                     "print('固定仿真依赖版本核对通过')")
    log.run([python, "-c", version_check], env=env)
    log.run([python, "-m", "pip", "check"], env=env)
    wb.write_json(work / "simulation.json", {"ready": False, "status": "repairing", "libero_revision": LIBERO_SHA})
    log.run([python, Path(wb.__file__).resolve(), "check-sim", "--workdir", work,
             "--log-root", log.directory / "render_probe", "--sim-fps", str(args.sim_fps)], env=env)
    wb.write_json(work / "simulation.json", {"ready": True, "libero_revision": LIBERO_SHA,
                  "python": str(python), "source": str(source), "requirements": SIM_PINS,
                  "render_probe": str(log.directory / "render_probe"), "sim_fps": args.sim_fps,
                  "repair": "namespace source-root registration; no package installation"})
    log.say("路径修复与真实渲染复查完成；没有重新训练或加载VLA。")


def prepare_entry(args, log, wb):
    """独立sim-venv借用训练依赖；不修改已跑通的训练venv或宿主Python。"""
    work = args.workdir
    training_python = work / "venv/bin/python"
    if not training_python.is_file() or not (work / "environment.json").is_file():
        raise RuntimeError("先运行训练环境prepare；仿真环境会借用其已验证的模型依赖。")
    training = json.loads((work / "environment.json").read_text())
    if training.get("profile") != "gpu" or training.get("lerobot_revision") != wb.LEROBOT_SHA:
        raise RuntimeError("需要已验证的真实SmolVLA训练环境，不能使用CPU小模型练习目录。")
    env = sim_environment(work, wb)
    env.pop("HF_HUB_OFFLINE", None)
    ready_path = work / "simulation.json"
    previous = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
    if (previous.get("ready") and previous.get("requirements") == SIM_PINS
            and previous.get("libero_revision") == LIBERO_SHA and (work / "sim-venv/bin/python").is_file()):
        site = Path(subprocess.check_output([str(work / "sim-venv/bin/python"), "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        register_source_root(site, work / "libero-original")
        previous["ready"] = False
        wb.write_json(ready_path, previous)
        log.run([work / "sim-venv/bin/python", "-m", "pip", "check"], env=env)
        log.run([work / "sim-venv/bin/python", Path(wb.__file__).resolve(), "check-sim", "--workdir", work,
                 "--log-root", log.directory / "render_probe", "--sim-fps", str(args.sim_fps)], env=env)
        previous.update(ready=True, render_probe=str(log.directory / "render_probe"), sim_fps=args.sim_fps)
        wb.write_json(ready_path, previous)
        log.say("复用现有仿真环境；真实渲染/录像复查通过，没有重装依赖。")
        return
    if shutil.disk_usage(work).free < 3 * 2**30:
        raise RuntimeError("首次仿真准备至少预留3GiB磁盘，保留已有文件，不清缓存。")
    wb.write_json(ready_path, {"ready": False, "status": "preparing", "libero_revision": LIBERO_SHA})
    sim_python = wb.ensure_environment(work / "sim-venv", log, env)
    train_site = subprocess.check_output([str(training_python), "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
    sim_site = Path(subprocess.check_output([str(sim_python), "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    # addsitedir也处理训练环境的editable .pth；sim自己安装的包排在前面。
    (sim_site / "training_dependencies.pth").write_text(f"import site; site.addsitedir({train_site!r})\n")
    if sys.platform != "linux":
        raise RuntimeError("此EGL准备入口面向Colab/AutoDL的Linux运行时。")
    if os.geteuid() == 0:
        log.run(["apt-get", "update", "-qq"], env=env)
        log.run(["apt-get", "install", "-y", "libegl1", "libgl1", "libgles2", "libexpat1",
                 "libfontconfig1", "libmagickwand-dev"], env=env)
    else:
        log.say("非root，跳过apt；系统须已安装EGL/OpenGL/ImageMagick运行库，后面的真实渲染检查会验证。")
    source = work / "libero-original"
    if not (source / ".git").is_dir():
        if source.exists() and any(source.iterdir()):
            raise RuntimeError(f"保留已有非Git目录：{source}")
        source.mkdir(exist_ok=True)
        log.run(["git", "init", str(source)], env=env)
        log.run(["git", "-C", str(source), "remote", "add", "origin", LIBERO_REPO], env=env)
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
        # 首次fetch断网留下的空仓库可重试；不清空目录或重置已存在的提交。
        log.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", LIBERO_SHA], env=env)
        log.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], env=env)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != LIBERO_SHA:
        raise RuntimeError("已有LIBERO版本不同，保留现场，不自动切换。")
    for required in ["libero/libero/assets/scenes", "libero/libero/assets/textures",
                     "libero/libero/init_files/libero_object", "libero/libero/bddl_files/libero_object"]:
        if not (source / required).is_dir():
            raise RuntimeError(f"LIBERO源码/资产不完整，保留现场：{source / required}")
    constraints = work / "simulation-constraints.txt"
    torch_pin = subprocess.check_output([str(training_python), "-c",
                      "from importlib.metadata import version; print('torch=='+version('torch'))"], text=True).strip()
    constraints.write_text("\n".join([torch_pin, *wb.PINS]) + "\n")
    # 不使用lerobot[libero]，不引入hf-libero/robomimic的EGL probe构建分支。
    log.run([sim_python, "-m", "pip", "install", "--constraint", constraints,
             "--report", log.directory / "pip-simulation.json", *SIM_PINS], env=env)
    log.run([sim_python, "-m", "pip", "install", "--no-deps", "-e", source], env=env)
    register_source_root(sim_site, source)
    root = source / "libero/libero"
    # import前写配置，避免子进程input()导致EOFError。
    config = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
              "init_states": str(root / "init_files"), "assets": str(root / "assets"),
              "datasets": str(work / "libero-unused-datasets")}
    Path(config["datasets"]).mkdir(exist_ok=True)
    config_file = work / "libero-config/config.yaml"
    config_file.parent.mkdir(exist_ok=True)
    # JSON是有效YAML，避免在宿主另装yaml。
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    macros = ("import importlib.util, pathlib, shutil; "
              "p=pathlib.Path(importlib.util.find_spec('robosuite').origin).parent; "
              "q=p/'macros_private.py'; "
              f"shutil.copyfile(p/'macros.py',q) if not q.exists() and p.is_relative_to(pathlib.Path({str(sim_site)!r})) else None")
    log.run([sim_python, "-c", macros], env=env)
    log.run([sim_python, "-c", "import libero.libero as m; print('LIBERO实际导入文件:', m.__file__)"], env=env)
    log.run([sim_python, "-m", "pip", "check"], env=env)
    frozen = subprocess.check_output([str(sim_python), "-m", "pip", "freeze"], text=True, env=env)
    (log.directory / "simulation-pip-freeze.txt").write_text(wb.redact(frozen))
    # 不加载VLA；实际reset、图像预处理、MP4编码由独立进程检查。
    log.run([sim_python, Path(wb.__file__).resolve(), "check-sim", "--workdir", work,
             "--log-root", log.directory / "render_probe", "--sim-fps", str(args.sim_fps)], env=env)
    wb.write_json(work / "simulation.json", {"ready": True, "libero_revision": LIBERO_SHA,
                  "python": str(sim_python), "source": str(source), "config": config,
                  "requirements": SIM_PINS,
                  "render_probe": str(log.directory / "render_probe"), "sim_fps": args.sim_fps})
    log.say("独立LIBERO环境及真实渲染检查完成；未加载GPU策略、未训练。")


def configure(args, wb):
    os.environ.update(sim_environment(args.workdir, wb))
    source = args.workdir / "libero-original"
    if not (args.workdir / "libero-config/config.yaml").is_file():
        raise RuntimeError("先执行prepare-sim，配置缺失时不导入LIBERO，避免交互提示。")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != LIBERO_SHA:
        raise RuntimeError("实际LIBERO提交与固定协议不同。")
    import libero.libero as libero
    if not Path(libero.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("导入了其他LIBERO，请用sim-venv/bin/python执行。")
    from libero.libero import benchmark
    task_id, task = find_task(benchmark)
    from lerobot.envs.configs import LiberoEnv
    cfg = LiberoEnv(task="libero_object", task_ids=[task_id], fps=args.sim_fps,
                    episode_length=args.max_steps, observation_height=256, observation_width=256,
                    control_mode="relative", is_libero_plus=False,
                    camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"})
    return cfg, task


def make_vector(cfg):
    from lerobot.envs.factory import make_env
    env = make_env(cfg, n_envs=1, use_async_envs=False)[cfg.task][cfg.task_ids[0]]
    return configure_vector(env, cfg.fps, cfg.episode_length)


def configure_vector(env, fps, horizon):
    """实例级录像FPS；TimeLimit让超时也产生done，避免官方聚合器argmax误判。"""
    import gymnasium as gym
    env.metadata = {**env.metadata, "render_fps": fps}
    for index, subenv in enumerate(env.envs):
        subenv.metadata = {**subenv.metadata, "render_fps": fps}
        env.envs[index] = gym.wrappers.TimeLimit(subenv, max_episode_steps=horizon)
    return env


def verify_video(path, fps):
    import av
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"录像未生成：{path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        count = sum(1 for _ in container.decode(video=0))
    if count < 1 or abs(rate - fps) > 1e-6:
        raise RuntimeError(f"录像帧数/播放FPS异常：{count}/{rate}")
    return {"path": str(path), "frames": count, "playback_fps": rate, "duration_s": count / rate}


def check_entry(args, log, wb):
    import numpy as np
    import torch
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.random_utils import set_seed
    cfg, task = configure(args, wb)
    env = make_vector(cfg)
    try:
        set_seed(args.seed)
        raw, _ = env.reset(seed=[args.seed])
        batch = preprocess_observation(raw)
        batch["task"] = [task.language]
        env_pre, _ = cfg.get_env_processors()
        batch = env_pre(batch)
        if batch["observation.state"].shape != (1, 8):
            raise RuntimeError("环境状态不是[eef_pos3,axis_angle3,gripper_qpos2]。")
        for key in ["observation.images.image", "observation.images.image2"]:
            if batch[key].shape != (1, 3, 256, 256) or not torch.isfinite(batch[key]).all():
                raise RuntimeError(f"相机输入错误：{key}")
        wb.save_inspection_images(batch, log.directory)
        frame = env.call("render")[0]
        initial = torch.as_tensor(env.envs[0].unwrapped._env.get_sim_state()).clone()
        from lerobot.envs.libero import get_libero_dummy_action
        _, _, _, _, step_info = env.step(np.asarray([get_libero_dummy_action()], dtype=np.float64))
        if "is_success" not in step_info:
            raise RuntimeError("环境step没有提供is_success，不能统计任务成功率。")
        video = log.directory / "render_probe.mp4"
        write_video(video, [frame, env.call("render")[0]], args.sim_fps)
        probe = verify_video(video, args.sim_fps)
    finally:
        env.close()
    repeated_env = make_vector(cfg)
    try:
        set_seed(args.seed)
        repeated_env.reset(seed=[args.seed])
        repeated = torch.as_tensor(repeated_env.envs[0].unwrapped._env.get_sim_state())
        torch.testing.assert_close(repeated, initial, rtol=1e-5, atol=1e-6)
    finally:
        repeated_env.close()
    wb.write_json(log.directory / "render_check.json", {"ok": True, "task": task.language,
                  "state_shape": list(batch["observation.state"].shape), "video": probe,
                  "step_success_available": True, "paired_initial_state_checked": True})
    log.say(f"真实场景reset/step、初始状态配对、两路图像/状态、MP4编码通过：{task.language}")


def evaluate_entry(args, log, wb):
    """核心链：环境观测→环境处理→保存的processor→select_action→反归一化→env.step。"""
    import torch
    from unittest.mock import patch
    from transformers import AutoModelForImageTextToText
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from peft import PeftConfig

    sim = json.loads((args.workdir / "simulation.json").read_text())
    if not sim.get("ready") or sim["libero_revision"] != LIBERO_SHA:
        raise RuntimeError("仿真准备/真实渲染检查尚未完成。")
    if not Path(inspect.getfile(make_policy)).resolve().is_relative_to((args.workdir / "lerobot").resolve()):
        raise RuntimeError("LeRobot源码导入路径与准备环境不同。")
    env_cfg, task = configure(args, wb)
    manifest = json.loads((args.workdir / "environment.json").read_text())
    candidates = [args.workdir / "train_run/checkpoints/000005/pretrained_model"]
    if args.archive_dir:
        candidates.append(args.archive_dir / "checkpoints/000005/pretrained_model")
    checkpoint = args.checkpoint or next((p for p in candidates if p.is_dir()), None)
    if checkpoint is None:
        raise FileNotFoundError("找不到官方五步checkpoint，请用--checkpoint指定pretrained_model。")
    adapter = PeftConfig.from_pretrained(checkpoint)
    evidence = json.loads((checkpoint.parent / "training_evidence.json").read_text())
    if (adapter.base_model_name_or_path != wb.BASE_ID or adapter.revision != wb.BASE_SHA
            or adapter.modules_to_save or adapter.bias != "none" or not evidence.get("trainable")
            or any("lora_" not in n for n in evidence["trainable"])):
        raise RuntimeError("此配对评测要求固定base、仅训练LoRA，不符合条件。")
    ds_meta = LeRobotDatasetMetadata(wb.DATA_ID, root=manifest["dataset_root"], revision=wb.DATA_SHA)
    languages = [str(name) for name, row in ds_meta.tasks.iterrows() if int(row["task_index"]) == 24]
    if languages != [TASK_LANGUAGE]:
        raise RuntimeError("数据任务24与仿真目标任务不一致。")
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path, cfg.device = checkpoint, "cuda"
    if set(cfg.image_features) != {"observation.images.image", "observation.images.image2"}:
        raise RuntimeError("本协议要求保存策略使用image/image2两路相机。")
    if not cfg.use_peft or cfg.input_features["observation.state"].shape != (8,) or cfg.output_features["action"].shape != (7,):
        raise RuntimeError("保存策略的状态/动作维度或PEFT配置不符。")
    if Path(args.eval_run).name != args.eval_run or args.eval_run in ("", ".", ".."):
        raise ValueError("eval-run必须是单个目录名。")
    output = (args.archive_dir or args.workdir) / "libero_evaluations" / args.eval_run
    if output.exists():
        raise FileExistsError(f"保留已有仿真结果：{output}，请更换eval-run。")
    if not torch.cuda.is_available():
        raise RuntimeError("完整VLA闭环请在T4运行；本地只做指标/接入检查。")
    # CPU和EGL检查应在prepare-sim中完成；这里复用成功的G2重载路径。
    original = AutoModelForImageTextToText.from_pretrained
    def load_vlm(*a, **kw):
        if "torch_dtype" in kw:
            kw.setdefault("dtype", kw.pop("torch_dtype"))
        kw.setdefault("token", False)
        return original(*a, **kw)
    output.mkdir(parents=True)
    protocol = {"complete": False, "scope": "single-task closed-loop protocol smoke; not generalization benchmark",
                "task": task.language, "suite": env_cfg.task, "task_id": env_cfg.task_ids[0],
                "checkpoint": str(checkpoint), "adapter_sha256": wb.digest(checkpoint / "adapter_model.safetensors"),
                "libero_revision": LIBERO_SHA, "lerobot_revision": wb.LEROBOT_SHA, "base_revision": wb.BASE_SHA,
                "dataset_revision": wb.DATA_SHA, "training_data_fps": ds_meta.fps,
                "control_fps": args.sim_fps, "playback_fps": args.sim_fps, "control_mode": "relative",
                "frequency_note": "data FPS does not prove simulator control rate; this explicit protocol needs behavior verification",
                "max_steps": args.max_steps, "episodes_per_condition": args.episodes, "seed": args.seed,
                "conditions": args.condition, "success_rate_units": "fraction in [0,1]",
                "n_action_steps": cfg.n_action_steps, "flow_steps": cfg.num_steps, "results": []}
    wb.write_json(output / "eval_info.json", protocol)
    with patch.object(AutoModelForImageTextToText, "from_pretrained", staticmethod(load_vlm)):
        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    policy.requires_grad_(False)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint,
                  preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env_pre, env_post = env_cfg.get_env_processors()
    conditions = ["base", "trained"] if args.condition == "both" else [args.condition]
    initial_states = {}
    for condition in conditions:
        env = make_vector(env_cfg)  # 新建环境使两个条件从同一初始状态序列起步
        try:
            for episode in range(args.episodes):
                set_seed(args.seed + episode)
                destination = output / condition / f"episode_{episode:03d}"
                destination.mkdir(parents=True)
                first = []
                control_steps = 0
                def observed_preprocessor(batch):
                    nonlocal control_steps
                    control_steps += 1
                    if not first:
                        # 环境processor已翻转图像/拼接8维状态；这里不能再翻转一次。
                        state = torch.as_tensor(env.envs[0].unwrapped._env.get_sim_state()).clone()
                        first.append(state)
                        key = episode
                        if condition == "trained" and key in initial_states:
                            torch.testing.assert_close(state, initial_states[key], rtol=1e-5, atol=1e-6)
                        else:
                            initial_states[key] = state
                        torch.save(state, destination / "initial_sim_state.pt")
                        wb.save_inspection_images(batch, destination / "input_images")
                    processed = pre(batch)
                    if not (destination / "first_policy_input.pt").exists():
                        torch.save({k: v.detach().cpu() for k, v in processed.items()
                                    if isinstance(v, torch.Tensor)}, destination / "first_policy_input.pt")
                    return processed
                with policy.disable_adapter() if condition == "base" else nullcontext():
                    info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre,
                         env_postprocessor=env_post, preprocessor=observed_preprocessor, postprocessor=post,
                         n_episodes=1, max_episodes_rendered=1, videos_dir=destination / "videos",
                         start_seed=args.seed + episode)
                videos = [verify_video(p, args.sim_fps) for p in info.get("video_paths", [])]
                if len(videos) != 1:
                    raise RuntimeError("本回合没有生成完整视频，不记作已完成评测。")
                if videos[0]["frames"] != control_steps:
                    raise RuntimeError("录像帧数与实际控制步数不一致，保留现场，不记作完整回合。")
                from PIL import Image
                Image.fromarray(env.call("render")[0]).save(destination / "final_frame.png")
                raw_env = env.envs[0].unwrapped
                initial_index = (raw_env.init_state_id - 1) % len(raw_env._init_states)
                row = {"condition": condition, "episode": episode, "initial_state_index": initial_index,
                       "control_steps": control_steps,
                       "seed": args.seed + episode, "success": bool(info["per_episode"][0]["success"]),
                       "metrics": info["aggregated"], "videos": videos}
                wb.write_json(destination / "episode_result.json", row)
                protocol["results"].append(row)
                wb.write_json(output / "eval_info.json", protocol)  # 每回合立即归档；断连仍有部分证据
                log.say(row)
        finally:
            env.close()
    protocol["success_rates"] = {condition: sum(r["success"] for r in protocol["results"] if r["condition"] == condition)
                                / args.episodes for condition in conditions}
    protocol["success_counts"] = {condition: sum(r["success"] for r in protocol["results"] if r["condition"] == condition)
                                 for condition in conditions}
    protocol["complete"] = True
    wb.write_json(output / "eval_info.json", protocol)
    log.say(f"LIBERO执行和录像完成：{output}；success_rates={protocol['success_rates']}，单任务少量回合不能代表泛化。")
