"""训练准备、CPU LoRA机制实验、官方SmolVLA训练；各阶段显式启动。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

LEROBOT_SHA = "7e241bd630a3719a56157a497ce5d08f244784f1"
BASE_ID = "lerobot/smolvla_base"
BASE_SHA = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
DATA_ID = "HuggingFaceVLA/libero"
DATA_SHA = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
TRAIN_EPISODES = (813, 818)
HOLDOUT_EPISODES = (825,)
# 此固定revision的元信息data/file_index与实际分片不符；以下文件已按物理episode列核对。
DATA_FILES = tuple(f"data/chunk-000/file-{i:03d}.parquet" for i in (213, 214, 215))
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PINS = ["peft==0.21.0", "accelerate==1.14.0", "transformers==5.5.4", "tokenizers==0.22.2", "numpy==2.1.3"]
MANAGED_TORCH_PINS = ["torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"]
MANAGED_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


class BatchInspectionComplete(Exception):
    """主动结束batch检查，不代表训练失败。"""


class PracticeStepComplete(Exception):
    """用户的单batch更新已运行，不作为官方训练器的完整checkpoint。"""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def redact(text):
    text = re.sub(r"(?:hf_|ghp_)[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", "<TOKEN_REDACTED>", text)
    return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1<REDACTED>@", text)


class Log:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "run.log"

    def say(self, message):
        line = redact(str(message))
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def run(self, command, *, env=None):
        self.say("运行：" + json.dumps([str(x) for x in command], ensure_ascii=False))
        with subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", env=env) as process:
            self.say(f"子进程PID={process.pid}；正在执行上方命令。")
            for line in process.stdout:
                self.say(line.rstrip())
            code = process.wait()
        if code:
            raise RuntimeError(f"子进程退出码{code}；完整日志：{self.path}")


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, message):
        message = redact(message)
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return False


def resources():
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ["MemTotal", "MemAvailable", "SwapFree"]:
                memory[key + "_GiB"] = round(int(value.split()[0]) / 2**20, 2)
    return {"python": sys.version, "executable": sys.executable,
            "cpu_threads": os.cpu_count(), "memory": memory}


def runtime_env(workdir):
    env = os.environ.copy()
    env.update({"HF_HOME": str(workdir / "hf"), "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1", "PIP_CACHE_DIR": str(workdir / "pip-cache"),
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                # Notebook的inline后端模块不一定装在隔离venv；CLI只需无显示器绘图。
                "MPLBACKEND": "Agg"})
    # 不继承Notebook中临时拼接的另一份LeRobot源码。
    env.pop("PYTHONPATH", None)
    return env


def check_persistent_path(path):
    if path and str(Path(path).absolute()).startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise RuntimeError("没有检测到/content/drive挂载；拒绝把普通本地目录标作云盘归档。")


def ensure_environment(environment, log, env):
    """不依赖宿主ensurepip；重试时修复已有Python但没有pip的半成品。"""
    environment = Path(environment).absolute()
    python = environment / "bin/python"
    if not python.exists():
        log.run([sys.executable, "-m", "venv", "--without-pip", environment], env=env)
    probe = subprocess.run(
        [str(python), "-c", "import sys,json;print(json.dumps([sys.prefix,list(sys.version_info[:2])]))"],
        text=True, capture_output=True, env=env, timeout=30,
    )
    if probe.returncode:
        log.say(probe.stdout + probe.stderr)
        raise RuntimeError(f"已有环境的Python无法运行，保留现场：{environment}")
    prefix, py_version = json.loads(probe.stdout)
    if Path(prefix).resolve() != environment.resolve() or py_version != list(sys.version_info[:2]):
        raise RuntimeError(f"已有Python属于其他环境或版本，保留现场：{environment}")
    pip_probe = subprocess.run([str(python), "-m", "pip", "--version"],
                               text=True, capture_output=True, env=env, timeout=30)
    if pip_probe.returncode:
        log.say("环境的Python已存在但pip未就绪，开始补齐；不删除已有文件。")
        host = subprocess.run([sys.executable, "-c", "import pip;print(pip.__version__)"],
                              text=True, capture_output=True, env=env, timeout=30)
        match = re.match(r"(\d+)\.(\d+)", host.stdout.strip())
        if host.returncode == 0 and match and tuple(map(int, match.groups())) >= (22, 3):
            # --python指定安装目标，不向Colab全局环境安装/升级依赖。
            log.run([sys.executable, "-m", "pip", "--python", python,
                     "install", "pip==25.0.1"], env=env)
        else:
            # 本机某些uv环境没有pip，但底层Python的ensurepip可用。
            log.say("宿主没有可用的pip>=22.3；尝试环境内ensurepip，保留完整诊断输出。")
            log.run([python, "-m", "ensurepip", "--upgrade", "--default-pip"], env=env)
    log.run([python, "-m", "pip", "--version"], env=env)
    return python


def torch_version_supported(version):
    match = re.match(r"^(\d+)\.(\d+)", version)
    return bool(match and (2, 7) <= tuple(map(int, match.groups())) < (2, 12))


def prepare_with_managed_torch(args, log):
    """宿主Torch不兼容时，用独立CUDA wheel环境启动prepare；不改Colab宿主。

    安装和导入发生在新解释器，不会在当前已import的Torch进程内降级。
    正式venv只链接此环境里的native包；不再借用不兼容的宿主native包。
    """
    workdir = args.workdir
    env = runtime_env(workdir)
    python = ensure_environment(workdir / "torch-runtime-2.11-cu128", log, env)
    probe_code = (
        "import json,torch,torchvision,torchaudio;"
        "assert torch.__version__.split('+')[0]=='2.11.0';"
        "assert torchvision.__version__.split('+')[0]=='0.26.0';"
        "assert torchaudio.__version__.split('+')[0]=='2.11.0';"
        "assert torch.cuda.is_available(), '独立环境CUDA不可用';"
        "x=torch.ones(2,device='cuda');assert x.sum().item()==2;"
        "torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device='cuda'),torch.tensor([1.],device='cuda'),.5);"
        "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
        "'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))"
    )
    probe = subprocess.run([str(python), "-c", probe_code], env=env, capture_output=True, text=True, timeout=60)
    if probe.returncode:
        log.say("宿主Torch不符合固定LeRobot约束；在独立torch-runtime安装官方2.11.0/cu128组合。宿主不变。")
        log.say("已有独立环境检查：" + probe.stderr[-2000:])
        if shutil.disk_usage(workdir).free < 8 * 2**30:
            raise RuntimeError("独立CUDA wheel安装至少保留8GiB磁盘余量；没有替换宿主Torch")
        log.run([python, "-m", "pip", "install", "--only-binary=:all:",
                 "--index-url", MANAGED_TORCH_INDEX, *MANAGED_TORCH_PINS], env=env)
    log.run([python, "-c", probe_code], env=env)  # 真正做CUDA算子与torchvision ABI检查
    command = [python, Path(__file__).resolve(), "prepare", "--profile", "gpu", "--workdir", workdir,
               "--log-root", log.directory / "managed_prepare"]
    if args.skip_assets:
        command.append("--skip-assets")
    # 不传allow-managed-torch：子解释器必须已经兼容，避免失败后无限递归。
    log.run(command, env=env)


def prepare(args, log):
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("此固定依赖组合使用Python3.12或3.13；不会自动替换系统Python或在其他版本上编译依赖。")
    workdir = args.workdir
    previous = json.loads((workdir / "environment.json").read_text()) if (workdir / "environment.json").exists() else {}
    if previous and previous["profile"] != args.profile:
        raise RuntimeError("此目录已用于另一准备模式；请另选workdir，不会覆盖已准备环境。")
    assets_cached = previous.get("assets_ready", False) and all(
        Path(previous.get(key, "/nonexistent")).is_dir() for key in ["dataset_root", "base_snapshot", "vlm_snapshot"])
    assets_cached = assets_cached and previous.get("data_files") == list(DATA_FILES) and all(
        (Path(previous["dataset_root"]) / file).is_file() for file in DATA_FILES)
    free = shutil.disk_usage(workdir).free / 2**30
    minimum = 6 if args.profile == "gpu" and not assets_cached else 1
    log.say({"resources": resources(), "free_disk_GiB": round(free, 2), "profile": args.profile})
    if free < minimum:
        raise RuntimeError(f"磁盘余量不足：{free:.2f}GiB；{args.profile}准备至少要求{minimum}GiB。")
    import torch
    log.say({"host_torch": torch.__version__, "host_torch_file": torch.__file__,
             "host_cuda_build": torch.version.cuda, "host_cuda_available": torch.cuda.is_available()})
    if not torch_version_supported(torch.__version__) or (args.profile == "gpu" and not torch.cuda.is_available()):
        if args.profile == "gpu" and getattr(args, "allow_managed_torch", False):
            prepare_with_managed_torch(args, log)
            return
        raise RuntimeError(f"实际宿主torch={torch.__version__}，固定LeRobot要求torch>=2.7,<2.12；"
                           "GPU可传--allow-managed-torch在独立环境安装兼容版本，宿主不会改变。")
    if args.profile == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU准备需要已能使用CUDA的PyTorch；在Colab先选择GPU，服务器先安装匹配的CUDA版PyTorch。")

    environment = workdir / "venv"
    env = runtime_env(workdir)
    log.say(f"阶段：创建/检查工作venv {environment}；Python与pip探测各限30秒。")
    python = ensure_environment(environment, log, env)
    log.say("阶段：定位site-packages并链接已验证的Torch/native包，不重复下载CUDA。")
    site = Path(subprocess.check_output([str(python), "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True,env=env,timeout=30).strip())
    # 只借用现有PyTorch/native wheel，避免复制大文件，也不继承全局diffusers/torchao等可选包。
    source_site = Path(torch.__file__).resolve().parent.parent
    names = ["torch", "torchgen", "functorch"]
    patterns = ["torch-*.dist-info"]
    if args.profile == "gpu":
        names += ["torchvision", "torchvision.libs", "torchaudio", "torio", "nvidia", "triton"]
        patterns += ["torchvision-*.dist-info", "torchaudio-*.dist-info", "nvidia_*.dist-info", "triton-*.dist-info"]
    sources = [source_site / name for name in names]
    for pattern in patterns:
        sources.extend(source_site.glob(pattern))
    for source in sources:
        if not source.exists():
            continue
        target = site / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"已有不同的包链接，停止：{target}")
        elif not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
        else:
            raise RuntimeError(f"目标已有独立副本，停止以免混用：{target}")

    from importlib.metadata import version, PackageNotFoundError
    constraints = [f"torch=={version('torch')}"]
    if args.profile == "gpu":
        for name in ["torchvision", "torchaudio"]:
            try:
                constraints.append(f"{name}=={version(name)}")
            except PackageNotFoundError:
                pass
    constraint_file = workdir / "torch-constraints.txt"
    constraint_file.write_text("\n".join(constraints) + "\n")
    log.say("阶段：安装固定Python依赖；缓存位于workdir/pip-cache。")
    log.run([python, "-m", "pip", "install", "--constraint", constraint_file,
             "--report", log.directory / "pip-core-report.json", *PINS], env=env)
    source = workdir / "lerobot"
    if args.profile == "gpu":
        if not (source / ".git").exists():
            source.mkdir(exist_ok=True)
            log.run(["git", "init", str(source)])
            log.run(["git", "-C", source, "remote", "add", "origin", "https://github.com/huggingface/lerobot.git"])
            log.run(["git", "-C", source, "fetch", "--depth=1", "origin", LEROBOT_SHA])
            log.run(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != LEROBOT_SHA:
            raise RuntimeError("已有LeRobot源码提交不同；请使用另一工作目录，不会重置现有源码。")
        import tomllib
        project = tomllib.loads((source / "pyproject.toml").read_text())
        dataset_requirements = project["project"]["optional-dependencies"]["dataset"]
        # 本次是Parquet图像列，视频回退显式使用PyAV；不引入与宿主Torch绑定的可选TorchCodec。
        dataset_requirements = [r for r in dataset_requirements if not r.startswith("torchcodec") and r != "lerobot[av-dep]"]
        log.run([python, "-m", "pip", "install", "--constraint", constraint_file,
                 "--report", log.directory / "pip-lerobot-report.json",
                 "-e", f"{source}[training,smolvla,peft,av-dep,scipy-dep]", *dataset_requirements, *PINS], env=env)
    check = "import torch,peft,transformers,accelerate;print('torch',torch.__version__,'peft',peft.__version__,'transformers',transformers.__version__,'accelerate',accelerate.__version__)"
    if args.profile == "gpu":
        check += ";from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy;import lerobot.scripts.lerobot_train;print('SmolVLA和官方训练入口导入成功')"
    log.run([python, "-c", check], env=env)
    log.run([python, "-m", "pip", "check"], env=env)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True, env=env)
    (log.directory / "pip-freeze.txt").write_text(redact(freeze))
    manifest = {**previous, "profile": args.profile, "python": str(python), "torch_source": str(source_site),
                "lerobot_revision": LEROBOT_SHA if args.profile == "gpu" else None, "assets_ready": bool(assets_cached)}
    if args.profile == "gpu":
        props = torch.cuda.get_device_properties(0)
        manifest["gpu"] = {"name": props.name, "memory_GiB": props.total_memory / 2**30,
                           "capability": list(torch.cuda.get_device_capability(0)), "torch_cuda": torch.version.cuda}
        log.say(manifest["gpu"])
    write_json(workdir / "environment.json", manifest)
    if args.profile == "gpu" and not args.skip_assets and not assets_cached:
        # 另起已准备的解释器下载，避免当前Notebook的旧包缓存影响结果。
        log.run([python, Path(__file__).resolve(), "assets", "--workdir", workdir], env=env)
    elif assets_cached:
        log.say("复用已记录的模型/数据快照，不重新解析VLM的main。")
        if args.profile == "gpu":
            log.run([python, Path(__file__).resolve(), "check-data", "--workdir", workdir], env=env)
    current_manifest = json.loads((workdir / "environment.json").read_text())
    if args.profile == "gpu" and current_manifest.get("assets_ready"):
        # 导入成功不等于配置可从离线缓存加载；实际解析固定SHA的config.json。
        config_check = (
            "from lerobot.configs.policies import PreTrainedConfig;"
            "from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig;"
            f"c=PreTrainedConfig.from_pretrained({BASE_ID!r},revision={BASE_SHA!r},local_files_only=True);"
            "print('固定SHA的策略配置离线读取成功：',c.type,'chunk_size=',c.chunk_size)"
        )
        log.run([python, "-c", config_check], env=env)
    log.say(f"准备完成。解释器：{python}；不会自动开始训练。")


def assets(args, log):
    from huggingface_hub import snapshot_download
    workdir = args.workdir
    manifest = json.loads((workdir / "environment.json").read_text())
    dataset = snapshot_download(DATA_ID, repo_type="dataset", revision=DATA_SHA, token=False,
                                local_dir=workdir / "dataset", allow_patterns=["meta/**", *DATA_FILES])
    model_files = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"]
    base = snapshot_download(BASE_ID, revision=BASE_SHA, token=False, allow_patterns=model_files)
    vlm = snapshot_download(VLM_ID, revision=manifest.get("vlm_resolved_revision") or "main", token=False, allow_patterns=model_files)
    manifest.update({"assets_ready": False, "dataset_root": dataset, "dataset_revision": DATA_SHA,
                     "data_files": list(DATA_FILES),
                     "base_revision": BASE_SHA, "base_snapshot": base,
                     "vlm_snapshot": vlm, "vlm_resolved_revision": Path(vlm).name})
    write_json(workdir / "environment.json", manifest)
    check_data(args, log)
    manifest["assets_ready"] = True
    write_json(workdir / "environment.json", manifest)
    log.say("模型和选定数据文件准备完成，真实样本读取通过；尚未加载完整模型或执行更新。")


def validate_dataset_files(root, episodes=(*TRAIN_EPISODES, *HOLDOUT_EPISODES)):
    """核对实际数据行与episode元信息；不信任发布的data/file_index。"""
    import pyarrow.parquet as pq
    root = Path(root)
    columns = ["episode_index", "task_index", "index", "frame_index"]
    selected = {episode: [] for episode in episodes}
    locations = {episode: [] for episode in episodes}
    for file in sorted((root / "data").glob("*/*.parquet")):
        found = set()
        for row in pq.read_table(file, columns=columns).to_pylist():
            episode = row["episode_index"]
            if episode in selected:
                selected[episode].append(row)
                found.add(episode)
        for episode in found:
            locations[episode].append(str(file.relative_to(root)))
    meta_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    if not meta_files:
        raise RuntimeError("缺少episode元信息，无法验证帧数和边界。")
    records = pq.read_table(meta_files, columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"]).to_pylist()
    meta = {row["episode_index"]: row for row in records}
    report = {}
    for episode, rows in selected.items():
        expected = meta[episode]
        if len(rows) != expected["length"]:
            raise RuntimeError(f"episode {episode}实际找到{len(rows)}帧，元信息要求{expected['length']}帧；请修复下载文件，不能进入训练。")
        if (sorted(row["frame_index"] for row in rows) != list(range(expected["length"]))
                or sorted(row["index"] for row in rows) != list(range(expected["dataset_from_index"], expected["dataset_to_index"]))
                or {row["task_index"] for row in rows} != {24}):
            raise RuntimeError(f"episode {episode}的帧索引、全局索引或任务不符合固定实验配置。")
        report[str(episode)] = {"frames": len(rows), "files": locations[episode], "task_index": 24}
    return report


def check_data(args, log):
    """CPU读取真实示范及50步动作标签，不加载模型、不更新参数。"""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest = json.loads((args.workdir / "environment.json").read_text())
    root = Path(manifest["dataset_root"])
    report = {"episodes": validate_dataset_files(root), "scope": "raw dataset CPU preflight; no policy update"}
    fps = json.loads((root / "meta/info.json").read_text())["fps"]
    dataset = LeRobotDataset(DATA_ID, root=root, revision=DATA_SHA, episodes=list(TRAIN_EPISODES),
                            delta_timestamps={"action": [i / fps for i in range(50)]}, video_backend="pyav")
    expected_rows = sum(report["episodes"][str(ep)]["frames"] for ep in TRAIN_EPISODES)
    if len(dataset) != expected_rows:
        raise RuntimeError("官方读取器的训练帧数不符，停止。")
    sample = dataset[0]
    report["train_rows"] = len(dataset)
    report["first_sample"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                  "finite": bool(torch.isfinite(v).all())} for k, v in sample.items() if isinstance(v, torch.Tensor)}
    report["task"] = sample.get("task")
    if not all(value["finite"] for value in report["first_sample"].values()):
        raise RuntimeError("真实示范样本包含非有限数，停止。")
    if sample["action"].shape != (50, 7) or sample["observation.state"].shape != (8,):
        raise RuntimeError("原始示范的状态或50步动作维度不符合本实验。")
    boundary = dataset[report["episodes"][str(TRAIN_EPISODES[0])]["frames"] - 1]
    report["last_frame_action_padding"] = int(boundary["action_is_pad"].sum())
    if report["last_frame_action_padding"] != 49:
        raise RuntimeError("episode末帧的动作padding不符合50步标签边界。")
    write_json(args.workdir / "dataset_check.json", report)
    log.say(report)


def cpu_lab(args, log):
    import torch
    from torch import nn
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(7)
    output = args.workdir / "cpu_lora"
    if output.exists():
        raise FileExistsError(f"保留已有CPU实验：{output}；另选workdir重做。")
    output.mkdir()

    class ProjectionPair(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(720, 960, bias=False)
            self.o_proj = nn.Linear(960, 720, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x))

    base = ProjectionPair()
    base_state = {name: p.detach().clone() for name, p in base.state_dict().items()}
    x = torch.randn(1, 50, 720)
    with torch.no_grad():
        original_output = base(x)
        target = original_output + .1 * x
    policy = get_peft_model(base, LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj"], lora_dropout=0.0))
    print("\n" + "="*50)
    print("LoRA可训练参数")
    total_trainable = 0

    for name, param in policy.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"参数名称: {name:60s} | 形状: {str(list(param.shape)):20s} | 数量: {num_params}")
            total_trainable += num_params

    print("-" * 50)
    print(f"可训练参数总数: {total_trainable}")
    print(f"目标值: 13440")
    print(f"是否匹配: {total_trainable == 13440}")
    print("="*50 + "\n")

    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
    evidence = {"scope": "CPU synthetic-activation LoRA mechanism experiment, not SmolVLA policy training",
                "input_shape": list(x.shape), "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                "module_after_injection": str(policy.base_model.model.q_proj), "weight_decay": 0.0, "steps": []}
    torch.testing.assert_close(policy(x), original_output)
    policy.train()
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
        loss = torch.nn.functional.mse_loss(policy(x), target)
        loss.backward()
        gradients = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in policy.named_parameters() if p.requires_grad}
        optimizer.step()
        deltas = {n: float((p.detach() - before[n]).abs().max()) for n, p in policy.named_parameters() if p.requires_grad}
        row = {"step": step + 1, "loss": float(loss.detach()), "gradient_norms": gradients, "parameter_max_changes": deltas}
        evidence["steps"].append(row)
        log.say(row)
    for name, weight in base_state.items():
        current = policy.base_model.model.q_proj.base_layer.weight if name == "q_proj.weight" else policy.base_model.model.o_proj.weight
        torch.testing.assert_close(current, weight, rtol=0, atol=0)
    policy.eval()
    expected = policy(x).detach()
    with torch.no_grad():
        projection = policy.base_model.model.q_proj
        base_q = projection.base_layer(x)
        low_rank = projection.lora_A["default"](x)
        correction = projection.lora_B["default"](low_rank) * projection.scaling["default"]
        adapted_q = projection(x)
        torch.testing.assert_close(adapted_q, base_q + correction)
        evidence["projection_example"] = {
            "scope": "measured CPU projection values with synthetic hidden inputs; not robot actions",
            "x_shape": list(x.shape), "A_output_shape": list(low_rank.shape), "q_output_shape": list(adapted_q.shape),
            "base_first3": base_q[0, 0, :3].tolist(), "correction_first3": correction[0, 0, :3].tolist(),
            "adapted_first3": adapted_q[0, 0, :3].tolist(), "scale": projection.scaling["default"]}
    torch.save(base_state, output / "base.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    policy.save_pretrained(output / "adapter")
    restored_base = ProjectionPair()
    restored_base.load_state_dict(torch.load(output / "base.pt", weights_only=True))
    restored = PeftModel.from_pretrained(restored_base, output / "adapter", is_trainable=True)
    restored.eval()
    error = float((restored(x).detach() - expected).abs().max())
    torch.testing.assert_close(restored(x), expected, rtol=1e-5, atol=1e-6)
    resumed_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
    resumed_optimizer.load_state_dict(torch.load(output / "optimizer.pt", weights_only=True))
    resumed_optimizer.zero_grad(set_to_none=True)
    resume_loss = torch.nn.functional.mse_loss(restored(x), target)
    resume_loss.backward()
    resumed_optimizer.step()
    evidence.update({"frozen_base_unchanged": True, "reload_max_error": error,
                     "optimizer_reloaded_and_one_more_step_completed": True})
    write_json(output / "evidence.json", evidence)
    log.say("CPU机制实验完成；冻结、梯度、参数变化、adapter重载与optimizer续一步均已检查。")


def archive_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    files = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file() and not p.is_symlink()}
    if not files or not (source / "pretrained_model").is_dir() or not (source / "training_state").is_dir():
        raise RuntimeError("checkpoint缺少模型或训练状态目录，拒绝标记完整。")
    if destination.exists():
        recorded = json.loads((destination / "COMPLETE.json").read_text())
        if recorded != files or any(not (destination / n).is_file() or digest(destination / n) != h for n, h in files.items()):
            raise RuntimeError("已有归档不同或不完整，拒绝覆盖。")
        return
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    extras = {str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()} - set(files) - {"COMPLETE.json"}
    if extras:
        raise RuntimeError("部分归档含其他文件，保留检查，拒绝混合不同checkpoint。")
    for name, expected in files.items():
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != expected:
            raise RuntimeError(f"部分归档内容不同，保留以检查：{target}")
        if not target.exists():
            shutil.copy2(source / name, target)
        if digest(target) != expected:
            raise RuntimeError(f"归档校验失败：{name}")
    write_json(staging / "COMPLETE.json", files)
    staging.rename(destination)


def save_inspection_images(batch, directory):
    """导出处理后batch的最后一帧，兼容[B,C,H,W]与[B,T,C,H,W]。"""
    import torch
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, value in batch.items():
        if not name.startswith("observation.images.") or value.ndim not in (4, 5):
            continue
        image = value[0, -1] if value.ndim == 5 else value[0]
        if image.shape[0] != 3:
            continue
        rgb = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).numpy()
        path = directory / (name.replace(".", "_") + ".png")
        Image.fromarray(rgb).save(path)
        files.append(path.name)
    return files


def load_policy_config(loader, pretrained_name_or_path, *args, **kwargs):
    """LeRobot的CLI配置读取未转发pretrained_revision；与预下载的SHA保持一致。"""
    if str(pretrained_name_or_path) == BASE_ID:
        if kwargs.get("revision") not in (None, BASE_SHA):
            raise ValueError("策略配置请求的revision与本实验固定的base SHA不一致。")
        kwargs["revision"] = BASE_SHA
        kwargs["local_files_only"] = True
    return loader(pretrained_name_or_path, *args, **kwargs)


def train(args, log):
    import gc
    import torch
    from unittest.mock import patch
    practice_step = None
    if args.practice_step_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_training_step", args.practice_step_file)
        practice_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(practice_module)
        practice_step = practice_module.train_one_batch
        if not callable(practice_step):
            raise TypeError("练习文件必须定义可调用的train_one_batch。")
    manifest = json.loads((args.workdir / "environment.json").read_text())
    if not manifest.get("assets_ready") or not torch.cuda.is_available():
        raise RuntimeError("先完成GPU prepare及资产下载，再启动完整策略训练。")
    from lerobot.scripts import lerobot_train
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from transformers import AutoModelForImageTextToText
    original_config_loader = PreTrainedConfig.from_pretrained

    def load_config(pretrained_name_or_path, *a, **kw):
        return load_policy_config(original_config_loader, pretrained_name_or_path, *a, **kw)

    log.say({"base_config_revision": BASE_SHA, "base_snapshot": manifest.get("base_snapshot"),
             "note": "配置、权重使用同一固定SHA；不依赖缓存中的main引用。"})
    if not Path(lerobot_train.__file__).resolve().is_relative_to((args.workdir / "lerobot").resolve()):
        raise RuntimeError("导入了另一份LeRobot；请用准备生成的解释器并移除外部PYTHONPATH。")
    if practice_step:
        output = (args.archive_dir or args.workdir) / "practice_step"
    else:
        output = args.workdir / ("batch_inspection" if args.inspect_only else "train_run")
    if output.exists():
        raise FileExistsError("保留已有训练目录；续训须单独核验checkpoint，本入口不会覆盖或假装自动续训。")
    evidence = {"scope": "user-written one-batch exercise" if practice_step else "official SmolVLA PEFT training",
                "source_manifest": manifest, "steps": []}
    original_update = lerobot_train.update_policy
    original_save = lerobot_train.save_checkpoint
    original_make = lerobot_train.make_policy
    holder = {}

    def capture_make(*a, **kw):
        holder["dataset_meta"] = kw["ds_meta"]
        return original_make(*a, **kw)

    def audited_update(*positional, **keywords):
        values = inspect.signature(original_update).bind(*positional, **keywords).arguments
        policy, optimizer = values["policy"], values["optimizer"]
        accelerator = values["accelerator"]
        if accelerator.num_processes != 1:
            raise RuntimeError("此实践入口只验证单进程训练。")
        if practice_step and accelerator.mixed_precision != "no":
            raise RuntimeError("手写单步练习暂不接入AMP/GradScaler；请使用本入口的use_amp=false配置。")
        unwrapped = accelerator.unwrap_model(policy)
        holder["policy"] = unwrapped
        if not evidence.get("trainable"):
            evidence["trainable"] = {n: list(p.shape) for n, p in policy.named_parameters() if p.requires_grad}
            evidence["trainable_count"] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            evidence["total_count"] = sum(p.numel() for p in policy.parameters())
            evidence["accelerator_mixed_precision"] = accelerator.mixed_precision
            dtype_counts = {}
            for p in policy.parameters():
                key = str(p.dtype)
                dtype_counts[key] = dtype_counts.get(key, 0) + p.numel()
            evidence["parameter_dtype_counts"] = dtype_counts
            for config in unwrapped.peft_config.values():
                config.revision = BASE_SHA
            batch = {n: v.detach().cpu() for n, v in values["batch"].items() if isinstance(v, torch.Tensor)}
            evidence["first_processed_batch"] = {n: {"shape": list(v.shape), "dtype": str(v.dtype),
                                                       "finite": bool(torch.isfinite(v).all()),
                                                       "min": float(v.min()) if v.numel() else None,
                                                       "max": float(v.max()) if v.numel() else None} for n, v in batch.items()}
            if not all(v["finite"] for v in evidence["first_processed_batch"].values()):
                raise RuntimeError("首个处理后batch含非有限数。")
            output.mkdir(parents=True, exist_ok=True)
            torch.save(batch, output / "first_processed_batch.pt")
            if args.inspect_only:
                evidence["inspection_images"] = save_inspection_images(batch, output)
                write_json(output / "batch_inspection.json", evidence)
                raise BatchInspectionComplete()
        original_step = optimizer.step
        original_forward = policy.forward
        row = {"step": len(evidence["steps"]) + 1, "optimizer_step_calls": 0}

        def audited_forward(*a, **kw):
            result = original_forward(*a, **kw)
            loss = result[0]
            row["loss"] = float(loss.detach())
            if not math.isfinite(row["loss"]) or not loss.requires_grad:
                raise RuntimeError("loss非有限或没有梯度路径，停止。")
            return result

        def audited_step(*a, **kw):
            row["optimizer_step_calls"] += 1
            if practice_step and row["optimizer_step_calls"] != 1:
                raise RuntimeError("单batch练习只能调用一次optimizer.step。")
            before = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
            gradients = [p.grad.detach() for p in policy.parameters() if p.requires_grad and p.grad is not None]
            row["any_nonzero_gradient"] = any(bool(g.abs().max() > 0) for g in gradients)
            row["all_recorded_gradients_finite"] = all(bool(torch.isfinite(g).all()) for g in gradients)
            if not gradients or not row["all_recorded_gradients_finite"]:
                raise RuntimeError("没有adapter梯度或梯度非有限，停止。")
            result = original_step(*a, **kw)
            row["any_trainable_parameter_changed"] = any(not torch.equal(before[n], p.detach().cpu()) for n, p in policy.named_parameters() if p.requires_grad)
            return result

        with patch.object(optimizer, "step", audited_step), patch.object(policy, "forward", audited_forward):
            if practice_step:
                row["returned_loss"] = float(practice_step(policy, values["batch"], optimizer))
                if (row["optimizer_step_calls"] != 1 or "loss" not in row
                        or not math.isfinite(row["returned_loss"])
                        or not math.isclose(row["returned_loss"], row["loss"], rel_tol=1e-5, abs_tol=1e-7)
                        or not row.get("any_trainable_parameter_changed")):
                    raise RuntimeError("练习未完成有效前向/更新，不能标记通过。")
                result = None
            else:
                result = original_update(*positional, **keywords)
        row["cuda_peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        row["cuda_peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
        evidence["steps"].append(row)
        write_json(output / "training_evidence.json", evidence)
        log.say(row)
        if practice_step:
            unwrapped.save_pretrained(output / "adapter")
            torch.save(optimizer.state_dict(), output / "optimizer.pt")
            shutil.copy2(args.practice_step_file, output / "train_one_batch.py")
            raise PracticeStepComplete()
        return result

    def saved_checkpoint(*a, **kw):
        result = original_save(*a, **kw)
        values = inspect.signature(original_save).bind(*a, **kw).arguments
        source = Path(values["checkpoint_dir"])
        holder["checkpoint"] = source
        holder["postprocessor"] = values["postprocessor"]
        shutil.copy2(output / "training_evidence.json", source / "training_evidence.json")
        if args.archive_dir:
            archive_checkpoint(source, args.archive_dir / "checkpoints" / source.name)
            log.say(f"完整checkpoint已核对归档：{args.archive_dir / 'checkpoints' / source.name}")
        return result

    original_loader = AutoModelForImageTextToText.from_pretrained
    def load_vlm(*a, **kw):
        if "torch_dtype" in kw:
            kw.setdefault("dtype", kw.pop("torch_dtype"))
        kw.setdefault("token", False)
        return original_loader(*a, **kw)

    arguments = [f"--policy.path={BASE_ID}", f"--policy.pretrained_revision={BASE_SHA}",
                 "--policy.input_features=null", "--policy.output_features=null", "--policy.device=cuda",
                 "--policy.use_amp=false", "--policy.push_to_hub=false", "--policy.optimizer_lr=1e-4",
                 "--policy.scheduler_warmup_steps=0", f"--dataset.repo_id={DATA_ID}",
                 f"--dataset.revision={DATA_SHA}", f"--dataset.root={manifest['dataset_root']}",
                 f"--dataset.episodes={json.dumps(TRAIN_EPISODES)}", "--dataset.video_backend=pyav", "--peft.method_type=LORA",
                 "--peft.r=8", "--peft.lora_alpha=8", "--batch_size=1", "--num_workers=0",
                 "--steps=5", "--log_freq=1", "--save_freq=5", "--env_eval_freq=0",
                 "--wandb.enable=false", f"--output_dir={output}"]
    with ExitStack() as stack:
        stack.enter_context(patch.object(PreTrainedConfig, "from_pretrained", staticmethod(load_config)))
        stack.enter_context(patch.object(lerobot_train, "update_policy", audited_update))
        stack.enter_context(patch.object(lerobot_train, "save_checkpoint", saved_checkpoint))
        stack.enter_context(patch.object(lerobot_train, "make_policy", capture_make))
        stack.enter_context(patch.object(AutoModelForImageTextToText, "from_pretrained", staticmethod(load_vlm)))
        stack.enter_context(patch.object(sys, "argv", ["lerobot-train", *arguments]))
        try:
            lerobot_train.main()
        except BatchInspectionComplete:
            log.say(f"真实batch与可训练参数已保存至{output}；尚未backward或optimizer.step。")
            return
        except PracticeStepComplete:
            log.say(f"你的train_one_batch已完成一次真实更新，证据与adapter：{output}；此目录不是官方可续训checkpoint。")
            return
    if not any(row.get("any_trainable_parameter_changed") for row in evidence["steps"]):
        raise RuntimeError("训练返回但未测到参数更新，不能验收为完成训练。")
    # 保存后的策略重载：同一处理后输入、同一初始噪声，不再重复归一化。
    batch_cpu = torch.load(output / "first_processed_batch.pt", weights_only=True)
    policy = holder.pop("policy")
    policy.eval()
    noise = torch.randn(len(batch_cpu["observation.state"]), policy.config.chunk_size,
                        policy.config.max_action_dim, generator=torch.Generator().manual_seed(71))
    batch = {n: v.to("cuda") for n, v in batch_cpu.items()}
    with torch.inference_mode():
        expected = policy.predict_action_chunk(batch, noise=noise.to("cuda")).cpu()
        repeated = policy.predict_action_chunk(batch, noise=noise.to("cuda")).cpu()
        expected_action = holder["postprocessor"](expected[:, 0].clone().to("cuda")).cpu()
    torch.testing.assert_close(repeated, expected, rtol=1e-3, atol=1e-4)
    del policy, batch, repeated
    gc.collect()
    torch.cuda.empty_cache()
    pretrained = holder["checkpoint"] / "pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(pretrained)
    cfg.pretrained_path = pretrained
    cfg.device = "cuda"
    with patch.object(AutoModelForImageTextToText, "from_pretrained", staticmethod(load_vlm)):
        restored = original_make(cfg=cfg, ds_meta=holder["dataset_meta"])
    restored.eval()
    _, restored_post = make_pre_post_processors(policy_cfg=restored.config, pretrained_path=pretrained)
    with torch.inference_mode():
        actual = restored.predict_action_chunk({n: v.to("cuda") for n, v in batch_cpu.items()}, noise=noise.to("cuda")).cpu()
        actual_action = restored_post(actual[:, 0].clone().to("cuda")).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(actual_action, expected_action, rtol=1e-3, atol=1e-4)
    reload_evidence = {"fixed_noise_seed": 71, "normalized_chunk_max_error": float((actual - expected).abs().max()),
                       "postprocessed_first_action_max_error": float((actual_action - expected_action).abs().max()),
                       "processors_loaded": True, "note": "GPU runtime validation; not task success or exact training-resume validation"}
    write_json(output / "reload_evidence.json", reload_evidence)
    if args.archive_dir:
        write_json(args.archive_dir / "reload_evidence.json", reload_evidence)
    log.say(reload_evidence)
    log.say("官方5步训练、固定输入/噪声重载比较已返回；真实任务表现仍需另行评测。")


def export_notebook(args):
    source = Path(__file__).read_text()
    practice_source = (Path(__file__).resolve().parents[1] / "studies/policy_generalization/train_one_batch.py").read_text()
    metric_source = (Path(__file__).resolve().parents[1] / "studies/policy_generalization/evaluation_metrics.py").read_text()
    eval_source = Path(__file__).with_name("smolvla_eval_tools.py").read_text()
    sim_source = Path(__file__).with_name("smolvla_libero_tools.py").read_text()
    cell = ("# 一次执行完成训练环境和资产准备；不会自动训练。\n"
            "from pathlib import Path\nimport subprocess, sys, os, shutil, traceback, json\nfrom datetime import datetime, timezone\n"
            "WORKDIR = Path('/content/smolvla-training')  # AutoDL可改为自己的持久化目录\nPROFILE = 'gpu'\nMOUNT_DRIVE = True  # 非Colab自动跳过；已有挂载不会重复授权\n"
            "PREPARE_ENV = not ((WORKDIR / 'environment.json').is_file() and (WORKDIR / 'venv/bin/python').is_file() and json.loads((WORKDIR / 'environment.json').read_text()).get('assets_ready', False))  # 新运行时/未完成准备时安装，已完成时只同步\n"
            "WORKDIR.mkdir(parents=True, exist_ok=True)\n"
            f"SOURCE = {source!r}\n"
            "SCRIPT = WORKDIR / 'smolvla_training_workbench.py'\nSCRIPT.write_text(SOURCE)\n"
            f"PRACTICE_SOURCE = {practice_source!r}\n"
            "PRACTICE_FILE = WORKDIR / 'train_one_batch.py'\n"
            "if not PRACTICE_FILE.exists(): PRACTICE_FILE.write_text(PRACTICE_SOURCE)\n"
            "else: print('保留已有练习文件：', PRACTICE_FILE, '；本地修改后需显式同步此文件。')\n"
            f"METRICS_SOURCE = {metric_source!r}\nEVAL_SOURCE = {eval_source!r}\n"
            "METRICS_FILE = WORKDIR / 'evaluation_metrics.py'\n"
            "if not METRICS_FILE.exists(): METRICS_FILE.write_text(METRICS_SOURCE)\n"
            "(WORKDIR / 'smolvla_eval_tools.py').write_text(EVAL_SOURCE)\n"
            f"SIM_SOURCE = {sim_source!r}\n"
            "(WORKDIR / 'smolvla_libero_tools.py').write_text(SIM_SOURCE)\n"
            "ARCHIVE = None\n"
            "LOG_ROOT = WORKDIR / 'logs'\n"
            "try:\n"
            "    if MOUNT_DRIVE and any(k in os.environ for k in ('COLAB_RELEASE_TAG', 'COLAB_BACKEND_VERSION')):\n"
            "        if not os.path.ismount('/content/drive'):\n"
            "            from google.colab import drive\n"
            "            drive.mount('/content/drive')\n"
            "        if not os.path.ismount('/content/drive'): raise RuntimeError('Drive未实际挂载，停止以免误存到临时目录')\n"
            "        ARCHIVE = Path('/content/drive/MyDrive/robot-lab/experiments/smolvla_peft_20261103_001')\n"
            "        ARCHIVE.mkdir(parents=True, exist_ok=True)\n"
            "        LOG_ROOT = ARCHIVE / 'setup_logs'\n"
            "    command = [sys.executable, str(SCRIPT), 'prepare', '--profile', PROFILE, '--workdir', str(WORKDIR)]\n"
            "    if PROFILE == 'gpu': command.append('--allow-managed-torch')\n"
            "    command += ['--log-root', str(LOG_ROOT)]\n"
            "    if PREPARE_ENV:\n"
            "        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace') as process:\n"
            "            for line in process.stdout: print(line, end='', flush=True)\n"
            "            code = process.wait()\n"
            "        if code: raise RuntimeError(f'准备失败，退出码{code}；完整日志：{LOG_ROOT}')\n"
            "        print('准备完成；不会自动训练。')\n"
            "    else:\n"
            "        assert (WORKDIR / 'venv/bin/python').is_file() and (WORKDIR / 'environment.json').is_file(), '本运行时尚未准备，需设PREPARE_ENV=True'\n"
            "        print('仅同步代码，跳过安装/下载；未重新验证环境。后续显式选择评测入口。')\n"
            "except Exception:\n"
            "    (WORKDIR / 'prepare_cell_error.log').write_text(traceback.format_exc())\n"
            "    if ARCHIVE is not None: (ARCHIVE / 'prepare_cell_error.log').write_text(traceback.format_exc())\n"
            "    raise\n"
            "finally:\n"
            "    if PREPARE_ENV and ARCHIVE is not None and (WORKDIR / 'logs').is_dir():\n"
            "        saved_logs = ARCHIVE / ('setup_logs_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))\n"
            "        shutil.copytree(WORKDIR / 'logs', saved_logs, dirs_exist_ok=False)\n"
            "        print('准备日志已归档到：', saved_logs)\n")
    notebook = {"cells": [{"cell_type": "code", "id": "training-prepare", "metadata": {},
                            "source": cell.splitlines(keepends=True), "outputs": [], "execution_count": None}],
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    if args.notebook.exists():
        existing = json.loads(args.notebook.read_text())
        matches = [i for i, c in enumerate(existing['cells']) if c.get('id') == 'training-prepare']
        if len(matches) != 1:
            raise RuntimeError("无法唯一定位自动生成的准备单元格；保留原Notebook，另选--notebook路径。")
        previous_cell = existing['cells'][matches[0]]
        notebook['cells'][0]['outputs'] = previous_cell.get('outputs', [])
        notebook['cells'][0]['execution_count'] = previous_cell.get('execution_count')
        notebook['cells'][0]['metadata']['source_updated_after_execution'] = True
        existing['cells'][matches[0]] = notebook['cells'][0]
        notebook = existing  # 保留用户自己添加的GPU检查等单元格与输出。
    write_json(args.notebook, notebook)
    print(args.notebook)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "assets", "check-data", "cpu", "train", "export-eval", "score-eval", "check-metric", "prepare-sim", "repair-sim", "check-sim", "eval-libero", "export-notebook"])
    parser.add_argument("--workdir", type=Path, default=Path("outputs/training_workbench"))
    parser.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--allow-managed-torch", action="store_true",
                        help="GPU宿主Torch不兼容时在独立环境安装官方2.11.0/cu128，不修改宿主")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true", help="train入口只保存首个真实batch，不执行参数更新")
    parser.add_argument("--practice-step-file", type=Path, help="train入口显式调用此文件的train_one_batch一次，然后结束；不影响正式五步入口")
    metric_default = Path(__file__).with_name("evaluation_metrics.py")
    if not metric_default.exists():
        metric_default = Path(__file__).resolve().parents[1] / "studies/policy_generalization/evaluation_metrics.py"
    parser.add_argument("--metric-file", type=Path, default=metric_default)
    parser.add_argument("--checkpoint", type=Path, help="已训练checkpoint的pretrained_model目录")
    parser.add_argument("--predictions", type=Path, help="score-eval读取的paired_predictions.pt")
    parser.add_argument("--eval-run", default="heldout_825_001")
    parser.add_argument("--eval-noise-seed", type=int, default=71)
    parser.add_argument("--episodes", type=int, default=1, help="LIBERO每条件回合数，首次只跑1回合")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--sim-fps", type=int, default=20, help="明确的仿真控制/录像FPS，不从数据FPS推断")
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--condition", choices=["both", "base", "trained"], default="both")
    parser.add_argument("--log-root", type=Path, help="可指定持久化日志目录；不影响权重/数据缓存位置")
    parser.add_argument("--notebook", type=Path, default=Path("scripts/smolvla_training_prepare.ipynb"))
    args = parser.parse_args(argv)
    if min(args.episodes, args.sim_fps, args.max_steps) <= 0:
        parser.error("episodes/sim-fps/max-steps必须为正整数")
    if args.practice_step_file and (args.mode != "train" or args.inspect_only):
        parser.error("--practice-step-file只用于train，不能与--inspect-only同时使用")
    if args.mode == "export-notebook":
        export_notebook(args)
        return
    args.workdir = args.workdir.absolute()
    args.workdir.mkdir(parents=True, exist_ok=True)
    check_persistent_path(args.archive_dir)
    check_persistent_path(args.log_root)
    os.environ.update(runtime_env(args.workdir))
    os.environ.pop("PYTHONPATH", None)
    if args.mode in ("train", "export-eval", "eval-libero"):
        os.environ["HF_HUB_OFFLINE"] = "1"
    log = Log((args.log_root or args.workdir / "logs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + args.mode))
    original_stdout, original_stderr = sys.stdout, sys.stderr
    console = (log.directory / "console.log").open("a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(original_stdout, console), Tee(original_stderr, console)
    try:
        if args.mode in ("prepare-sim", "repair-sim", "check-sim", "eval-libero"):
            import smolvla_libero_tools
            function = {"prepare-sim": smolvla_libero_tools.prepare_entry,
                        "repair-sim": smolvla_libero_tools.repair_entry,
                        "check-sim": smolvla_libero_tools.check_entry,
                        "eval-libero": smolvla_libero_tools.evaluate_entry}[args.mode]
            function(args, log, sys.modules[__name__])
        elif args.mode in ("export-eval", "score-eval", "check-metric"):
            import smolvla_eval_tools
            function = {"export-eval": smolvla_eval_tools.export_entry,
                        "score-eval": smolvla_eval_tools.score_entry,
                        "check-metric": smolvla_eval_tools.check_metric_entry}[args.mode]
            function(args, log, sys.modules[__name__])
        else:
            {"prepare": prepare, "assets": assets, "check-data": check_data, "cpu": cpu_lab, "train": train}[args.mode](args, log)
    except Exception as error:
        log.say(traceback.format_exc())
        write_json(log.directory / "status.json", {"ok": False, "type": type(error).__name__, "error": redact(str(error))})
        raise SystemExit(1) from None
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        console.close()
    write_json(log.directory / "status.json", {"ok": True, "mode": args.mode, "inspect_only": args.inspect_only})


if __name__ == "__main__":
    # 终端/ROS可能通过PYTHONPATH混入另一份库。只重启本CLI子进程，不改父终端或Notebook环境。
    if "PYTHONPATH" in os.environ:
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], clean_env)
    main()
