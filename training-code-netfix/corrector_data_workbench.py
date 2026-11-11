"""Corrector数据试运行：CPU选三条真实示范，GPU调用上游视觉提取器，逐轨迹校验归档。

这不是预测器训练器；不复制MLP/损失实现。完整缓存每episode独立，断连只重做未完成episode。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import time

DATA_ID = "HuggingFaceVLA/libero"
DATA_SHA = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
CORRECTOR_SHA = "927a06da8f00d7df483459e9039219e076c846de"
POLICY_ID = "lerobot/smolvla_libero"
POLICY_SHA = "a24f1f380461b665158d52acd140152c79e8b36d"
EPISODES = {813: (213, 155, "train"), 818: (214, 164, "train"), 825: (215, 153, "validation")}
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def decode_rgb(blob):
    """PIL转成拥有独立、可写内存的RGB数组，避免Torch读取只读NumPy时报警。"""
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as image:
        return np.array(image.convert("RGB"), copy=True)


def load_vlm_compat(loader, *args, **kwargs):
    """固定上游旧参数名适配新Transformers；不修改模型dtype或参考仓库。"""
    if "torch_dtype" in kwargs:
        old = kwargs.pop("torch_dtype")
        if "dtype" in kwargs and kwargs["dtype"] != old:
            raise ValueError("dtype与torch_dtype冲突，拒绝静默改变精度")
        kwargs["dtype"] = old
    return loader(*args, **kwargs)


def cached_policy_snapshot(allow_network):
    """完整策略优先使用固定revision缓存；数据下载阶段仍可联网。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        snapshot = snapshot_download(POLICY_ID, revision=POLICY_SHA, token=False, local_files_only=True)
        if not (Path(snapshot) / 'model.safetensors').is_file():
            raise LocalEntryNotFoundError('本地snapshot尚无完整策略权重')
        return snapshot
    except LocalEntryNotFoundError:
        if not allow_network:
            raise
        print('固定策略snapshot未缓存，开始下载完整SmolVLA checkpoint。', flush=True)
        return snapshot_download(POLICY_ID, revision=POLICY_SHA, token=False)


def load_complete_policy(loader, path, cfg, device, **kwargs):
    """省去原始VLM权重加载；完整checkpoint严格覆盖参数，视觉精度保持上游bf16。"""
    import torch
    if not (Path(path) / 'model.safetensors').is_file():
        raise FileNotFoundError('需要完整本地SmolVLA model.safetensors，不能用随机初始化模型提取特征')
    cfg.load_vlm_weights = False
    cfg.device = device
    kwargs.update(config=cfg, local_files_only=True, strict=True)
    policy = loader(path, **kwargs)
    # 上游原路径AutoModel.from_pretrained(torch_dtype=bfloat16)；这里只改初始化来源，
    # 恢复视觉模块原本精度，不把此修复悄悄变成float32特征协议。
    policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    return policy


def cached_visual_extractor(extractor_cls, snapshot, device, *, allow_network=False):
    """保留上游Extractor与预处理；局部适配其初始化，不修改安装包/参考源码。"""
    from unittest.mock import patch
    from transformers import AutoConfig, AutoProcessor
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg = SmolVLAConfig.from_pretrained(snapshot, local_files_only=True)
    policy_loader = SmolVLAPolicy.from_pretrained
    config_loader, processor_loader = AutoConfig.from_pretrained, AutoProcessor.from_pretrained

    def policy(*a, **kw):
        return load_complete_policy(policy_loader, *a, cfg=cfg, device=device, **kw)

    def cached(loader):
        def call(*a, **kw):
            kw['local_files_only'] = True
            try:
                return loader(*a, **kw)
            except OSError:
                if not allow_network:
                    raise
                print('配置/processor缓存缺少文件，仅补齐小配置或分词器文件。', flush=True)
                kw['local_files_only'] = False
                return loader(*a, **kw)
        return call

    print('从缓存严格加载完整SmolVLA；只读已有VLM配置/processor，不重复下载原始VLM权重。', flush=True)
    with patch.object(SmolVLAPolicy, 'from_pretrained', staticmethod(policy)), \
         patch.object(AutoConfig, 'from_pretrained', staticmethod(cached(config_loader))), \
         patch.object(AutoProcessor, 'from_pretrained', staticmethod(cached(processor_loader))):
        try:
            encoder = extractor_cls(snapshot, device=device, local_files_only=True)
        except OSError as error:
            raise RuntimeError('完整策略/配置/processor加载失败，未回退随机权重；'
                               '请保留前面的完整异常日志。') from error
    print('完整策略加载成功；视觉模块保持bfloat16，后续开始逐轨迹下载/提取。', flush=True)
    return encoder


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verified(directory):
    """完成标记最后写入；重启后必须重新验SHA，不能凭目录存在就跳过。"""
    directory = Path(directory)
    marker = directory / "COMPLETE.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    for name, sha in data["files"].items():
        if Path(name).name != name or not (directory / name).is_file() or digest(directory / name) != sha:
            raise RuntimeError(f"完成标记存在但文件不完整/哈希不符：{directory}/{name}；保留现场")
    return True


def publish(local, destination):
    """先复制到.partial，校验后改名；从不覆盖已发布的轨迹。"""
    local, destination = Path(local), Path(destination)
    if destination.exists():
        raise FileExistsError(f"目标已经存在，不覆盖：{destination}")
    partial = destination.with_name(destination.name + ".partial")
    # 同一任务的中断复制可续传；完成标记在最后一刻才生成。
    partial.mkdir(parents=True, exist_ok=True)
    for p in local.iterdir():
        if p.is_file():
            shutil.copy2(p, partial / p.name)
    files = {p.name: digest(p) for p in local.iterdir() if p.is_file()}
    for name, sha in files.items():
        if digest(partial / name) != sha:
            raise RuntimeError(f"复制校验失败：{partial / name}")
    # 不允许上次中断留下的额外文件进入有效缓存。
    if {p.name for p in partial.iterdir()} - set(files) - {"COMPLETE.json"}:
        raise RuntimeError(f"中间目录含额外文件，请先核对：{partial}")
    write_json(partial / "COMPLETE.json", {"files": files, "completed_utc": datetime.now(timezone.utc).isoformat()})
    partial.rename(destination)
    assert verified(destination)


def episode_rows(root, episode):
    import pyarrow.parquet as pq
    file_id, length, _ = EPISODES[episode]
    path = Path(root) / f"data/chunk-000/file-{file_id:03d}.parquet"
    table = pq.read_table(path)
    rows = sorted((r for r in table.to_pylist() if r["episode_index"] == episode), key=lambda r: r["frame_index"])
    if (len(rows) != length or [r["frame_index"] for r in rows] != list(range(length))
            or any(r["task_index"] != 24 or len(r["action"]) != 7 for r in rows)):
        raise RuntimeError(f"实际轨迹不符合固定版本协议：{episode}")
    return path, table, rows


def prepare(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    pending = []
    for ep in args.episodes:
        final = args.output_root / "raw" / f"episode_{ep}"
        if verified(final):
            provenance = json.loads((final / "provenance.json").read_text())
            if provenance.get("revision") != DATA_SHA or provenance.get("episode") != ep:
                raise RuntimeError("已有原始归档来源不同；请使用新output-root")
            print(f"已校验并跳过raw episode{ep}", flush=True)
        else:
            pending.append(ep)
    if not pending:
        print("原始轨迹已全部归档；不需要本会话保留下载缓存，也不重新下载", flush=True)
        return
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(DATA_ID, repo_type="dataset", revision=DATA_SHA, local_dir=args.dataset_root,
                          token=False, allow_patterns=["meta/**", *[
                              f"data/chunk-000/file-{EPISODES[e][0]:03d}.parquet" for e in pending]])
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    if float(info["fps"]) != 10.0:
        raise RuntimeError("数据FPS与固定数据版本不符")
    for ep in pending:
        final = args.output_root / "raw" / f"episode_{ep}"
        source, table, rows = episode_rows(args.dataset_root, ep)
        local = args.work_root / "raw" / f"episode_{ep}"
        local.mkdir(parents=True, exist_ok=True)
        selected = pa.Table.from_pylist(rows, schema=table.schema)
        pq.write_table(selected, local / "frames.parquet")
        for key in IMAGE_KEYS:
            with Image.open(io.BytesIO(rows[0][key]["bytes"])) as image:
                if image.size != (256, 256):
                    raise RuntimeError(f"意外的图像尺寸：{key} {image.size}")
                image.convert("RGB").save(local / (key.rsplit(".", 1)[-1] + "_frame0.png"))
        write_json(local / "provenance.json", {"dataset": DATA_ID, "revision": DATA_SHA,
                   "episode": ep, "split": EPISODES[ep][2], "num_frames": len(rows), "fps": info["fps"],
                   "source_file": source.name, "source_sha256": digest(source), "image_keys": IMAGE_KEYS,
                   "task": "pick up the alphabet soup and place it in the basket",
                   "scope": "three demonstration episodes for pipeline validation, not a trained corrector"})
        final.parent.mkdir(parents=True, exist_ok=True)
        publish(local, final)
        print(f"已归档原始episode{ep}，{len(rows)}帧，SHA校验通过：{final}", flush=True)


def ensure_source(source):
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", str(source)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin",
                        "https://github.com/ZJU-OmniAI/vla-corrector.git"], check=True)
    probe = subprocess.run(["git", "-C", str(source), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    if probe.returncode:
        subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", CORRECTOR_SHA], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    else:
        head = probe.stdout.strip()
    if head != CORRECTOR_SHA or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("参考仓库不是固定的干净版本；不会覆盖你的修改")


def extraction_device(device, allow_cpu):
    import torch
    if device == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU视觉提取需显式--allow-cpu-extraction；建议先复用已保存特征")
        return "cpu"
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但不可用；CPU需--device cpu --allow-cpu-extraction，不静默切换")
    return "cuda"


def extract(args):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    todo = []
    raw_root = getattr(args, "raw_root", None) or args.output_root / "raw"
    before_episode = getattr(args, "before_episode", None)
    on_complete = getattr(args, "on_complete", None)
    for ep in args.episodes:
        if verified(args.output_root / "features" / f"episode_{ep}"):
            meta = json.loads((args.output_root / "features" / f"episode_{ep}" / "metadata.json").read_text())
            if meta.get("policy_revision") != POLICY_SHA or meta.get("corrector_revision") != CORRECTOR_SHA:
                raise RuntimeError("已有特征缓存来源不一致，请使用新output-root")
        else:
            if before_episode is None and not verified(raw_root / f"episode_{ep}"):
                raise RuntimeError(f"先prepare原始episode{ep}")
            todo.append(ep)
    if not todo:
        print("全部特征缓存已完成并校验；未加载模型")
        return
    device = extraction_device(args.device, args.allow_cpu_extraction)
    setup_started = time.perf_counter()
    ensure_source(args.source_root)
    # 上游helper会调整sys.path；明确指定已安装0.6.1的位置，避免换成参考仓库的旧LeRobot。
    import lerobot
    os.environ["LEROBOT_SRC"] = str(Path(lerobot.__file__).resolve().parent.parent)
    sys.path.append(str(args.source_root / "src"))
    from siglip_dynamics.extract import SmolVLAExtractor, _quantize_per_token
    from lerobot.policies.factory import make_pre_post_processors
    # CPU试验默认只复用本机Hub缓存，不自动下载接近GB的权重。
    local_only = device == "cpu" and not args.allow_model_download
    snapshot = cached_policy_snapshot(allow_network=not local_only)
    encoder = cached_visual_extractor(SmolVLAExtractor, snapshot, device, allow_network=not local_only)
    encoder.policy.requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=encoder.policy.config, pretrained_path=snapshot,
                                     preprocessor_overrides={"device_processor": {"device": "cpu"}})
    camera_keys = list(encoder.policy.config.image_features)
    if len(camera_keys) != 2:
        raise RuntimeError(f"本协议要求两路真实相机，实际为{camera_keys}")
    setup_seconds = time.perf_counter() - setup_started
    for ep in todo:
        if before_episode is not None:
            before_episode(ep)
        raw = raw_root / f"episode_{ep}"
        if not verified(raw):
            raise RuntimeError(f"原始轨迹不完整：{raw}")
        provenance = json.loads((raw / "provenance.json").read_text())
        if provenance.get("revision") != DATA_SHA or provenance.get("episode") != ep:
            raise RuntimeError("原始轨迹来源不一致")
        began = time.perf_counter()
        rows = pq.read_table(raw / "frames.parquet").to_pylist()
        local = args.work_root / "features" / f"episode_{ep}"
        local.mkdir(parents=True, exist_ok=True)
        z_q = z_scale = None
        actions = []
        for i, row in enumerate(rows):
            images = {key: decode_rgb(row[key]["bytes"]) for key in IMAGE_KEYS}
            # 保留极少量参照帧，原始分片清理后仍能解释首个预测例子；不是预测器生成图。
            if i in {0,5,10}:
                for camera,key in enumerate(IMAGE_KEYS,1):
                    Image.fromarray(images[key]).save(local/f'reference_camera{camera}_frame{i:04d}.png')
            z = encoder.encode_item(images, image_key=IMAGE_KEYS[0], wrist_image_key=IMAGE_KEYS[1], use_wrist_image=True)
            if not np.isfinite(z).all():
                raise RuntimeError("视觉特征含NaN/Inf")
            if i == 0:
                n, (l_tokens, d_dim) = len(rows), z.shape
                if shutil.disk_usage(args.work_root).free < 2 * n * l_tokens * d_dim + 64 * 2**20:
                    raise RuntimeError("本地特征缓存磁盘余量不足")
                z_q = np.lib.format.open_memmap(local / "z_q.npy", mode="w+", dtype=np.int8, shape=(n, l_tokens, d_dim))
                z_scale = np.lib.format.open_memmap(local / "z_scale.npy", mode="w+", dtype=np.float16, shape=(n, l_tokens))
                print(f"真实视觉输出episode{ep}: {z.shape}；相机顺序{camera_keys}", flush=True)
            if z.shape != (l_tokens, d_dim):
                raise RuntimeError("同一轨迹特征维度变化")
            q, scale = _quantize_per_token(z)  # 完全使用固定上游量化函数
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise RuntimeError("float16量化scale无效；保留现场，不发布本轨迹")
            z_q[i], z_scale[i] = q, scale
            # 视觉采用上游提取器；动作采用当前checkpoint保存的processor，不照搬QUANTILES。
            batch = {dst: torch.from_numpy(images[src].copy()).permute(2, 0, 1).float() / 255
                     for src, dst in zip(IMAGE_KEYS, camera_keys, strict=True)}
            batch.update({"observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
                          "action": torch.tensor(row["action"], dtype=torch.float32),
                          "task": provenance["task"]})
            a = pre(batch)["action"].reshape(-1).detach().cpu().numpy()
            if a.shape != (7,) or not np.isfinite(a).all():
                raise RuntimeError(f"动作processor输出错误：{a.shape}")
            actions.append(a)
            if i % 25 == 0:
                print(f"episode{ep}: {i + 1}/{n}；本轨迹完成后立即归档", flush=True)
        z_q.flush(); z_scale.flush()
        cached_actions = np.asarray(actions, dtype=np.float16)  # 与上游默认缓存一致，Dataset读取时转float32
        if not np.isfinite(cached_actions).all():
            raise RuntimeError("动作转换float16后溢出，拒绝发布缓存")
        np.save(local / "actions.npy", cached_actions)
        np.save(local / "raw_actions.npy", np.asarray([r["action"] for r in rows], dtype=np.float32))
        np.save(local / "episode_index.npy", np.asarray([r["episode_index"] for r in rows], dtype=np.int32))
        np.save(local / "frame_index.npy", np.asarray([r["frame_index"] for r in rows], dtype=np.int64))
        np.save(local / "timestamp.npy", np.asarray([r["timestamp"] for r in rows], dtype=np.float64))
        write_json(local / "metadata.json", {"format": "siglip_frame_cache_v2", "num_frames": n,
                   "l_tokens": l_tokens, "d_dim": d_dim, "action_dim": 7, "split": provenance["split"],
                   "fps": 10.0, "policy_id": POLICY_ID, "policy_revision": POLICY_SHA,
                   "corrector_revision": CORRECTOR_SHA, "image_keys": IMAGE_KEYS, "camera_keys": camera_keys,
                   "dtype_actions": "float16", "dtype_z_q": "int8", "dtype_z_scale": "float16",
                   "encoder_info": encoder.backend_info(), "action_representation": "checkpoint_processor_normalized",
                   "extraction_device": device,
                   "action_normalization": "saved policy preprocessor; raw_actions.npy retained for audit",
                   "raw_archive_sha256": digest(raw / "COMPLETE.json"),
                   "raw_provenance": provenance,
                   "scope": "offline demonstration features; no predictor trained, no online frequency alignment claimed"})
        final = args.output_root / "features" / f"episode_{ep}"
        final.parent.mkdir(parents=True, exist_ok=True)
        extraction_seconds = time.perf_counter() - began
        publish_started = time.perf_counter()
        publish(local, final)
        print(f"特征episode{ep}已完成并归档：{final}", flush=True)
        if on_complete is not None:
            on_complete(ep, {"frames": n, "extraction_seconds": extraction_seconds,
                             "publish_seconds": time.perf_counter() - publish_started,
                             "encoder_setup_seconds": setup_seconds,
                             "feature_bytes": sum(p.stat().st_size for p in final.iterdir() if p.is_file())})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "extract", "verify"])
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path("outputs/source_audit/vla-corrector"))
    p.add_argument("--episodes", type=int, nargs="+", choices=list(EPISODES), default=list(EPISODES))
    p.add_argument("--download", action="store_true", help="只在prepare显式下载固定版本元信息/三份Parquet，不下载整个数据集")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--allow-cpu-extraction", action="store_true", help="显式允许慢速CPU视觉提取；尚未在本机完整验证")
    p.add_argument("--allow-model-download", action="store_true", help="CPU提取时允许缺少的模型权重下载；默认只查本机缓存")
    args = p.parse_args()
    args.output_root, args.work_root = args.output_root.absolute(), args.work_root.absolute()
    if str(args.output_root).startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Drive未挂载，拒绝将临时本地目录当作云盘")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    log = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "." + args.mode + ".log")
    # 同时保留实时输出和持久化日志；依赖Python标准库，无需额外日志框架。
    class Tee:
        def write(self, text):
            sys.__stdout__.write(text); file.write(text); file.flush()
            return len(text)
        def flush(self):
            sys.__stdout__.flush(); file.flush()
    with log.open("w") as file:
        previous = sys.stdout
        sys.stdout = Tee()
        try:
            print(f"阶段{args.mode}；Python={sys.executable}；仅三轨迹链路验证", flush=True)
            if args.mode == "prepare":
                prepare(args)
            elif args.mode == "extract":
                extract(args)
            else:
                for group in ("raw", "features"):
                    for ep in args.episodes:
                        print(group, ep, "COMPLETE" if verified(args.output_root / group / f"episode_{ep}") else "未完成")
        except BaseException:
            traceback.print_exc(file=file)
            raise
        finally:
            sys.stdout = previous
    print("完整日志：", log)


if __name__ == "__main__":
    main()
