"""读取冻结验证索引，评估Corrector视觉残差；plot阶段不依赖Torch或VLA。"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np


def prediction_metrics(pred, target, mean):
    """每个样本独立归约；与旧版单样本报告的展平cosine/MSE定义一致。"""
    import torch.nn.functional as F
    p, t = pred.flatten(1), target.flatten(1)
    m = mean.reshape(1, -1).expand_as(t)
    return {"cosine": F.cosine_similarity(p, t, dim=1),
            "mse": (p-t).square().mean(1), "zero_mse": t.square().mean(1),
            "train_mean_mse": (m-t).square().mean(1),
            "train_mean_cosine": F.cosine_similarity(m, t, dim=1),
            "pred_norm": p.norm(dim=1), "target_norm": t.norm(dim=1)}


def predict(args):
    import torch
    from torch.utils.data import DataLoader
    import corrector_train as train

    started = time.perf_counter()
    torch.set_num_threads(getattr(args, "cpu_threads", 2))
    device = torch.device(getattr(args, "device", "cpu"))
    batch_size, workers = getattr(args, "batch_size", 8), getattr(args, "loader_workers", 0)
    if batch_size < 1 or workers < 0 or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError("报告batch/workers无效或请求的CUDA不可用")
    selected = getattr(args, "checkpoint", None)
    if selected is None and not (args.run / "result.json").is_file():
        raise RuntimeError("需要已完成并重载验证的训练运行；不读取正在写入的best权重")
    protocol = json.loads((args.run / "protocol.json").read_text())
    config, dataset_cls, model_cls, _, hashes = train.reference_api(args.source_root)
    if hashes != protocol["source_files"]:
        raise ValueError("参考源码与训练时不同")
    groups, records, _ = train.datasets(args.features, None, dataset_cls, protocol["k_step"])
    if records != protocol["records"]:
        raise ValueError("特征归档与训练记录不同")
    split = protocol["split"]
    groups, actual, indices = train.select_split(groups, split["mode"], split["val_ratio"] or .1, protocol["seed"])
    if actual != split:
        raise ValueError("验证划分与冻结协议不同")
    frozen = np.load(args.run / "split_indices.npz")
    if any(not np.array_equal(frozen[k], indices[k]) for k in indices):
        raise ValueError("冻结索引文件与协议不一致")
    cfg = dict(protocol["model"])
    cfg["scale"] = config.ModelScale(cfg["scale"])
    cfg["custom_widths"] = tuple(cfg["custom_widths"])
    model = model_cls(config.SiglipMLPConfig(**cfg))
    checkpoint_path = selected if selected is not None else args.run / "best_model.pt"
    checkpoint_sha = train.archive.digest(checkpoint_path)
    previous = args.output / "summary.json"
    if previous.exists() and json.loads(previous.read_text())["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("报告目录已有另一份权重结果；为每个checkpoint指定独立output")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("feature_protocol") != protocol or checkpoint.get("model_cfg") != protocol["model"]:
        raise ValueError("checkpoint来源与本运行协议不一致")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    # 均值基线只从训练样本估计，不用验证目标拟合基线。
    mean = torch.zeros_like(groups["train"][0]["delta_z"], dtype=torch.float64)
    def loader(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                          pin_memory=device.type == 'cuda')
    for item in loader(groups["train"]):
        mean += item["delta_z"].double().sum(0)
    mean = (mean / len(groups["train"])).float().to(device)
    rows = []
    validation = groups["validation"]
    master = validation.dataset
    frame_arrays = {str(ds.data_path): np.load(ds.data_path / "frame_index.npy", mmap_mode="r")
                    for ds in master.datasets}
    task_by_episode = {int(ds.ep_idx[0]): json.loads((ds.data_path/'metadata.json').read_text())
                       .get('raw_provenance', {}).get('task_id') for ds in master.datasets}
    with torch.no_grad():
        i = 0
        for batch in loader(validation):
            pred = model(batch["z_t"].to(device), batch["a_t"].to(device))
            target = batch["delta_z"].to(device)
            metrics = {k: v.cpu().tolist() for k, v in prediction_metrics(pred, target, mean).items()}
            if i == 0:
                np.savez_compressed(args.output / "first_pair.npz", z_t=batch["z_t"][0].numpy(),
                                    predicted_delta=pred[0].cpu().numpy(),
                                    target_delta=batch["delta_z"][0].numpy(), action=batch["a_t"][0].numpy())
            for j in range(len(pred)):
                row = validation_row(validation, frame_arrays, i)
                row['task_id'] = task_by_episode[row['episode']]
                row.update({k: values[j] for k, values in metrics.items()})
                rows.append(row)
                i += 1
            if i == len(pred) or i == len(validation) or i % (batch_size * 100) == 0:
                print(f"预测报告：{i}/{len(validation)}对，device={device}", flush=True)
    with (args.output / "per_pair.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    numeric = ("cosine", "mse", "zero_mse", "train_mean_mse", "train_mean_cosine", "pred_norm", "target_norm")
    per_task = []
    for task in sorted({r['task_id'] for r in rows if r['task_id'] is not None}):
        subset = [r for r in rows if r['task_id'] == task]
        per_task.append({'task_id': task, 'pairs': len(subset),
                         **{key: float(np.mean([r[key] for r in subset])) for key in numeric}})
    if per_task:
        with (args.output/'per_task.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(per_task[0]))
            writer.writeheader(); writer.writerows(per_task)
    summary = {"run": str(args.run), "parameters": sum(p.numel() for p in model.parameters()),
               "checkpoint_sha256": checkpoint_sha, "checkpoint": str(checkpoint_path),
               "checkpoint_epoch": checkpoint.get("epoch"), "optimizer_step": checkpoint.get("step"),
               "split": split, "sample_count": len(rows),
               "report_runtime": {"device": str(device), "batch_size": batch_size, "loader_workers": workers,
                                  "seconds": time.perf_counter()-started},
               "metrics": {key: float(np.mean([r[key] for r in rows])) for key in numeric},
               "per_task": per_task,
               "task_macro_metrics": {key: float(np.mean([r[key] for r in per_task])) for key in numeric}
                                     if per_task else None,
               "first_pair": {k: rows[0][k] for k in ("episode", "frame", "future_frame")},
               "first_pair_selection": "first frozen validation index, not selected for prediction quality",
               "fps": protocol["fps"], "k_step": protocol["k_step"],
               "scope": "latent residual regression; no reconstructed future RGB, alarm or robot success-rate claim"}
    train.archive.write_json(args.output / "summary.json", summary)
    if args.raw_root:
        import io
        import pyarrow.parquet as pq
        from PIL import Image
        ep, t, future = (summary["first_pair"][k] for k in ("episode", "frame", "future_frame"))
        path = args.raw_root / f"episode_{ep}/frames.parquet"
        if not path.is_file():
            raise FileNotFoundError("预测已保存，但可视化所需原始轨迹不存在")
        images = pq.read_table(path, columns=["frame_index", *train.archive.IMAGE_KEYS]).to_pylist()
        for label, idx in (("t", t), ("future", future)):
            row = next(r for r in images if r["frame_index"] == idx)
            for camera, key in enumerate(train.archive.IMAGE_KEYS, 1):
                Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB").save(args.output / f"real_{label}_camera{camera}.png")
    else:
        ep,t,future=(summary['first_pair'][k] for k in ('episode','frame','future_frame'))
        references=[(args.features/f'episode_{ep}'/f'reference_camera{camera}_frame{frame:04d}.png',
                     args.output/f'real_{label}_camera{camera}.png')
                    for label,frame in [('t',t),('future',future)] for camera in (1,2)]
        if all(src.is_file() for src,_ in references):
            for src,dst in references:shutil.copy2(src,dst)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validation_row(validation, frame_arrays, i):
    master = validation.dataset
    index = int(validation.indices[i])
    part = bisect.bisect_right(master.cumulative_sizes, index)
    local_index = index - (master.cumulative_sizes[part - 1] if part else 0)
    ds = master.datasets[part]
    t, future = ds.pairs[local_index]
    frame = frame_arrays[str(ds.data_path)]
    return {"episode": int(ds.ep_idx[t]), "frame": int(frame[t]), "future_frame": int(frame[future])}


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    root = args.output
    summary = json.loads((root / "summary.json").read_text())
    values = np.load(root / "first_pair.npz")
    pred, target = values["predicted_delta"], values["target_delta"]
    with (root / "per_pair.csv").open() as file:
        rows = list(csv.DictReader(file))
    magnitude = np.stack([np.sqrt((target**2).mean(-1)), np.sqrt((pred**2).mean(-1))])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    heat = axes[0].imshow(magnitude, aspect="auto", vmin=0)
    axes[0].set_yticks([0, 1], ["True delta", "Predicted delta"])
    axes[0].set_xlabel("Visual token index (not pixel position)")
    axes[0].set_title("Per-token delta RMS; shared color scale")
    fig.colorbar(heat, ax=axes[0], fraction=.05)
    selected = np.linspace(0, pred.size - 1, 128, dtype=int)
    axes[1].plot(selected, target.reshape(-1)[selected], label="True", alpha=.8)
    axes[1].plot(selected, pred.reshape(-1)[selected], label="Predicted", alpha=.8)
    axes[1].set_title("128 fixed feature coordinates")
    axes[1].set_xlabel("Flattened latent coordinate"); axes[1].legend()
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    for ep in sorted({int(r["episode"]) for r in rows}):
        subset = sorted((r for r in rows if int(r["episode"]) == ep), key=lambda r: int(r["frame"]))
        axes[2].scatter([int(r["frame"]) for r in subset], [float(r["cosine"]) for r in subset], s=12, label=f"ep{ep}")
    axes[2].set_title("Validation residual cosine")
    axes[2].axhline(summary["metrics"]["train_mean_cosine"], linestyle="--", color="gray", label="Train-mean baseline")
    axes[2].set_ylim(-1.05, 1.05); axes[2].set_xlabel("Recorded frame")
    if len({r['episode'] for r in rows}) <= 10:
        axes[2].legend()
    fig.suptitle(f"{summary['parameters']:,} parameters / {summary['split']['mode']} / {summary['sample_count']} pairs")
    fig.tight_layout(); fig.savefig(root / "latent_prediction.png", dpi=150); plt.close(fig)
    if summary.get('per_task'):
        task_rows = summary['per_task']
        x = np.arange(len(task_rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(x-.18, [r['cosine'] for r in task_rows], .36, label='Predictor')
        axes[0].bar(x+.18, [r['train_mean_cosine'] for r in task_rows], .36, label='Train-mean')
        axes[0].set_title('Validation residual cosine by task')
        axes[1].bar(x-.25, [r['mse'] for r in task_rows], .25, label='Predictor')
        axes[1].bar(x, [r['zero_mse'] for r in task_rows], .25, label='Zero')
        axes[1].bar(x+.25, [r['train_mean_mse'] for r in task_rows], .25, label='Train-mean')
        axes[1].set_title('Validation residual MSE by task')
        for ax in axes:
            ax.set_xticks(x, [str(r['task_id']) for r in task_rows]); ax.set_xlabel('Dataset task ID'); ax.legend()
        fig.tight_layout(); fig.savefig(root/'per_task_metrics.png', dpi=150); plt.close(fig)
    if (root / "real_t_camera1.png").is_file():
        fig, axes = plt.subplots(2, 2, figsize=(7, 7))
        for camera in (1, 2):
            for column, label in enumerate(("t", "future")):
                axes[camera - 1, column].imshow(plt.imread(root / f"real_{label}_camera{camera}.png"))
                axes[camera - 1, column].axis("off")
                axes[camera - 1, column].set_title(f"Camera{camera}: recorded {label}")
        pair = summary["first_pair"]
        fig.suptitle(f"Recorded ep{pair['episode']} frames {pair['frame']} and {pair['future_frame']}\nReference images, not generated pixels")
        fig.tight_layout(); fig.savefig(root / "real_observations.png", dpi=140); plt.close(fig)
    print("Saved plots:", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["predict", "plot"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="显式评估已发布的中间权重；省略时评估best_model.pt")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "predict":
        if any(x is None for x in (args.run, args.features, args.source_root)):
            parser.error("predict需要--run、--features、--source-root")
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "summary.json").exists():
            raise FileExistsError("预测报告已存在；使用新output避免覆盖")
        predict(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
