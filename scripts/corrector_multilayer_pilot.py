"""Object10 冻结SmolVLA多层视觉变化筛选：真实小样本，零训练，零隐式下载。
对2/5/8等中间层测试4x4与8x8池化、FP16/INT8变化保真度，核对既有Connector缓存。
只写独立小型JSON报告，不写全量特征，不改旧cache。需有原Parquet与冻结checkpoint。
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from corrector_multilayer import ReadOnlyVisionTap, mean_pool_tokens, quantize_per_token, dequantize

BASE = Path("/root/autodl-tmp/corrector")
POLICY_SHA = "a24f1f380461b665158d52acd140152c79e8b36d"
DATA_SHA = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("report exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    tmp.rename(path)


def compare_arrays(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1).astype(np.float64)
    b = np.asarray(b, dtype=np.float32).reshape(-1).astype(np.float64)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("mismatched/non-finite arrays")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    diff = float(np.linalg.norm(a-b))
    dot = float(np.dot(a, b))
    return {
        "relative_l2": diff / max(na, 1e-12),
        "cosine": dot / max(na*nb, 1e-12),
        "reference_l2": na,
        "absolute_rmse": diff / len(a)**0.5,
    }


def nearest_upsample(x, target_side=32):
    if target_side % x.shape[1]:
        raise ValueError("grid cannot be upsampled to target")
    n = target_side // x.shape[1]
    return np.repeat(np.repeat(x, n, axis=1), n, axis=2)


def evaluate_pair(previous, current, selected_layers, grids=(4, 8, 16)):
    """Changes are measured over k=10 data frames; no failure-classification labels."""
    result = {}
    for layer in selected_layers:
        a, b = previous[layer], current[layer]
        delta_full = (b-a).reshape(2, 32, 32, -1)
        metrics = {}
        for grid in grids:
            xa, xb = mean_pool_tokens(a, grid), mean_pool_tokens(b, grid)
            delta = xb-xa
            pool_err = compare_arrays(delta_full, nearest_upsample(delta))
            fp_a = xa.astype(np.float16).astype(np.float32)
            fp_b = xb.astype(np.float16).astype(np.float32)
            fp_err = compare_arrays(delta, fp_b-fp_a)
            qa, sa = quantize_per_token(xa)
            qb, sb = quantize_per_token(xb)
            q_err = compare_arrays(delta, dequantize(qb,sb)-dequantize(qa,sa))
            metrics[str(grid)] = {
                "pooled_vs_full_delta": pool_err,
                "fp16_delta": fp_err,
                "int8_delta": q_err,
                "bytes_per_frame_fp16": int(2*grid*grid*a.shape[-1]*2),
                "bytes_per_frame_int8_scales": int(2*grid*grid*(a.shape[-1]+2)),
            }
        result[layer] = metrics
    return result


def summarize_pairs(pairs):
    summary = {}
    if not pairs:
        return summary
    for layer in pairs[0]["metrics"]:
        summary[layer] = {}
        for grid in sorted(pairs[0]["metrics"][layer], key=int):
            summary[layer][grid] = {}
            for kind in ("pooled_vs_full_delta", "fp16_delta", "int8_delta"):
                arr = [r["metrics"][layer][grid][kind]["relative_l2"] for r in pairs]
                summary[layer][grid][kind] = {
                    "pairs":len(arr), "median_relative_l2":float(np.median(arr)),
                    "p90_relative_l2":float(np.percentile(arr,90)),
                    "max_relative_l2":float(max(arr)),
                }
            summary[layer][grid]["bytes_per_frame_fp16"] = pairs[0]["metrics"][layer][grid]["bytes_per_frame_fp16"]
            summary[layer][grid]["bytes_per_frame_int8_scales"] = pairs[0]["metrics"][layer][grid]["bytes_per_frame_int8_scales"]
    return summary


def locate_parquet(data_root, episode, manifest):
    rec = next((r for r in manifest["records"] if r["episode"] == episode), None)
    if rec is None or len(rec["files"]) != 1:
        raise ValueError("episode missing from immutable manifest or spans multiple shards")
    p = data_root / rec["files"][0]["path"]
    if not p.is_file():
        raise FileNotFoundError(f"missing source Parquet {p}; no implicit download")
    return p, rec


def deep_cache(root, episode, record):
    from corrector_data_workbench import verified
    d = root / f"episode_{episode}"
    if not verified(d):
        raise RuntimeError("old deep cache not verified: "+str(d))
    meta = json.loads((d/"metadata.json").read_text())
    if meta.get("policy_revision") != POLICY_SHA or meta.get("action_representation") != "checkpoint_processor_normalized":
        raise RuntimeError("old deep feature/action protocol mismatch")
    if (meta.get("num_frames"),meta.get("fps"),meta.get("l_tokens"),meta.get("d_dim")) != (record["frames"],10.,128,960):
        raise RuntimeError("old deep cache shape/rate mismatch")
    fi = np.load(d/"frame_index.npy",mmap_mode="r")
    ts = np.load(d/"timestamp.npy",mmap_mode="r")
    q = np.load(d/"z_q.npy",mmap_mode="r")
    scale = np.load(d/"z_scale.npy",mmap_mode="r")
    if (fi.shape != (record["frames"],) or q.shape != (record["frames"],128,960)
        or scale.shape != (record["frames"],128) or not np.array_equal(fi,np.arange(record["frames"]))
        or not np.allclose(np.diff(ts),0.1,atol=1e-4,rtol=1e-4)):
        raise RuntimeError("old deep cache frame/timebase mismatch")
    return q,scale,ts


def run(args):
    # Fail before importing the huge model or writing files.
    if args.output.exists():
        raise FileExistsError("use a new output directory; old reports must not be overwritten")
    if not (1<=len(args.episodes)<=6 and 1<=len(args.layers)<=4):
        raise ValueError("pilot is limited to at most 6 episodes/4 block outputs")
    if len(set(args.episodes)) != len(args.episodes) or len(set(args.layers)) != len(args.layers):
        raise ValueError("duplicate episode/layer")
    if shutil.disk_usage(args.output.parent).free < args.min_free_gib * 2**30:
        raise RuntimeError("destination disk below 5GiB free reserve")
    os.environ.update(HF_HOME=str(args.runtime/"hf"),HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",
                      LEROBOT_SRC=str(args.runtime/"lerobot/src"))
    sys.path.insert(0,str(args.source_root/"src"))
    sys.path.insert(0,str(args.runtime/"lerobot/src"))
    import torch
    import pyarrow.parquet as pq
    from corrector_data_workbench import cached_policy_snapshot,cached_visual_extractor,decode_rgb
    from siglip_dynamics.extract import SmolVLAExtractor
    if not torch.cuda.is_available():
        raise RuntimeError("real pilot needs CUDA")
    import subprocess
    processes=subprocess.check_output(
        ["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    if processes:
        raise RuntimeError("GPU is busy, refusing to compete: "+processes[:100])
    manifest=json.loads(args.manifest.read_text())
    if manifest.get("revision")!=DATA_SHA or manifest.get("fps")!=10:
        raise RuntimeError("unexpected dataset revision/fps")
    sources=[]
    for episode in args.episodes:
        p,rec=locate_parquet(args.data_root,episode,manifest)
        deep=deep_cache(args.deep_root,episode,rec)
        sources.append((episode,p,rec,deep))
    snapshot=Path(cached_policy_snapshot(False))
    weights=snapshot/"model.safetensors"
    weight_sha=sha256(weights)
    encoder=cached_visual_extractor(SmolVLAExtractor,snapshot,"cuda",allow_network=False)
    encoder.policy.eval().requires_grad_(False)
    wrapper=encoder.model.vlm_with_expert
    vision=wrapper.get_vlm_model().vision_model
    if len(vision.encoder.layers)!=12 or max(args.layers)>=12:
        raise RuntimeError("unexpected frozen vision block count")
    cameras=list(encoder.policy.config.image_features)
    if len(cameras)!=2:
        raise RuntimeError("expected exactly two cameras")
    torch.set_num_threads(4)
    results=[]
    begin=time.perf_counter()
    args.output.mkdir(parents=True)
    with torch.inference_mode(), ReadOnlyVisionTap(wrapper,tuple(args.layers)) as tap:
        for ep,parquet,rec,(old_q,old_scale,old_time) in sources:
            table=pq.read_table(parquet,columns=["episode_index","frame_index","timestamp",*IMAGE_KEYS])
            rows=sorted((r for r in table.to_pylist() if r["episode_index"]==ep),key=lambda r:r["frame_index"])
            if len(rows)!=rec["frames"] or [r["frame_index"] for r in rows]!=list(range(rec["frames"])):
                raise RuntimeError(f"source episode {ep} is incomplete")
            history=deque(maxlen=10)
            pairs=[]
            old_diff=[]
            old_quantized_match=[]
            old_q_element_match=[]
            old_scale_element_match=[]
            frame_count=0
            last_frame=-1
            for row in rows:
                fi=int(row["frame_index"])
                if args.max_frames and fi>=args.max_frames:
                    break
                if fi!=last_frame+1 or abs(float(row["timestamp"])-float(old_time[fi]))>1e-4:
                    raise RuntimeError("new frame and old cache timebase misaligned")
                last_frame=fi
                views={}
                for camera_key,image_key in zip(cameras,IMAGE_KEYS,strict=True):
                    raw=decode_rgb(row[image_key]["bytes"])
                    inp=encoder._prepare_image(raw)
                    with tap.capture(camera_key) as captured:
                        connector=wrapper.embed_image(inp)
                    if not torch.equal(connector,captured["connector"]):
                        raise RuntimeError("capture changed frozen Connector output")
                    views[camera_key]={name:t[0].float().cpu().numpy().copy() for name,t in captured.items()}
                feature={name:np.stack([views[c][name] for c in cameras]) for name in views[cameras[0]]}
                old=dequantize(old_q[fi],old_scale[fi])
                deep=feature["connector"].reshape(128,960)
                old_diff.append(compare_arrays(deep,old))
                check_q, check_scale = quantize_per_token(deep)
                q_equal = check_q == old_q[fi]
                scale_equal = check_scale == old_scale[fi]
                old_quantized_match.append(bool(q_equal.all() and scale_equal.all()))
                old_q_element_match.append(float(np.mean(q_equal)))
                old_scale_element_match.append(float(np.mean(scale_equal)))
                current={name:feature[name] for name in feature if name.startswith("block_")}
                if len(history)==10:
                    pairs.append({"episode":ep,"frame_index":fi-10,"future_frame_index":fi,
                                  "metrics":evaluate_pair(history[0],current,current,args.grids)})
                history.append(current)
                frame_count+=1
                if frame_count%50==0:
                    print(f"episode {ep}: {frame_count}/{rec['frames']} frames",flush=True)
            record={
                "episode":ep,"task_id":rec.get("task_id"),"frames":frame_count,"pairs":len(pairs),
                "source_parquet":parquet.name,"source_sha256":sha256(parquet),
                "old_deep_median_relative_l2":float(np.median([x["relative_l2"] for x in old_diff])),
                "old_deep_max_relative_l2":max(x["relative_l2"] for x in old_diff),
                "old_deep_min_cosine":min(x["cosine"] for x in old_diff),
                "old_deep_exact_requantized_frames":int(sum(old_quantized_match)),
                "old_deep_exact_requantized_frame_fraction":float(np.mean(old_quantized_match)),
                "old_deep_q_element_agreement_fraction":float(np.mean(old_q_element_match)),
                "old_deep_scale_element_agreement_fraction":float(np.mean(old_scale_element_match)),
                "layer_metrics":summarize_pairs(pairs),
            }
            # This is an intentionally generous *sanity* threshold, not a scientific criterion.
            # Mismatch can arise from image order, preprocessing or model/source drift.
            if record["old_deep_max_relative_l2"]>0.06 or record["old_deep_min_cosine"]<0.997:
                raise RuntimeError(f"new Connector incompatible with old cached features, ep {ep}: {record}")
            results.append(record)
            print(json.dumps({"episode":ep,"frames":frame_count,"pairs":len(pairs),
                              "deep_median_relative_l2":record["old_deep_median_relative_l2"]},ensure_ascii=False),flush=True)
    if sha256(weights)!=weight_sha:
        raise RuntimeError("original model weights changed")
    report={
        "schema":"multilayer_select_pilot_v1",
        "scope":"Small real demonstration screening; no training, no closed-loop/OGG claims",
        "policy_sha":POLICY_SHA,"dataset_sha":DATA_SHA,"weights_sha256":weight_sha,
        "layer_indices_zero_based":args.layers,"grids":args.grids,
        "prediction_k_data_frames":10,"fps":10,
        "elapsed_seconds":time.perf_counter()-begin,"episodes":results,
        "limitations":[
            "Pooling-vs-full delta measures spatial information loss, not predictor generalization",
            "INT8 errors are measured on changes of pooled features, not classification accuracy",
            "Deep cache comparison is sensitive to the old per-token quantization protocol",
            "No failure labels, online interruption precision, or OGG gradients evaluated",
            "No feature tensors or old checkpoint files were written",
        ],
    }
    atomic_json(args.output/"report.json",report)
    print("COMPLETE REPORT "+str(args.output/"report.json"),flush=True)


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--runtime",type=Path,default=BASE/"runtime")
    p.add_argument("--source-root",type=Path,default=BASE/"runtime/vla-corrector-reference")
    p.add_argument("--data-root",type=Path,default=BASE/"object10-work/source")
    p.add_argument("--deep-root",type=Path,default=BASE/"object10/features")
    p.add_argument("--manifest",type=Path,default=BASE/"object10/manifest.json")
    p.add_argument("--episodes",nargs="+",type=int,default=[813])
    p.add_argument("--layers",nargs="+",type=int,default=[2,5,8])
    p.add_argument("--grids",nargs="+",type=int,default=[4,8,16],help="allowed grids 4,8,16,32; screen only")
    p.add_argument("--max-frames",type=int,default=0,help="0=entire episode; 11+=first N frames")
    p.add_argument("--min-free-gib",type=float,default=5)
    a=p.parse_args()
    if len(set(a.grids))!=len(a.grids) or not a.grids or any(g not in (4,8,16,32) for g in a.grids):
        p.error("--grids must be a unique nonempty subset of 4 8 16 32")
    if a.max_frames and a.max_frames<11:
        p.error("--max-frames must be 0 or at least 11")
    if a.min_free_gib<5:
        p.error("--min-free-gib must remain >=5")
    return a

if __name__=="__main__":
    run(args_parser())
