"""只下载公开数据集清单和info.json，记录全量字节数；不下载训练分片/视频/权重。"""
import argparse
import json
from pathlib import Path

REVISIONS = {
    "HuggingFaceVLA/libero": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
    "lerobot/libero_plus": "f3f49f426d75030177b18778374005bc12ccd588",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-cache", type=Path, help="已完成pilot的features/episode_x目录，用实测L/D估算全库缓存")
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    args.output.mkdir(parents=True, exist_ok=True)
    for repo, revision in REVISIONS.items():
        info = HfApi().dataset_info(repo, revision=revision, files_metadata=True, token=False)
        if any(f.size is None for f in info.siblings):
            raise RuntimeError("部分文件大小未知，不能把不完整求和当全量预算")
        files = [{"path": f.rfilename, "bytes": f.size} for f in info.siblings]
        local = hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision=revision,
                                local_dir=args.output / "metadata" / repo.replace("/", "--"), token=False)
        meta = json.loads(Path(local).read_text())
        report = {"repo_id": repo, "revision": revision, "raw_bytes": sum(f["bytes"] for f in files),
                  "files": files, "fps": meta["fps"], "episodes": meta["total_episodes"],
                  "frames": meta["total_frames"], "features": meta["features"]}
        if args.pilot_cache:
            if not (args.pilot_cache / "COMPLETE.json").is_file():
                raise RuntimeError("pilot尚未完成，不能把预期维度当实测预算")
            pilot = json.loads((args.pilot_cache / "metadata.json").read_text())
            l, d, n = pilot["l_tokens"], pilot["d_dim"], report["frames"]
            report["estimated_quantized_cache_bytes"] = n * (l * d + l * 2 + 7 * 2 + 7 * 4 + 4 + 8 + 8)
            report["estimate_note"] = "仅按同一冻结编码器、同分辨率/相机数的pilot实测L/D估算；不含备份、临时副本、文件头或训练状态"
        (args.output / (repo.replace("/", "--") + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(repo, f"{report['raw_bytes'] / 2**30:.3f}GiB", report["episodes"], "episodes", report["frames"], "frames", report["fps"], "FPS")
        if "estimated_quantized_cache_bytes" in report:
            print("全库特征缓存估算：", round(report["estimated_quantized_cache_bytes"] / 2**30, 2), "GiB")


if __name__ == "__main__":
    main()
