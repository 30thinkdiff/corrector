"""CPU核对真实三条示范的时序索引；不加载VLA，不下载数据，不训练。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.policy_generalization.corrector.transitions import make_transition_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect-only", action="store_true", help="只读实际数据，不调用用户配对函数")
    p.add_argument("--dataset-root", type=Path, default=ROOT / "outputs/training_audit_20261105/raw_dataset")
    args = p.parse_args()
    import pyarrow.parquet as pq
    episodes, frames, lengths = [], [], {}
    for episode, file_id, length in [(813, 213, 155), (818, 214, 164), (825, 215, 153)]:
        path = args.dataset_root / f"data/chunk-000/file-{file_id:03d}.parquet"
        rows = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"]).to_pylist()
        rows = sorted([r for r in rows if r["episode_index"] == episode], key=lambda r: r["frame_index"])
        if (len(rows) != length or [r["frame_index"] for r in rows] != list(range(length))
                or any(r["task_index"] != 24 for r in rows)):
            raise RuntimeError(f"固定版本的实际轨迹不符合预期：{episode}")
        lengths[episode] = length
        episodes.extend(r["episode_index"] for r in rows)
        frames.extend(r["frame_index"] for r in rows)
    report = {"scope": "real frame index alignment; no visual predictor trained",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "pair_source_sha256": hashlib.sha256(Path(make_transition_pairs.__code__.co_filename).read_bytes()).hexdigest(),
              "episode_lengths": lengths, "rows": len(frames),
              "k1_expected_train_pairs": 317, "k1_expected_validation_pairs": 152,
              "boundary_examples": {"valid_last_813": [153, 154], "invalid_cross_episode": [154, 155]},
              "user_function_called": not args.inspect_only}
    if not args.inspect_only:
        for k in (1, 2):
            actual = make_transition_pairs(episodes, frames, k_step=k)
            expected_count = sum(max(length-k, 0) for length in lengths.values())
            if not isinstance(actual, list) or len(actual) != expected_count:
                raise AssertionError(f"k={k}应返回{expected_count}对组成的list。")
            if len(set(actual)) != len(actual):
                raise AssertionError("存在重复的时序对。")
            for i, j in actual:
                if not (0 <= i < j < len(frames) and episodes[i] == episodes[j] and frames[j]-frames[i] == k):
                    raise AssertionError(f"无效时序对：{i,j}；请检查轨迹边界和frame差值。")
        for bad_k in (0, -1):
            try:
                make_transition_pairs(episodes, frames, k_step=bad_k)
            except (TypeError, ValueError):
                continue
            raise AssertionError("k_step非正数应明确拒绝。")
        report["user_pairs_verified"] = True
    destination = ROOT / "outputs/corrector_audit_20261105"
    destination.mkdir(parents=True, exist_ok=True)
    name = "real_pair_inputs.json" if args.inspect_only else "user_pair_verification.json"
    (destination / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("已保存：", destination / name)


if __name__ == "__main__":
    main()
