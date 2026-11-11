"""CPU读取旧相机回合，生成成功率、错误/未完成分类与失败视频索引；不运行模型。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies.policy_generalization.corrector.batch_evaluation import make_plan, read_legacy_case, summarize, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records = []
    for directory in sorted(args.legacy_root.glob("n_action_steps_*/*/seed_*")):
        steps = int(directory.parents[1].name.removeprefix("n_action_steps_"))
        condition = directory.parent.name
        seed = int(directory.name.removeprefix("seed_"))
        case = make_plan([condition], [seed], {f"H{steps}": {"n_action_steps": steps}}, {})["cases"][0]
        record = {"case": case, "scope": "imported legacy evidence; no new evaluation run"}
        try:
            record.update(status="completed", result=read_legacy_case(directory))
        except (OSError, ValueError, KeyError) as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        records.append(record)
    if not records:
        raise RuntimeError("没有找到旧回合目录；不要把空目录当成0%成功率")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "imported_records.json", records)
    rows = summarize(records, args.output)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("旧结果只做重新统计，不是Corrector成绩；计划总数未给定，不从现存目录推测缺失回合。")
    print("失败视频索引：", args.output / "failure_videos.html")


if __name__ == "__main__":
    main()
