"""可中断的顺序评测与CPU汇总；具体策略执行由已注册callback提供，不伪造Corrector接入。"""
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
import math
from pathlib import Path
import shutil
import time


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_recording_fps(env, fps):
    """只改当前向量环境/子环境实例的录像元信息，不改物理控制频率和安装包源码。"""
    if type(fps) is not int or fps <= 0:
        raise ValueError("录像FPS须为正整数")
    for item in [env, getattr(env, "unwrapped", env), *getattr(env, "envs", [])]:
        item.metadata = {**item.metadata, "render_fps": fps}


def make_plan(conditions, seeds, methods, protocol):
    """冻结条件×seed×方法的完整集合；小集合验证后再扩大，不自动启动GPU。"""
    if not conditions or not seeds or not methods or len(seeds) != len(set(seeds)):
        raise ValueError("条件/seed/方法不能为空，seed不可重复")
    if len(conditions) != len(set(conditions)):
        raise ValueError("相机条件不可重复，避免重复计数")
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seed须为非负整数")
    cases = []
    for condition in conditions:
        for seed in seeds:
            for method, config in methods.items():
                case = {"condition": condition, "seed": seed, "method": method, "config": config}
                case["id"] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]
                cases.append(case)
    return json.loads(json.dumps({"protocol": protocol, "cases": cases}))


def read_legacy_case(directory):
    """读取旧相机实验的一回合真实结果；不使用pc_success阈值反推真假。"""
    directory = Path(directory)
    info = json.loads((directory / "eval_info.json").read_text())
    timing = json.loads((directory / "timing.json").read_text())
    schema = json.loads((directory / "grasp_schema.json").read_text())
    values = [s for task in info["per_task"] for s in task["metrics"]["successes"]]
    if len(values) != 1 or type(values[0]) is not bool:
        raise ValueError("需要一回合明确的bool success，不能将聚合百分数当逐回合记录")
    if timing.get("evaluation_returned") is not True or schema.get("status") != "evaluation_returned":
        raise ValueError("旧回合中断/未完整返回，不计入完成率")
    videos = sorted((directory / "videos").rglob("*.mp4"))
    if len(videos) != 1 or videos[0].stat().st_size == 0:
        raise ValueError("需要恰好一段非空的原始执行视频，不混入派生慢放视频")
    return {"success": values[0], "env_steps": timing["env_steps"],
            "chunk_calls": timing["chunk_calls"], "policy_s": timing["chunk_wall_s_total"],
            "monitor_s": None, "event_replans": None, "video_paths": [str(videos[0].resolve())],
            "scope": "legacy fixed-window rollout; no Corrector used"}


def run_plan(root, plan, callbacks, *, max_seconds=0., min_free_gib=0., stop_file=None):
    """callback(case, attempt_dir)→真实结果dict；每次只运行一回合并立即持久化。

    attempt_dir位于root下，建议root是已挂载Drive。callback将日志、MP4、输入审计直接写该目录。
    标准输出不捕获为success；必须明确返回bool并有非空视频。断连的running/error不算机器人失败。
    """
    root = Path(root).absolute()
    if not all(math.isfinite(x) and x >= 0 for x in (max_seconds,min_free_gib)):
        raise ValueError('时间预算与磁盘下限须为非负有限数；时间0为不限时')
    if str(root).startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Drive未挂载，拒绝把本地目录当作持久化评测")
    unknown = {c["method"] for c in plan["cases"]} - set(callbacks)
    if unknown:
        raise RuntimeError(f"这些方法尚未接入，启动前停止：{sorted(unknown)}")
    for c in plan["cases"]:
        if c["config"].get("uses_corrector") and not c["config"].get("corrector_sha256"):
            raise RuntimeError("Corrector方法必须先固定已训练checkpoint的SHA，不能用未训练模块做正式比较")
    frozen = root / "plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise RuntimeError("已有冻结计划不同，请换实验目录；不覆盖已完成记录")
    write_json(frozen, plan)
    began = time.perf_counter()
    stop_file = Path(stop_file) if stop_file is not None else root/'STOP'
    execution = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
                 'max_seconds':max_seconds,'min_free_gib':min_free_gib,'stop_file':str(stop_file)}
    write_json(root/'execution.json',execution)
    for case in plan["cases"]:
        status_file = root / "trials" / f"{case['id']}.json"
        old = json.loads(status_file.read_text()) if status_file.exists() else {}
        if old.get("status") == "completed":
            for artifact in old["video_artifacts"]:
                p = root / artifact["path"]
                if not p.is_file() or hash_file(p) != artifact["sha256"]:
                    raise RuntimeError(f"已完成回合视频缺失或损坏：{p}")
            continue
        reason = ('operator_stop_file' if stop_file.exists() else
                  'time_budget' if max_seconds and time.perf_counter()-began >= max_seconds else
                  'disk_space' if min_free_gib and shutil.disk_usage(root).free < min_free_gib*2**30 else None)
        if reason:
            execution.update(status='paused',reason=reason)
            print(f'在回合边界暂停：{reason}；已完成结果保留，同计划可续跑。',flush=True)
            break
        attempt = int(old.get("attempt", 0)) + 1
        directory = root / "attempts" / case["id"] / f"attempt_{attempt:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        record = {"case": case, "status": "running", "attempt": attempt,
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(status_file, record)
        start = time.perf_counter()
        try:
            result = callbacks[case["method"]](case, directory)
            video_policy=case['config'].get('video_policy','all')
            if video_policy not in {'all','failures','none'}:raise ValueError('未知录像留存规则')
            require_video=video_policy=='all' or (video_policy=='failures' and result.get('success') is False)
            if type(result.get("success")) is not bool or (require_video and not result.get("video_paths")):
                raise ValueError("callback必须返回明确bool success与执行视频路径")
            artifacts = []
            for v in result.get("video_paths",[]):
                p = Path(v).resolve()
                p.relative_to(directory.resolve())  # 不接受指向另一回合的旧视频
                if not p.is_file() or p.stat().st_size == 0:
                    raise ValueError("执行视频缺失/空文件")
                artifacts.append({"path": str(p.relative_to(root)), "sha256": hash_file(p)})
            result = {**result, "video_paths": [str(root / a["path"]) for a in artifacts]}
            record.update(status="completed", result=result, video_artifacts=artifacts,
                          wall_s=time.perf_counter() - start)
            write_json(status_file, record)
        except BaseException as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}",
                          wall_s=time.perf_counter() - start)
            write_json(directory / "error.json", record)
            write_json(status_file, record)
            execution.update(status='interrupted_or_error',reason=type(error).__name__,
                             wall_seconds=time.perf_counter()-began)
            write_json(root/'execution.json',execution)
            raise  # 先定位环境/程序错误，不把几十个错误自动跑完当性能评测
        finally:
            records = [json.loads(p.read_text()) for p in sorted((root / "trials").glob("*.json"))]
            summarize(records, root / "analysis", plan=plan)
            execution.update(wall_seconds=time.perf_counter()-began,planned=len(plan['cases']),
                             completed=sum(r['status']=='completed' for r in records),
                             errors=sum(r['status']=='error' for r in records))
            write_json(root/'execution.json',execution)
    records = [json.loads(p.read_text()) for p in sorted((root / "trials").glob("*.json"))]
    if execution['status']=='running':execution.update(status='completed',reason='all_planned_trials_completed')
    execution.update(wall_seconds=time.perf_counter()-began,
                     planned=len(plan['cases']),completed=sum(r['status']=='completed' for r in records),
                     errors=sum(r['status']=='error' for r in records))
    write_json(root/'execution.json',execution)
    return summarize(records, root / "analysis", plan=plan)


def summarize(records, output, plan=None):
    """CPU统计：success/已完成、完成数/计划数分开；失败与程序错误分开。"""
    records = list(records)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    groups, failures, csv_rows = {}, [], []
    seen = set()
    if plan is not None:
        for case in plan["cases"]:
            key = (case["method"], case["condition"])
            groups.setdefault(key, {"planned": 0, "completed": 0, "successes": 0, "errors": 0})["planned"] += 1
    for record in records:
        case, status = record["case"], record["status"]
        if case["id"] in seen:
            raise ValueError("存在重复回合，不允许重复计数")
        seen.add(case["id"])
        key = (case["method"], case["condition"])
        group = groups.setdefault(key, {"planned": None, "completed": 0, "successes": 0, "errors": 0})
        result = record.get("result", {})
        if status == "completed":
            if type(result.get("success")) is not bool:
                raise ValueError("完成回合缺少bool success")
            group["completed"] += 1
            group["successes"] += int(result["success"])
            if not result["success"]:
                failures.append({**case, "video_paths": result.get("video_paths", []), "env_steps": result.get("env_steps")})
        elif status == "error":
            group["errors"] += 1
        csv_rows.append({"id": case["id"], "condition": case["condition"], "method": case["method"],
                         "seed": case["seed"], "status": status, "success": result.get("success") if status == "completed" else None,
                         **{k: result.get(k) for k in ["env_steps", "chunk_calls", "policy_s", "monitor_s", "event_replans", "guidance_calls", "sequence_prediction_skips", "cancelled_condition_predictions"]},
                         "video_paths": json.dumps(result.get("video_paths", []))})
    rows = [{"method": key[0], "condition": key[1], **g,
             "success_rate": g["successes"] / g["completed"] if g["completed"] else None,
             "pending": g["planned"] - g["completed"] - g["errors"] if g["planned"] is not None else None}
            for key, g in sorted(groups.items())]
    if rows:
        with (output / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    totals = []
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        planned = sum(r["planned"] for r in subset) if all(r["planned"] is not None for r in subset) else None
        completed = sum(r["completed"] for r in subset)
        successes = sum(r["successes"] for r in subset)
        total = {"method": method, "planned": planned, "completed": completed, "successes": successes,
                 "errors": sum(r["errors"] for r in subset),
                 "success_rate": successes / completed if completed else None,
                 "completion_rate": completed / planned if planned else None}
        completed_records = [r["result"] for r in records if r["status"] == "completed" and r["case"]["method"] == method]
        for metric in ["env_steps", "chunk_calls", "policy_s", "monitor_s", "event_replans", "guidance_calls", "sequence_prediction_skips", "cancelled_condition_predictions", "peak_torch_allocated_gib"]:
            values = [r[metric] for r in completed_records if type(r.get(metric)) in (int, float)]
            total[f"mean_{metric}"] = sum(values) / len(values) if values else None
            total[f"observed_{metric}_n"] = len(values)
        times = [r['wall_s'] for r in records if r['status']=='completed' and r['case']['method']==method
                 and type(r.get('wall_s')) in (int,float)]
        total['mean_episode_wall_s'] = sum(times)/len(times) if times else None
        totals.append(total)
    write_json(output / "summary.json", {"groups": rows, "overall_by_method": totals,
               "note": "总体按成功数/完成数加权，不直接平均各组百分比；error/未完成不当作任务失败；非配对因果结论"})
    if totals:
        with (output / "overall.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(totals[0]))
            writer.writeheader(); writer.writerows(totals)
    write_json(output / "failure_videos.json", failures)
    if csv_rows:
        with (output / "trials.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
    page = ["<!doctype html><meta charset='utf-8'><title>失败回合视频索引</title>",
            "<h1>失败回合视频索引</h1><p>来自完整回合的环境success；视频播放时间不自动等于控制时间。路径/哈希校验不等于已解码检查。</p>"]
    for failure in failures:
        page.append("<h2>" + html.escape(f"{failure['method']} / {failure['condition']} / seed={failure['seed']}") + "</h2>")
        for video in failure["video_paths"]:
            relative = html.escape(os.path.relpath(video, output), quote=True)
            page.append(f'<video controls preload="none" width="480" src="{relative}"></video><p><a href="{relative}">打开原视频</a></p>')
    (output / "failure_videos.html").write_text("\n".join(page))
    return rows
