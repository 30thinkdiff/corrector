"""从已有采集记录找事件线索，并可导出视频关键帧；不运行策略或仿真。"""
import argparse
import json
import subprocess
from pathlib import Path

import torch


def first_time(time, mask):
    indices = mask.nonzero(as_tuple=False).flatten()
    return float(time[indices[0]]) if indices.numel() else None


def summarize_events(trace, final_success):
    """阈值只用于筛选待复核片段，不将接触、抬升或高度下降直接判为成败。"""
    time = trace["elapsed_s"]
    position = trace["object_xyz"]
    bilateral = trace["both_fingers_same_physics_sample"]
    if time.ndim != 1 or time.numel() == 0 or position.shape != (len(time), 3):
        raise ValueError("需要非空时间序列和对齐的 [T,3] 物体位置")
    if bilateral.shape != time.shape or bilateral.dtype != torch.bool:
        raise ValueError("双侧接触需要与时间等长的 bool 张量")
    if not torch.isfinite(time).all() or not torch.isfinite(position).all() or (time[1:] <= time[:-1]).any():
        raise ValueError("时间和位置需要有限，时间必须严格递增")

    rise_cm = (position[:, 2] - position[0, 2]) * 100
    lifted = rise_cm >= 10  # 筛选明显抬升；不是通用抓取成功阈值。
    peak_index = int(rise_cm.argmax())
    previous_peak = torch.cummax(rise_cm, dim=0).values
    decrease = (previous_peak >= 10) & (previous_peak - rise_cm >= 5)

    # 记录“曾采到双侧接触后，连续三个控制区间没再采到”的起点。
    seen_contact, absent_count, contact_change = False, 0, None
    for index, value in enumerate(bilateral.tolist()):
        if value:
            seen_contact, absent_count = True, 0
        elif seen_contact:
            absent_count += 1
            if absent_count == 3:
                contact_change = float(time[index - 2])
                break

    relative_change = None
    if lifted.any():
        # 用夹爪自身坐标系观察物体相对位置，避免把整只手的旋转当作滑移。
        delta = position - trace["gripper_xyz"]
        relative = torch.bmm(trace["gripper_rotation_world"].transpose(1, 2), delta.unsqueeze(-1)).squeeze(-1)
        anchor = int(lifted.nonzero(as_tuple=False)[0, 0])
        drift_cm = torch.linalg.vector_norm(relative - relative[anchor], dim=-1) * 100
        relative_change = first_time(time, (time >= time[anchor]) & (drift_cm >= 3))

    if final_success:
        candidate = "任务已完成；采集窗内有明显抬升" if lifted.any() else "任务已完成；首采集窗不足以解释成功过程"
    elif lifted.any():
        candidate = "曾明显抬升但最终失败；复核保持、搬运和放置阶段"
    else:
        candidate = "首采集窗内无10cm抬升且最终失败；优先复核接近与夹持"

    return {
        "final_success": bool(final_success),
        "capture_start_s": float(time[0]), "capture_end_s": float(time[-1]),
        "first_bilateral_interval_end_s": first_time(time, bilateral),
        "first_rise_10cm_s": first_time(time, lifted),
        "peak_rise_cm": float(rise_cm[peak_index]), "peak_time_s": float(time[peak_index]),
        "first_height_decrease_5cm_after_lift_s": first_time(time, decrease),
        "bilateral_absent_3_intervals_start_s": contact_change,
        "relative_position_change_3cm_after_lift_s": relative_change,
        "candidate_for_review": candidate,
        "limits": "事件仅覆盖采集窗口；高度下降可能是正常放置；相对位移不直接证明滑脱；未采集目标与篮子的接触。",
    }


def make_contact_sheet(video, events, control_hz, target):
    """以控制帧编号取图；不按原片80fps播放时间误读仿真事件。"""
    from PIL import Image, ImageDraw, ImageFont
    import io

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_frames,avg_frame_rate,width,height", "-of", "json", str(video)],
                           capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)["streams"][0]
    count = int(info["nb_frames"])
    times = {0., 2., 4., 6., 8., 10., 12., (count - 1) / control_hz}
    event_keys = ["first_bilateral_interval_end_s", "first_rise_10cm_s",
                  "first_height_decrease_5cm_after_lift_s", "relative_position_change_3cm_after_lift_s"]
    times.update(events[k] for k in event_keys if events[k] is not None)
    frames = sorted({min(count - 1, max(0, round(t * control_hz))) for t in times if t <= (count - 1) / control_hz})
    width, height = info["width"], info["height"]
    columns = 4
    sheet = Image.new("RGB", (columns * width, ((len(frames) + columns - 1) // columns) * (height + 35)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, frame in enumerate(frames):
        result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                                 "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                                capture_output=True, check=True)
        image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        x, y = (index % columns) * width, (index // columns) * (height + 35)
        sheet.paste(image, (x, y + 35))
        time_s = frame / control_hz
        scope = "trace+video" if time_s <= events["capture_end_s"] + 1e-6 else "video only"
        draw.text((x + 5, y + 4), f"~{time_s:.2f}s | frame {frame} | {scope}", fill="black", font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return {"original_video_fps": info["avg_frame_rate"], "control_hz": control_hz,
            "selected_frame_indices": frames, "alignment": "approximately one recorded frame per control step; manually confirm event boundaries"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", action="store_true", help="额外使用 ffmpeg/Pillow 导出关键帧拼图")
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    metrics = json.loads((args.run_dir / "eval_info.json").read_text())
    schema = json.loads((args.run_dir / "grasp_schema.json").read_text())
    trace = torch.load(args.run_dir / "grasp_trace.pt", map_location="cpu", weights_only=True)
    events = summarize_events(trace, metrics["overall"]["pc_success"] == 100)
    case = f"{manifest['condition']}_seed{manifest['seed']}_H{manifest['n_action_steps']}"
    events.update({"case": case, "source": "saved simulation data; CPU review; no policy inference"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.frames:
        video = next(args.run_dir.glob("videos/*/*.mp4"))
        events["frame_extraction"] = make_contact_sheet(video, events, schema["control_frequency_hz"], args.output_dir / f"{case}.png")
    (args.output_dir / f"{case}.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))
    print(json.dumps(events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
