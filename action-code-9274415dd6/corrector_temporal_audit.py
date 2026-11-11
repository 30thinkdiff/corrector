"""只读检查已完成特征/仿真的时间协议；不加载模型、不重跑回合。

检查声明时间戳与在线到期队列。无法仅据LeRobot的timestamp证明原始采集物理频率。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def audit(features, manifest_path, evaluation):
    manifest = read(manifest_path)
    plan, execution = read(evaluation/'plan.json'), read(evaluation/'execution.json')
    fps = float(plan['protocol']['fps'])
    errors, counts, horizon_set, first_scores, sampling = [], {}, set(), set(), []
    outcomes = {}
    for item in manifest['records']:
        ep = item['episode']
        directory = features/f'episode_{ep}'
        if not (directory/'COMPLETE.json').is_file():
            errors.append(f'episode{ep}: missing COMPLETE')
            continue
        meta = read(directory/'metadata.json')
        times = np.load(directory/'timestamp.npy', allow_pickle=False)
        if times.ndim != 1 or len(times) != item['frames'] or not np.isfinite(times).all():
            errors.append(f'episode{ep}: invalid timestamp shape/length/values')
            continue
        data_fps = float(meta['fps'])
        if not math.isfinite(data_fps) or data_fps <= 0 or len(times) < 2:
            errors.append(f'episode{ep}: invalid fps or too few frames')
            continue
        intervals = np.diff(times)
        sampling.append(float(np.median(intervals)))
        if not np.allclose(intervals, 1/data_fps, atol=1e-4, rtol=1e-4):
            errors.append(f'episode{ep}: stored timestamps disagree with metadata fps={data_fps}')
        if not math.isclose(data_fps, float(manifest['fps'])):
            errors.append(f'episode{ep}: metadata fps differs from manifest')
    records = {p.stem: read(p) for p in (evaluation/'trials').glob('*.json')}
    for case in plan['cases']:
        record = records.get(case['id'])
        if not record or record.get('status') != 'completed':
            errors.append(f"trial {case['id']}: not completed")
            continue
        method = case['method']
        counts[method] = counts.get(method, 0)+1
        success = record['result'].get('success')
        if type(success) is bool:
            key = (case.get('condition', case['id']), case.get('seed', 0))
            outcomes.setdefault(key, {})[method] = success
        if not case['config'].get('uses_corrector'):
            continue
        directory = evaluation/'attempts'/case['id']/f"attempt_{record['attempt']:03d}"
        actual = read(directory/'actual_protocol.json')
        protocol = actual['predictor_protocol']
        seconds = float(protocol['prediction_seconds'])
        data_fps, k = float(protocol['fps']), int(protocol['k_step'])
        physical_steps = seconds*fps
        expected = round(physical_steps)
        first_due = actual.get('monitor_first_due_step', expected)
        if (not math.isclose(seconds, k/data_fps) or not math.isclose(physical_steps, expected)
                or actual['monitor_horizon_steps'] != expected):
            errors.append(f"trial {case['id']}: online horizon disagrees with training seconds")
        horizon_set.add((data_fps, k, seconds, fps, actual['monitor_horizon_steps']))
        trace = directory/'monitor_steps.jsonl'
        if not trace.is_file():
            errors.append(f"trial {case['id']}: missing monitor trace, due-step not verified")
            continue
        rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        scores = [r['step'] for r in rows if 'step' in r and r.get('score') is not None]
        stride = plan['protocol']['corrector_settings']['stride']
        if record['result']['env_steps'] > first_due:
            if not scores or scores[0] != first_due:
                errors.append(f"trial {case['id']}: first score at {scores[:1]}, expected {first_due}")
            else:
                first_scores.add(scores[0])
        for step in scores:
            if step < expected or step % stride:
                errors.append(f"trial {case['id']}: score on unexpected control step {step}")
                break
    paired = {}
    for baseline in ('H50', 'H20', 'lvm'):
        matched = [row for row in outcomes.values() if baseline in row and 'ogg' in row]
        paired[baseline+'_vs_ogg'] = {
            'matched_cases': len(matched),
            'both_success': sum(row[baseline] and row['ogg'] for row in matched),
            'both_failure': sum(not row[baseline] and not row['ogg'] for row in matched),
            'ogg_only_success': sum(not row[baseline] and row['ogg'] for row in matched),
            'baseline_only_success': sum(row[baseline] and not row['ogg'] for row in matched)}
    return {'scope': 'stored dataset timebase + saved online horizon/trace audit; no physical acquisition-clock proof',
            'feature_episodes_checked': len(sampling), 'expected_feature_episodes': len(manifest['records']),
            'stored_frame_dt_seconds_range': [min(sampling), max(sampling)] if sampling else None,
            'execution_status': execution['status'], 'planned_trials': len(plan['cases']),
            'completed_trials_checked': sum(counts.values()), 'by_method': counts,
            'horizon_columns': ['data_fps', 'training_k', 'prediction_seconds', 'online_fps', 'online_due_steps'],
            'horizons': [list(x) for x in sorted(horizon_set)],
            'first_score_steps': sorted(first_scores), 'error_count': len(errors), 'errors_first_10': errors[:10],
            'paired_outcomes': paired,
            'source_physical_clock_verified': False,
            'notes': ['10 FPS dataset vs 20 Hz control is an explicit protocol difference.',
                      'A timestamp generated during format conversion need not be a measurement of source physical time.',
                      'Video encoding fps, control frequency and wall-clock inference time are different quantities.',
                      'This script does not validate image/feature SHA or re-run physics.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/root/autodl-tmp/corrector/object10')
    p.add_argument('--features', type=Path, default=base/'features')
    p.add_argument('--manifest', type=Path, default=base/'manifest.json')
    p.add_argument('--evaluation', type=Path, default=base/'queue_4090_001/evaluation/ac82257d743645b7/paired')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    value = audit(args.features, args.manifest, args.evaluation)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if value['error_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
