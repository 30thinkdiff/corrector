"""Read-only ordinary J36M trial audit; writes only a separate derived report."""
import argparse
import json
from pathlib import Path
import statistics


def coverage(rows, env_steps, due_steps=20):
    # Dense stride1 traces: only score rows count, not denoise/event rows.
    scored = {r['step'] for r in rows if 'score' in r and r.get('score') is not None}
    opportunities = range(due_steps, env_steps)
    valid = sum(step in scored for step in opportunities)
    gap = longest = 0
    for step in opportunities:
        gap = 0 if step in scored else gap + 1
        longest = max(longest, gap)
    return {'scored_steps': valid, 'eligible_steps': len(opportunities),
            'fraction': valid / len(opportunities) if opportunities else None,
            'longest_gap_steps': longest, 'gap_timebase_hz': 20,
            'definition': 'unique actually scored steps / post-warmup stride1 control steps; skips/cancels not added'}


def analyze(records):
    rows, seen, groups, paired = [], set(), {}, {}
    for record in records:
        case = record['case']
        key = (case['condition'], case['seed'], case['method'])
        if key in seen:
            raise ValueError('duplicate condition/seed/method')
        seen.add(key)
        group = groups.setdefault(case['method'], {'completed': 0, 'successes': 0, 'errors': 0, 'pending': 0})
        if record['status'] != 'completed':
            group['errors' if record['status'] == 'error' else 'pending'] += 1
            continue
        result = record['result']
        if type(result['success']) is not bool:
            raise ValueError('completed trial requires boolean success')
        group['completed'] += 1
        group['successes'] += result['success']
        rows.append((case, result))
        paired.setdefault((case['condition'], case['seed']), {})[case['method']] = result['success']
    for name, group in groups.items():
        group['success_rate'] = group['successes'] / group['completed'] if group['completed'] else None
        for metric in ['env_steps', 'chunk_calls', 'event_replans', 'guidance_calls', 'policy_s', 'monitor_s', 'episode_wall_s', 'peak_torch_allocated_gib']:
            values = [r[metric] for c, r in rows if c['method'] == name and type(r.get(metric)) in (int, float)]
            group['mean_' + metric] = statistics.mean(values) if values else None
    comparisons = {}
    for baseline, method in [('H50', 'j_lvm'), ('H20', 'j_lvm'), ('j_lvm', 'j_lvm_ogg'),
                             ('c0_lvm', 'c0_ogg'), ('c0_lvm', 'j_lvm'), ('c0_ogg', 'j_lvm_ogg')]:
        counts = dict(matched=0, both_success=0, both_failure=0, method_only_success=0, baseline_only_success=0)
        for values in paired.values():
            if baseline not in values or method not in values:
                continue
            a, b = values[baseline], values[method]
            counts['matched'] += 1
            counts['both_success' if a and b else 'both_failure' if not a and not b else
                   'method_only_success' if b else 'baseline_only_success'] += 1
        comparisons[baseline + '_vs_' + method] = counts
    return {'methods': groups, 'paired': comparisons, 'scope': 'small paired pilot; no significance or generalization claim'}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('write derived report outside raw experiment root')
    if args.output.exists():
        raise FileExistsError('do not overwrite a prior report')
    report = {'source': str(args.root.resolve()), 'phases': {}}
    for name in ('passive', 'calibration', 'forced', 'pilot'):
        root = args.root / name
        if not root.exists():
            continue
        records = [json.loads(p.read_text()) for p in sorted((root / 'trials').glob('*.json'))]
        report['phases'][name] = analyze(records)
        case_coverage = []
        for record in records:
            if record['status'] != 'completed': continue
            directory = root / 'attempts' / record['case']['id'] / f'attempt_{record["attempt"]:03d}'
            trace = directory / 'monitor_steps.jsonl'
            if trace.exists():
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                case_coverage.append({'case': record['case'], **coverage(rows, record['result']['env_steps'])})
        report['phases'][name]['coverage'] = case_coverage
        report['phases'][name]['performance_scope'] = name == 'pilot'
        if (root / 'execution.json').exists():
            report['phases'][name]['execution'] = json.loads((root / 'execution.json').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.output), 'phases': list(report['phases'])}))


if __name__ == '__main__':
    main()
