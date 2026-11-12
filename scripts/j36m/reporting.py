"""Evidence-only summaries; missing/error outcomes never become robot failures."""
import csv,json
from pathlib import Path
from corrector_data_workbench import write_json


def trace_coverage(rows,expected_steps=None):
    steps=[r for r in rows if 'score' in r]
    by_step={r['step']:r for r in steps}
    if len(by_step)!=len(steps):raise ValueError('duplicate control-step score records')
    horizon=expected_steps if expected_steps is not None else (max(by_step)+1 if by_step else 0)
    missing=[s for s in range(horizon) if s not in by_step]
    eligible=[by_step[s] for s in sorted(by_step) if s>=20]
    valid=sum(r['score'] is not None for r in eligible);longest=streak=0
    for row in eligible:
        streak=streak+1 if row['score'] is None else 0;longest=max(longest,streak)
    return {'scored_steps':valid,'eligible_steps_after_warmup':len(eligible),
            'coverage_after_warmup':valid/len(eligible) if eligible and not missing else None,'longest_unscored_run_steps':longest,
            'missing_trace_steps':missing,'tail_completeness_verified':expected_steps is not None,
            'scope':'control-step alignment, warmup20 excluded; not evidence of detection accuracy'}


def summarize(root):
    root=Path(root);offline={};cases={};binding=None;coverage=[]
    for name in ('C0','S2','S8','J','J_equal','JC'):
        p=root/('train_'+name)/'result.json'
        if p.exists():offline[name]=json.loads(p.read_text())
    for phase in ('development','heldout'):
        folder=root/('plus_'+phase);plan=folder/'plan.json'
        if not plan.exists():continue
        protocol=json.loads(plan.read_text())['protocol']
        pair=(protocol['checkpoint_sha256'],protocol['calibration_sha256'])
        if binding is not None and pair!=binding:raise ValueError('cannot pool evaluations with different checkpoint/calibration')
        binding=pair
        for p in (folder/'trials').glob('*.json'):
            r=json.loads(p.read_text());c=r['case'];key=(c['condition'],c['seed'])
            group=cases.setdefault(key,{})
            if c['method'] in group:raise ValueError('duplicate matched trial')
            group[c['method']]=r
            if r['status']=='completed':
                path=folder/'attempts'/c['id']/f'attempt_{r["attempt"]:03d}'/'monitor_steps.jsonl'
                if path.exists():coverage.append({'condition':c['condition'],'seed':c['seed'],'method':c['method'],**trace_coverage([json.loads(x) for x in path.read_text().splitlines()],r['result']['env_steps'])})
    pairs=[]
    for (condition,seed),group in sorted(cases.items()):
        a,b=group.get('j_lvm'),group.get('j_lvm_ogg')
        if not a or not b or a['status']!='completed' or b['status']!='completed':continue
        sa,sb=a['result']['success'],b['result']['success']
        pairs.append({'condition':condition,'seed':seed,'lvm_success':sa,'ogg_success':sb,
                      'outcome':'both_success' if sa and sb else 'both_failure' if not sa and not sb else 'ogg_rescue' if sb else 'ogg_regression'})
    counts={k:sum(p['outcome']==k for p in pairs) for k in ('both_success','both_failure','ogg_rescue','ogg_regression')}
    output=root/'analysis';output.mkdir(parents=True,exist_ok=True)
    report={'status':'available_results_only' if cases else 'no_formal_evaluations_available','offline':offline,'paired_j_lvm_vs_ogg':{'matched_cases':len(pairs),'counts':counts,'cases':pairs},
            'monitor_coverage':coverage,'formal_evaluations_only':True,
            'note':'excludes normal calibration/forced smoke; no significance claim; absent/error trials omitted, not counted failures'}
    write_json(output/'final_summary.json',report)
    with (output/'paired_j_lvm_ogg.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','seed','lvm_success','ogg_success','outcome'],lineterminator='\n');w.writeheader();w.writerows(pairs)
    return report
