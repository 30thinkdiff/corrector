"""Bounded, non-destructive selection of observed paired outcomes (CPU only).

Unknown/incomplete outcomes never become failures. Selected cohorts are for
diagnosis, not an unbiased success-rate estimate. No file is removed here.
"""
from collections import defaultdict
import json
from pathlib import Path

PAIRS=(('H50','H20'),('mlp_lvm','mlp_ogg'),('j_lvm','j_ogg'),
       ('mlp_ogg','j_ogg'),('transformer-h1_ogg','transformer-a10-h1_ogg'),
       ('transformer-h1_ogg','hybrid_t1_a10_ogg'),('planned_lvm','executed_lvm'))


def select_groups(records, required, *, max_groups=8, per_pattern=2):
    groups=defaultdict(dict)
    for r in records:
        if r.get('status')!='completed' or type(r.get('result',{}).get('success')) is not bool:continue
        c=r['case'];key=(c['condition'],c['seed'])
        if c['method'] in groups[key]:raise ValueError('duplicate method within paired cohort')
        groups[key][c['method']]=r['result']['success']
    eligible=[]
    for (condition,seed),outcomes in sorted(groups.items()):
        if not set(required)<=set(outcomes):continue
        flips=[list(p) for p in PAIRS if set(p)<=set(outcomes) and outcomes[p[0]]!=outcomes[p[1]]]
        eligible.append({'condition':condition,'seed':seed,'signature':''.join('1' if outcomes[m] else '0' for m in required),
                         'outcomes':outcomes,'pair_flips':flips})
    eligible.sort(key=lambda x:(-len(x['pair_flips']),x['condition'],x['seed']))
    chosen=[];patterns=defaultdict(set);seen=set()
    for row in eligible:
        if len(chosen)>=max_groups:break
        physical=row['condition'];signature=row['signature']
        if physical in seen or len(patterns[signature])>=per_pattern:continue
        chosen.append(row);seen.add(physical);patterns[signature].add(physical)
    return {'selection':'observed outcomes; bounded diagnostic selection, no seed search',
            'required_methods':list(required),'eligible_groups':len(eligible),
            'selected_groups':chosen,'max_groups':max_groups,'per_signature_physical_limit':per_pattern,
            'deletion_performed':False,'unknown_is_failure':False}


def from_directory(root,required,**kw):
    return select_groups([json.loads(p.read_text()) for p in (Path(root)/'trials').glob('*.json')],required,**kw)
