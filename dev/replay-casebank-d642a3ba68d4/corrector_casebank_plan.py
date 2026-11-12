"""冻结Replay诊断样本库：旧结果筛选与新seed重复分开，不按新结果调参。"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

STUDY='replay_casebank_20261112_001'
PATTERNS=['001','000','011','111','010','101','100','110']
CONFIRMATION_SEEDS=[2000,2001,2002]
LABELS={'mlp':'MLP','transformer-h1':'T-H1','transformer-h3':'T-H3',
        'transformer-a10-h1':'A10-H1','transformer-a10-h3':'A10-H3'}


def methods():
    rows={'H50':{'family':'baseline','action_steps':50,'model_label':'SmolVLA'},
          'H20':{'family':'baseline','action_steps':20,'model_label':'SmolVLA'}}
    for name,label in LABELS.items():
        for mode in ['lvm','ogg']:
            rows[name+'_'+mode]={'family':'original','model':name,'mode':mode,'action_steps':50,'model_label':label}
    rows['hybrid_t1_a10_ogg']={'family':'hybrid','model':'transformer-h1','guide':'transformer-a10-h1','mode':'ogg','action_steps':50,'model_label':'T-H1 → A10-H1'}
    for mode in ['planned_lvm','executed_lvm']:
        rows[mode]={'family':'consistency','model':'transformer-a10-h1','mode':mode,'action_steps':50,'model_label':'A10-H1'}
    return {name:{**row,'video_policy':'all'} for name,row in rows.items()}


def discover(archive):
    """只采用同一正式五模型/seed42协议，基线不跨模型重复计数。"""
    groups=defaultdict(dict)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        for name in sorted(z.namelist()):
            if '/paired/trials/' not in name or not name.endswith('.json'):continue
            model=next((m for m in LABELS if f'/evaluation/{m}_s42/' in name),None)
            if model is None:continue
            raw=z.read(name);r=json.loads(raw);c=r['case']
            if r.get('status')!='completed':continue
            if c['method'] in ['H50','H20'] and model!='mlp':continue
            key=(c['condition'],c['seed']);method=c['method'] if c['method'].startswith('H') else model+'_'+c['method']
            if method in groups[key]:raise ValueError('同协议出现重复条件，不合并')
            groups[key][method]={'success':r['result']['success'],'member':name,
                'trial_sha256':hashlib.sha256(raw).hexdigest(),'retained_video':bool(r.get('video_artifacts'))}
    classified=defaultdict(list)
    for (condition,seed),rows in sorted(groups.items()):
        if not all(m in rows for m in ['H50','H20','mlp_ogg']):continue
        pattern=''.join('1' if rows[m]['success'] else '0' for m in ['H50','H20','mlp_ogg'])
        classified[pattern].append({'condition':condition,'seed':seed,'pattern':pattern,'methods':rows})
    selected=[];used=set()
    for pattern in PATTERNS:
        candidates=classified[pattern];chosen=[]
        # 先选不同物理状态，尽量覆盖不同任务；不以新实验成败挑选。
        for _ in range(2):
            remaining=[c for c in candidates if c['condition'] not in {s['condition'] for s in chosen}]
            if not remaining:break
            choice=min(remaining,key=lambda c:(c['condition'] in used,
                c['condition'].split('/init_')[0] in {s['condition'].split('/init_')[0] for s in chosen},c['condition'],c['seed']))
            chosen.append(choice);used.add(choice['condition'])
        selected.extend(chosen)
    return {'study_id':STUDY,'pattern_order':['H50','H20','mlp_ogg'],'source_archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'inventory':[{'pattern':p,'matched_cases':len(classified[p]),'physical_states':len({c['condition'] for c in classified[p]}),
                      'video_complete_triples':sum(all(c['methods'][m]['retained_video'] for m in ['H50','H20','mlp_ogg']) for c in classified[p])} for p in PATTERNS],
        'anchors':selected,'missing_patterns':[p for p in PATTERNS if not classified[p]],
        'confirmation_seeds':CONFIRMATION_SEEDS,'models':list(LABELS),'methods':methods(),
        'scope':'outcome-selected diagnostic bank; repeated held-out noise seeds, not unbiased benchmark success rate',
        'original_corrector':'MLP seed42 + original robust LVM + original first-action OGG; checkpoint pinned in models manifest',
        'do_not_force_requested_outcomes':True,'no_training':True,'automatic_expansion':False}


def groups(selection):
    anchors=selection['anchors'];ordered=[]
    for rank in range(2):
        for pattern in PATTERNS:
            selected=[a for a in anchors if a['pattern']==pattern]
            if rank<len(selected):ordered.append(selected[rank])
    rows=[{'condition':a['condition'],'seed':a['seed'],'phase':'discovery_replay',
           'anchor_pattern':a['pattern']} for a in ordered]
    for condition in sorted({a['condition'] for a in selection['anchors']}):
        for seed in selection['confirmation_seeds']:
            rows.append({'condition':condition,'seed':seed,'phase':'confirmation',
                         'anchor_pattern':next(a['pattern'] for a in selection['anchors'] if a['condition']==condition)})
    unique={(r['condition'],r['seed']) for r in rows}
    if len(rows)!=len(unique):raise ValueError('筛选seed与确认seed重复，停止')
    return rows


def cases(selection):
    result=[]
    for group in groups(selection):
        for method,setting in selection['methods'].items():
            case={'condition':group['condition'],'seed':group['seed'],'method':method,
                  'config':{**setting,'phase':group['phase'],'anchor_pattern':group['anchor_pattern']}}
            case['id']=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()[:20];result.append(case)
    # 先完整补齐四种核心三方法组合，数分钟即可取回第一份可用回放；不改正式集合。
    priority=[]
    for pattern in ['001','000','011','111']:
        anchor=next((a for a in selection['anchors'] if a['pattern']==pattern),None)
        if anchor:
            priority += [c for c in result if c['condition']==anchor['condition'] and c['seed']==anchor['seed'] and c['method'] in ['H50','H20','mlp_ogg']]
    seen={c['id'] for c in priority}
    return priority+[c for c in result if c['id'] not in seen]
