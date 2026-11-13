"""Freeze task-balanced episodes by ID, never by validation/model performance."""
import copy,hashlib,json
from pathlib import Path
from corrector_data_workbench import POLICY_SHA as POLICY
WEIGHT='9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8'

REUSE={813:'/root/multilayer_export_813_full_001/episode_813',
       **{e:f'/root/multilayer_export_file213_rest_001/episode_{e}' for e in (814,815,816,817)}}


def balanced(source,new_root='/root/autodl-tmp/corrector/j36m/raw70'):
    records=source['records'];chosen=[]
    for task in range(20,30):
        for split,n in [('train',5),('validation',2)]:
            rows=sorted((r for r in records if r['task_id']==task and r['split']==split),key=lambda r:r['episode'])
            if len(rows)<n:raise ValueError('not enough episodes')
            chosen.extend(copy.deepcopy(rows[:n]))
    ids={r['episode'] for r in chosen}
    if not set(REUSE)<=ids:raise ValueError('selection must reuse all five existing episodes')
    files={f['path']:f['bytes'] for r in chosen for f in r['files']}
    frames=sum(r['frames'] for r in chosen)
    summary={'episodes':len(chosen),'frames':frames,
             'pairs':{s:sum(r['frames']-10 for r in chosen if r['split']==s) for s in ('train','validation')},
             'parquet_files':len(files),'parquet_bytes':sum(files.values()),
             'raw_bytes':frames*2*2*1024*768*2,'reuse_raw_bytes':749*2*2*1024*768*2,
             'new_raw_bytes':(frames-749)*2*2*1024*768*2}
    manifest={'format':'j36m_balanced70_v1','selection':'per task20..29: lowest episode IDs, 5 train + 2 validation; no metric access',
              'source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
              'dataset':source['dataset'],'revision':source['revision'],'fps':10.,'summary':summary,'records':chosen}
    config={'features_root':'/root/autodl-tmp/corrector/object10/features','policy_revision':POLICY,'policy_weight_sha256':WEIGHT,
            'raw_shape':[2,1024,768],'connector_shape':[128,960],'fps':10.,'k_step':10,
            'episodes':[{'episode':r['episode'],'task_id':r['task_id'],'split':r['split'],'frames':r['frames'],
                         'raw_path':REUSE.get(r['episode'],str(Path(new_root)/f'episode_{r["episode"]}'))} for r in chosen]}
    return manifest,config
