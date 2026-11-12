"""Reuse the validated raw FP16 exporter one episode at a time, without its pilot selection.

New unpublished attempts stay separate. Only a fully verified new episode is
atomically published; existing canonical or five reused caches are never changed.
"""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from corrector_data_workbench import digest,write_json
from multilayer_mvp.data import verify_marker,POLICY,WEIGHT,CAMERAS
from .manifest import REUSE


def extract70(args,cfg,manifest,guard):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    specs={r['episode']:r for r in manifest['records']};done=[]
    if args.allow_source_download:
        command=[sys.executable,str(Path(__file__).resolve().parents[1]/'corrector_prefetch.py'),
                 '--manifest',str(args.manifest),'--local-dir',str(args.data_root),'--workers','3',
                 '--max-new-files','0','--min-free-gib','20']
        if args.proxy_url:command+=['--proxy-url',args.proxy_url]
        subprocess.run(command,check=True)
    # Fail on missing shards before loading any large model. No implicit network.
    missing=[f['path'] for r in manifest['records'] for f in r['files'] if not (args.data_root/f['path']).is_file()]
    if missing:raise FileNotFoundError('source shards absent; explicit allow-source-download needed: '+str(sorted(set(missing))))
    checked={}
    for spec in cfg['episodes']:
        meta=json.loads((Path(cfg['features_root'])/f'episode_{spec["episode"]}'/'metadata.json').read_text())
        sources=meta['raw_provenance']['sources']
        if {s['file'] for s in sources}!={f['path'] for f in specs[spec['episode']]['files']}:raise ValueError('source mapping mismatch')
        for source in sources:
            name=source['file']
            if name not in checked:checked[name]=digest(args.data_root/name)
            value=checked[name]
            if value!=source['sha256']:raise ValueError('Parquet SHA differs from frozen Connector provenance: '+name)
    for spec in cfg['episodes']:
        ep=spec['episode'];canonical=Path(spec['raw_path']);guard()
        if canonical.exists():
            marker=verify_marker(canonical,raw=True)
            if (marker['episode']!=ep or marker['num_frames']!=spec['frames'] or marker['pooling']!='none'
                or marker['policy_revision']!=POLICY or marker['policy_weight_sha256']!=WEIGHT
                or marker['camera_keys']!=CAMERAS or marker['shape_per_layer']!=[spec['frames'],2,1024,768]
                or marker['deep_cache_complete_sha256']!=digest(Path(cfg['features_root'])/f'episode_{ep}'/'COMPLETE.json')):
                raise ValueError('existing raw protocol conflict')
            done.append(ep);continue
        if ep in REUSE:raise FileNotFoundError('required original raw cache missing, will not recreate over it')
        canonical.parent.mkdir(parents=True,exist_ok=True)
        attempt=canonical.parent/'.work'/f'episode_{ep}_{time.time_ns()}'
        attempt.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'corrector_multilayer_pilot.py'),
             '--runtime',str(args.runtime),'--source-root',str(args.source_root),'--deep-root',cfg['features_root'],
             '--data-root',str(args.data_root),'--manifest',str(args.manifest),'--output',str(attempt),
             '--episodes',str(ep),'--layers','2','8','--save-fp16','--min-free-gib','20']
        log=output/f'extract_{ep}.log'
        with log.open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        raw=attempt/f'episode_{ep}';marker=verify_marker(raw,raw=True)
        if marker['episode']!=ep or marker['num_frames']!=spec['frames']:raise ValueError('incomplete raw export')
        for layer in ('block_02','block_08'):
            a=np.load(raw/f'{layer}.fp16.npy',mmap_mode='r',allow_pickle=False)
            if a.shape!=(spec['frames'],2,1024,768) or a.dtype!='float16':raise ValueError('raw resolution/dtype changed')
        raw.rename(canonical)  # own unpublished data only, same filesystem; no copies or old-file moves
        done.append(ep);write_json(output/'status.json',{'status':'running','completed_episodes':done,'planned':70})
    write_json(output/'status.json',{'status':'completed','completed_episodes':done,'planned':70})
    return {'status':'completed','episodes':len(done)}
