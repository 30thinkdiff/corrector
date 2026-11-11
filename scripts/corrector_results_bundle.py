"""只读导出/取回Corrector结果：不含权重、图像、视频、特征或完整环境。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

DEFAULT_ROOTS={'night':Path('/root/autodl-tmp/corrector/object10/night_20261109_001'),
               'actions':Path('/root/autodl-tmp/corrector/object10/action_followup_20261109_001')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())


class Exporter:
    def __init__(self,archive,roots,max_bytes=256*2**20,hash_weights=True):
        self.z,self.roots,self.max_bytes,self.hash_weights=archive,roots,max_bytes,hash_weights
        self.manifest={'format':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'read-only metrics/protocol/trial export; no model weights, raw images, videos, features or secrets',
            'roots':{k:str(v) for k,v in roots.items()},'members':{},'runs':[],'evaluations':[],'warnings':[]}
        self.total=0;self.pending_traces=[]
    def put(self,name,raw,source=None,optional=False):
        if name in self.manifest['members']:return name
        if self.total+len(raw)>self.max_bytes:
            if optional:self.manifest['warnings'].append('Optional file omitted by budget: '+name);return None
            raise RuntimeError('必需结果超过导出预算，未修改远程数据；增大--max-mib')
        self.z.writestr(name,raw);self.total+=len(raw)
        self.manifest['members'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            **({'source':str(source)} if source else {})}
        return name
    def add(self,alias,root,path,optional=False):
        if not path.is_file():return None
        if path.is_symlink():raise RuntimeError('结果元数据不能是指向外部的链接：'+str(path))
        rel=path.relative_to(root);before=path.stat();raw=path.read_bytes();after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('结果仍在写入：'+str(path))
        return self.put(alias+'/'+rel.as_posix(),raw,path,optional)
    def collect(self):
        for alias,root in self.roots.items():
            if not root.is_dir():self.manifest['warnings'].append('Missing source root: '+str(root));continue
            for path in root.rglob('*'):
                if path.name in {'sweep_plan.json','sweep_status.json','followup_plan.json','followup_status.json','comparison.csv','comparison.json'}:
                    self.add(alias,root,path)
            for protocol_path in sorted(root.rglob('protocol.json')):
                protocol=read(protocol_path)
                if 'model' not in protocol or 'k_step' not in protocol:continue
                directory=protocol_path.parent
                item={'root':alias,'relative':directory.relative_to(root).as_posix(),'files':{}}
                for name in ('protocol.json','result.json','training_status.json','history.json','metrics.csv',
                             'config.json','initial_validation.json','JOB_COMPLETE.json','split_indices.npz'):
                    member=self.add(alias,root,directory/name)
                    if member:item['files'][name]=member
                best=directory/'best_model.pt'
                if best.is_file():item['weight']={'path':str(best),'bytes':best.stat().st_size,
                    'sha256':sha(best) if self.hash_weights else None,'weights_exported':False}
                latest=directory/'latest.pt';item['latest']={'exists':latest.is_file(),'symlink':latest.is_symlink(),
                    'resolved':str(latest.resolve()) if latest.exists() else None}
                self.manifest['runs'].append(item)
            for summary in sorted(root.rglob('summary.json')):
                value=read(summary)
                if 'checkpoint_sha256' in value and 'metrics' in value:
                    self.add(alias,root,summary)
                    for name in ('per_task.csv','per_pair.csv'):
                        self.add(alias,root,summary.parent/name)
            for execution in sorted(root.rglob('execution.json')):
                directory=execution.parent;plan=directory/'plan.json'
                if not plan.is_file():continue
                rel=directory.relative_to(root).as_posix()
                item={'root':alias,'relative':rel,'kind':directory.name,'files':{},'trials':[],
                      'actual_protocols':{},'traces':{},'video_inventory':{}}
                item['files']['execution']=self.add(alias,root,execution)
                item['files']['plan']=self.add(alias,root,plan)
                for path in sorted((directory/'analysis').glob('*')):
                    if path.suffix in {'.json','.csv'}:self.add(alias,root,path)
                for path in sorted((directory/'trials').glob('*.json')):
                    record=read(path);item['trials'].append(self.add(alias,root,path))
                    cid=record['case']['id'];attempt=record.get('attempt',1)
                    local=directory/'attempts'/cid/f'attempt_{attempt:03d}'
                    actual=local/'actual_protocol.json'
                    if actual.is_file():
                        value=read(actual);predictor=value.get('predictor_protocol')
                        if predictor:
                            keys=['source_sha','source_files','model','model_type','h_window','anchor_history','k_step','fps',
                                  'prediction_seconds','action_steps','action_conditioning','encoder_policy_revision',
                                  'training_entry_sha256','model_factory_sha256','precision','data_manifest_sha256','split']
                            value['predictor_protocol']={k:predictor[k] for k in keys if k in predictor}
                        value['full_source_file_sha256']=sha(actual)
                        name=alias+'/'+actual.relative_to(root).as_posix().replace('.json','.export.json')
                        item['actual_protocols'][cid]=self.put(name,json.dumps(value,ensure_ascii=False).encode(),actual)
                    self.pending_traces.append((item,alias,root,local/'monitor_steps.jsonl',cid))
                    videos=[]
                    for video in record.get('video_artifacts',[]):
                        source=directory/video['path']
                        safe=True
                        try:source.resolve().relative_to(directory.resolve())
                        except ValueError:safe=False
                        videos.append({**video,'remote_path':str(source),'present':safe and source.is_file(),
                            'bytes':source.stat().st_size if safe and source.is_file() else None,
                            'content_sha_verified':False})
                    item['video_inventory'][cid]=videos
                self.manifest['evaluations'].append(item)
        for item,alias,root,path,cid in self.pending_traces:
            trace=self.add(alias,root,path,optional=True)
            if trace:item['traces'][cid]=trace
        self.manifest['payload_bytes']=self.total
        self.z.writestr('bundle_manifest.json',json.dumps(self.manifest,ensure_ascii=False,indent=2))
        return self.manifest


def export(args):
    destination=sys.stdout.buffer if args.stdout else args.output
    if not args.stdout:args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        manifest=Exporter(archive,DEFAULT_ROOTS,args.max_mib*2**20,not args.no_weight_hash).collect()
    stream=sys.stderr if args.stdout else sys.stdout
    print(json.dumps({'exported_runs':len(manifest['runs']),'evaluations':len(manifest['evaluations']),
        'payload_mib':manifest['payload_bytes']/2**20,'warnings':manifest['warnings']},ensure_ascii=False),file=stream,flush=True)


def validate_bundle(path):
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('bundle_manifest.json'))
        if manifest.get('format')!=1:raise ValueError('未知结果格式')
        if sum(info.file_size for info in archive.infolist())>300*2**20:raise ValueError('结果包超过本地安全预算')
        for name,item in manifest['members'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('非法成员路径')
            raw=archive.read(name)
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('结果SHA不一致：'+name)
        return manifest


def fetch(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output).free<512*2**20:raise RuntimeError('本地可用空间不足512MiB，不取回')
    final=args.output/'results_bundle.zip';partial=args.output/'results_bundle.zip.partial'
    if final.exists():raise RuntimeError('结果包已存在；直接用analyze，或为重新导出指定新output')
    command=['ssh','-T','-o','ConnectTimeout=15']
    if args.ssh_port:command+=['-p',str(args.ssh_port)]
    remote=[args.remote_python,'-','export','--stdout','--max-mib',str(args.max_mib)]
    command += [args.ssh_host,shlex.join(remote)]
    print('请在本终端完成SSH认证；只读远程结果，流式下载ZIP，不在远程复制数据。',flush=True)
    with partial.open('wb') as file:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=file)
        process.communicate(Path(__file__).read_bytes())
    if process.returncode:raise RuntimeError(f'SSH导出失败({process.returncode})；未修改远程数据。')
    manifest=validate_bundle(partial);partial.replace(final)
    print(json.dumps({'bundle':str(final),'runs':len(manifest['runs']),'evaluations':len(manifest['evaluations'])},ensure_ascii=False))
    if not args.fetch_only:
        from corrector_results_analysis import analyze
        analyze(final,args.output/'analysis',plot=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('export');p.add_argument('--stdout',action='store_true')
    p.add_argument('--output',type=Path,default=Path('/root/corrector_results_20261111.zip'))
    p.add_argument('--no-weight-hash',action='store_true');p.add_argument('--max-mib',type=int,default=256)
    p=sub.add_parser('fetch');p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com')
    p.add_argument('--ssh-port',type=int);p.add_argument('--remote-python',default='/root/autodl-tmp/corrector/bootstrap/bin/python')
    p.add_argument('--output',type=Path,default=Path('outputs/corrector_audit_20261111'))
    p.add_argument('--max-mib',type=int,default=256);p.add_argument('--fetch-only',action='store_true')
    args=parser.parse_args()
    if args.max_mib<1 or args.max_mib>256:raise ValueError('导出预算须在1～256MiB')
    if args.mode=='export':export(args)
    else:fetch(args)

if __name__=='__main__':main()
