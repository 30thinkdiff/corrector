"""只读取回AutoDL项目源码与环境元信息；排除环境、数据、权重、认证文件。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

BASE=Path('/root/autodl-tmp/corrector')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(destination,base=BASE):
    files={};total=0;skipped=[]
    def add(path):
        nonlocal total
        if not path.is_file() or path.is_symlink():return
        relative=path.relative_to(base).as_posix()
        if path.stat().st_size>4*2**20:skipped.append(relative);return
        raw=path.read_bytes();total+=len(raw)
        if total>64*2**20:raise RuntimeError('源码导出超过64MiB预算，未修改远程')
        files[relative]=raw
    roots=[base,base/'runtime']
    roots += [p for p in base.iterdir() if p.is_dir() and ('-code' in p.name or p.name.startswith('local-pipeline-'))]
    for root in roots:
        if not root.exists():continue
        for path in root.iterdir():
            if path.suffix in {'.py','.sh'} or path.name in {'environment.json','simulation.json','bundle_manifest.json',
                    'queue_bundle_manifest.json','guardian_bundle.json','recovery_manifest.json','simulation-constraints.txt','torch-constraints.txt'}:
                add(path)
    for path in [base/'object10/manifest.json',base/'object10/night_20261109_001/sweep_plan.json',
        base/'object10/action_followup_20261109_001/followup_plan.json',
        base/'object10/action_followup_20261109_001/experiments/sweep_plan.json']:
        add(path)
    dependencies=[]
    for name in ('vla-corrector-reference','lerobot','libero-original'):
        repo=base/'runtime'/name
        if not repo.is_dir():continue
        result=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
        dependencies.append({'name':name,'git_sha':result.stdout.strip() if result.returncode==0 else None})
        for pattern in ('LICENSE*','NOTICE*'):
            for path in repo.glob(pattern):add(path)
    manifest={'format':1,'source_base':str(base),'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'top-level own project sources, frozen deployment snapshots, manifests and dependency license/version metadata; excludes data, models, environments, Git config and credentials',
        'files':{name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} for name,raw in files.items()},
        'dependencies':dependencies,'skipped_oversize':skipped,'payload_bytes':total}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
        z.writestr('repository_export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'source_files':len(files),'payload_mib':total/2**20,'skipped':skipped},ensure_ascii=False),file=sys.stderr,flush=True)


def fetch(args):
    args.output.mkdir(parents=True,exist_ok=True)
    destination=args.output/'remote_sources.zip';partial=destination.with_suffix('.zip.partial')
    if destination.exists():raise RuntimeError('源快照已存在，使用新output而非覆盖')
    command=['ssh','-T','-o','ConnectTimeout=15']
    if args.ssh_port:command += ['-p',str(args.ssh_port)]
    command += [args.ssh_host,shlex.join([args.remote_python,'-','export','--stdout'])]
    print('请在此终端完成SSH登录；只读导出源码，不收集权重、数据或账号文件。',flush=True)
    with partial.open('wb') as file:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=file)
        process.communicate(Path(__file__).read_bytes())
    if process.returncode:raise RuntimeError('SSH导出失败，远程项目未修改')
    with zipfile.ZipFile(partial) as z:
        m=json.loads(z.read('repository_export_manifest.json'))
        if sum(v.file_size for v in z.infolist())>65*2**20:raise ValueError('源包超预算')
        for name,info in m['files'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('非法文件路径')
            if hashlib.sha256(z.read(name)).hexdigest()!=info['sha256']:raise ValueError('源快照SHA不一致')
    partial.replace(destination)
    print(json.dumps({'bundle':str(destination),'files':len(m['files']),'sha256':digest(destination)},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('export');p.add_argument('--stdout',action='store_true');p.add_argument('--output',type=Path,default=Path('/root/corrector_sources.zip'))
    p=sub.add_parser('fetch');p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com');p.add_argument('--ssh-port',type=int)
    p.add_argument('--remote-python',default='/root/autodl-tmp/corrector/bootstrap/bin/python')
    p.add_argument('--output',type=Path,default=Path('outputs/corrector_repository_export_20261111'))
    args=parser.parse_args()
    if args.mode=='fetch':fetch(args)
    else:export(sys.stdout.buffer if args.stdout else args.output)

if __name__=='__main__':main()
