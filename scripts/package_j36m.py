"""Source-only J36M ZIP with per-file SHA and safe no-overwrite installer."""
import argparse,hashlib,json,stat,subprocess,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()

def files():
    names=['corrector_j36m.py','corrector_j36m_pipeline.py','corrector_j36m_longrun.py','corrector_j36m_extension.py','corrector_j36m_watchdog.py','package_j36m.py','corrector_train.py','corrector_models.py',
           'corrector_data_workbench.py','corrector_online_runtime.py','corrector_multilayer.py','corrector_multilayer_pilot.py',
           'corrector_prefetch.py','smolvla_libero_tools.py','smolvla_training_workbench.py','package_multilayer_mvp.py','corrector_multilayer_train.py']
    paths=[ROOT/'scripts'/n for n in names]
    for directory in ['scripts/j36m','scripts/multilayer_mvp','tests/j36m','tests/multilayer_mvp',
                      'studies/policy_generalization/corrector/j36m','studies/policy_generalization/corrector/multilayer_mvp']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and (p.suffix in ('.py','.md','.json') or p.name=='LICENSE'))
    paths.extend(ROOT/p for p in ['AGENTS.md','THIRD_PARTY.md','studies/policy_generalization/corrector/online_practice.py',
                 'studies/policy_generalization/corrector/batch_evaluation.py','tests/test_corrector_multilayer_pilot.py',
                 'studies/policy_generalization/corrector/manifests/libero_object10_8695891.json',
                 'studies/policy_generalization/corrector/J36M_LVM_OGG_CODEX_TASK_20261112.md',
                 'studies/policy_generalization/corrector/FORMAL_MULTILAYER_CORRECTOR_J36M_DESIGN_20261112.md',
                 'studies/policy_generalization/corrector/OVERNIGHT_RAW_SINGLE_JOINT_LIBEROPLUS_PLAN_20261112.md',
                 'studies/policy_generalization/corrector/MULTILAYER_TRAINING_MVP_CODEX_BRIEF.md'])
    return sorted(set(paths))


def build(output):
    data={str(p.relative_to(ROOT)):p.read_bytes() for p in files()}
    if any(p.startswith(('/','../')) or Path(p).suffix in ('.pt','.npy','.mp4','.parquet') for p in data):raise ValueError('source-only allowlist')
    manifest={'format':'j36m_source_v1','git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
              'files':{n:sha(v) for n,v in data.items()},'models_or_raw_data_included':False}
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in data.items():z.writestr(n,v)
        z.writestr('DEPLOYMENT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    digest=sha(output.read_bytes());output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
    print(json.dumps({'zip':str(output),'sha256':digest,'bytes':output.stat().st_size,'files':len(data),'git_commit':manifest['git_commit']}))


def install(archive,digest,destination):
    if sha(archive.read_bytes())!=digest:raise ValueError('ZIP SHA mismatch')
    if destination.exists():raise FileExistsError('never overwrite deployment')
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('DEPLOYMENT_MANIFEST.json'))
        if manifest['format']!='j36m_source_v1' or set(z.namelist())!=set(manifest['files'])|{'DEPLOYMENT_MANIFEST.json'}:raise ValueError('unexpected archive members')
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate members')
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16) or info.file_size>2**20:raise ValueError('unsafe member')
        for n,v in manifest['files'].items():
            if sha(z.read(n))!=v:raise ValueError('member hash mismatch')
        destination.mkdir(parents=True,exist_ok=False)
        for n in z.namelist():
            p=destination/n;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(z.read(n))
    print(json.dumps({'installed':str(destination),'code_commit':manifest['git_commit'],'gpu_started':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['build','install']);p.add_argument('--output',type=Path)
    p.add_argument('--archive',type=Path);p.add_argument('--sha256');p.add_argument('--destination',type=Path);args=p.parse_args()
    if args.mode=='build':
        if not args.output:p.error('--output required')
        build(args.output)
    else:
        if None in (args.archive,args.sha256,args.destination):p.error('install requires archive, sha256, destination')
        install(args.archive,args.sha256,args.destination)
