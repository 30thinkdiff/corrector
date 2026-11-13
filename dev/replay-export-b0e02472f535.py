"""Read-only, bounded export of completed Replay-v1 episodes; never load pickle.

Run on the machine holding results. Only the new output ZIP is written.
Snapshot exports preserve source status and list omissions explicitly.
"""
import argparse,hashlib,json,os,shutil,zipfile
from pathlib import Path


def export(root,output,ids=None):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.is_relative_to(root):raise ValueError('export must be outside read-only result root')
    if output.exists():raise FileExistsError(output)
    read=lambda p:json.loads(p.read_text())
    plan=read(root/'plan.json');execution=read(root/'execution.json');records=[];omitted=[];files={}
    def include(path,name):
        path=path.resolve();path.relative_to(root)
        if not path.is_file():raise ValueError('required file missing: '+name)
        if path.stat().st_size>40*2**20:raise ValueError('single artifact above 40MiB: '+name)
        files['results/'+name]=path
    for path in sorted((root/'trials').glob('*.json')):
        raw=path.read_bytes();r=json.loads(raw)
        if r['status']!='completed' or ids is not None and r['case']['id'] not in ids:continue
        if r['result'].get('capture_schema')!='replay_casebank_v1':raise ValueError('requires native Replay-v1 capture')
        include(path,'trials/'+path.name);records.append(r)
        base=Path('attempts')/r['case']['id']/f"attempt_{r['attempt']:03d}"
        for p in (root/base).rglob('*'):
            if not p.is_file():continue
            name=p.relative_to(root).as_posix()
            if p.suffix in {'.json','.jsonl','.csv','.mp4','.npz'}:include(p,name)
            else:omitted.append({'file':name,'reason':'not safe numeric/text/video format; source retained'})
        for artifact in r['video_artifacts']:
            p=root/artifact['path']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=artifact['sha256']:raise ValueError('video hash differs')
    if not records:raise ValueError('no completed requested trials')
    if ids is not None and set(ids)!={r['case']['id'] for r in records}:raise ValueError('some requested trials not completed')
    for n in ['evidence.json','lerobot_eval.py']:include(root/'recording'/n,'recording/'+n)
    total=sum(p.stat().st_size for p in files.values())
    if total>6*2**30:raise ValueError('export exceeds 6GiB; choose episode IDs')
    derived_plan={**plan,'cases':[r['case'] for r in records]}
    snapshot={**execution,'completed':len(records),'source_execution':execution,'selected_completed_only':True}
    manifest={'source_plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
              'source_root':str(root),'selected_ids':[r['case']['id'] for r in records],
              'source_status':execution['status'],'omitted':omitted,'source_unchanged':True}
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<total+5*2**30:raise ValueError('export would leave less than 5GiB; select fewer episodes')
    partial=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in files.items():z.write(path,name)
        for name,value in [('plan.json',derived_plan),('execution.json',snapshot),('export_manifest.json',manifest)]:
            z.writestr('results/'+name,json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    os.link(partial,output) # atomic publish, fails instead of overwriting a racing export
    partial.unlink()
    return {'completed':len(records),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_status':execution['status']}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--ids',nargs='+');a=p.parse_args();print(json.dumps(export(a.root,a.output,a.ids)))
