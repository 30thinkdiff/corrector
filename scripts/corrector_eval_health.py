"""Read-only status: persisted 'running' is not evidence of a live experiment."""
import argparse,datetime,json,shutil
from pathlib import Path


def process_matches(pid,script,output,proc_root=Path('/proc')):
    if type(pid) is not int or pid<=0:return False
    try:args=(proc_root/str(pid)/'cmdline').read_bytes().decode().split('\0')
    except (OSError,UnicodeError):return False
    return any(Path(arg).name==script for arg in args) and str(output) in args


def inspect(root,proc_root=Path('/proc')):
    root=Path(root).resolve();result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root)}
    for label,relative,script,output in [('queue','queue_status.json','corrector_plus_queue.py',root),
                                         ('worker','results/status.json','corrector_expanded_eval.py',root/'results')]:
        path=root/relative
        if not path.exists():result[label]={'status':'not_created'};continue
        raw=json.loads(path.read_text());alive=process_matches(raw.get('pid'),script,output,proc_root)
        result[label]={'persisted':raw,'process_matches':alive,'status':
                       'interrupted_process_missing' if raw.get('status')=='running' and not alive else raw.get('status'),
                       'status_file_mtime_utc':datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).isoformat()}
    phases={}
    for name in ['plus_smoke','plus','plus_replicates']:
        path=root/'results'/name/'execution.json'
        if not path.exists():continue
        d=json.loads(path.read_text());states={}
        for trial in (path.parent/'trials').glob('*.json'):
            row=json.loads(trial.read_text());key=row['status'];states[key]=states.get(key,0)+1
        phases[name]={'execution':d,'record_counts':states}
    result['phases']=phases
    result['free_gib']={};result['disk_query_errors']={}
    for p in [Path('/root'),Path('/root/autodl-tmp')]:
        try:result['free_gib'][str(p)]=shutil.disk_usage(p).free/2**30
        except OSError as e:result['disk_query_errors'][str(p)]=type(e).__name__
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('/root/corrector_plus_expanded_20261113_001'));a=p.parse_args()
    print(json.dumps(inspect(a.root),ensure_ascii=False,indent=2))
