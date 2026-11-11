#!/usr/bin/env python3
"""Guarded, explicit priority handoff for one historical AutoDL Object10 pipeline.

This is an external control script, NOT a replacement for any frozen training code.
No state/STOP is modified by inspect. arm requests boundary pause only after the
specified target job has a valid JOB_COMPLETE marker. start-actions and
resume-original each require a separate explicit invocation and refuse on
uncertain process/lock/state conditions.

This script has NOT been tested on the user's live remote server.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

GIB = 1024**3
TARGET = 'transformer-h3_s43'
ACTION_MODELS = ['transformer-a10-h1','transformer-a10-h3']
ACTION_SEEDS = [42,43,44]
RELEVANT_NAMES = (
    'corrector_guardian.py','corrector_followup_queue.py',
    'corrector_night_sweep.py','corrector_train.py','corrector_closed_loop_eval.py',
    'guardian_watch.sh','corrector_recover_nofile.py','launch_recovery.sh'
)


def die(msg):
    raise RuntimeError(msg)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with tmp.open('x') as f:
            json.dump(value,f,ensure_ascii=False,indent=2)
            f.write('\n')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def active_processes(base):
    """Best-effort *additional* filter; an active lock is a separate gate."""
    items=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal() or int(p.name)==os.getpid():
            continue
        try:
            raw=(p/'cmdline').read_bytes()
            cmd=raw.replace(b'\x00',b' ').decode(errors='replace')
        except (OSError,ValueError):
            continue
        if str(base) in cmd and any(name in cmd for name in RELEVANT_NAMES):
            items.append((int(p.name),cmd[:550]))
    return sorted(items)


def lock_busy(path):
    path=Path(path)
    if not path.parent.is_dir() or not path.exists():
        return False
    with path.open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:fcntl.flock(f,fcntl.LOCK_UN)
            except OSError:pass
    return False


def cfg(a):
    base=Path(a.base).resolve()
    sup=Path(a.supervision).resolve()
    original=base/'object10/night_20261109_001'
    followup=base/'object10/action_followup_20261109_001'
    return {
        'base':base,'supervision':sup,'original':original,'followup':followup,
        'action':followup/'experiments',
        'night_code':base/'night-code-1e80242789',
        'action_code':base/'action-code-9274415dd6',
        'guardian_code':base/'guardian-code-2263c95a33',
        'marker':original/'runs'/TARGET/'JOB_COMPLETE.json',
        'control':sup/'priority_switch_20261110',
    }


def statuses(p):
    paths={
        'guardian':p['supervision']/'guardian_status.json',
        'followup':p['followup']/'followup_status.json',
        'original':p['original']/'sweep_status.json',
        'action':p['action']/'sweep_status.json',
    }
    out={}
    for key,file in paths.items():
        try:
            x=load(file)
            out[key]={k:x.get(k) for k in ('status','phase','job','pid','child_pid','reason','error','log')}
        except FileNotFoundError:out[key]=None
        except (ValueError,OSError) as e:out[key]={'invalid':str(e)}
    return out


def stops(p):
    return [('original',p['original']/'STOP'),('followup',p['followup']/'STOP'),('guardian',p['supervision']/'STOP')]


def locks(p):
    return {
        'original':p['original']/'.sweep.lock',
        'followup':p['followup']/'.followup.lock',
        'action':p['action']/'.sweep.lock',
        'guardian':p['supervision']/'.guardian.lock',
    }


def deadline(p):
    plan=load(p['supervision']/'guardian_plan.json')
    ts=float(plan['deadline_unix'])
    if not 1750000000<ts<2100000000:die('guardian deadline_unix appears invalid')
    return ts


def inspect(p):
    state=statuses(p)
    print('UTC:',datetime.now(timezone.utc).isoformat())
    try:
        end=deadline(p)
        print('GUARDIAN_DEADLINE_UTC:',datetime.fromtimestamp(end,timezone.utc).isoformat())
        print('REMAINING_HOURS:',round((end-time.time())/3600,3))
    except Exception as e:
        print('GUARDIAN_DEADLINE_UNAVAILABLE:',str(e))
    print('CURRENT_STATES:',json.dumps(state,ensure_ascii=False,indent=2))
    print('TARGET_MARKER:',str(p['marker']),p['marker'].is_file())
    print('STOP_FILES:',{n:str(f) for n,f in stops(p) if f.exists()})
    print('LOCK_BUSY:',{n:lock_busy(f) for n,f in locks(p).items()})
    print('PROCESSES:',json.dumps(active_processes(p['base']),ensure_ascii=False,indent=2))
    for name,path in (('data',p['base']),('supervision',p['supervision'])):
        if path.is_dir():print(f'FREE_GIB_{name.upper()}:',round(shutil.disk_usage(path).free/GIB,3))
    if p['control'].is_dir():
        for name in ('handoff.json','actions_started.json','original_resumed.json'):
            item=p['control']/name
            if item.is_file():print(name,load(item))


def check_sha(plan_path,code_root,required):
    plan=load(plan_path)
    mapping=plan.get('code_sha256')
    if not isinstance(mapping,dict) or not mapping:
        die(f'No code_sha256 in {plan_path}')
    for name,expected in mapping.items():
        file=code_root/name
        if not file.is_file() or sha(file)!=expected:
            die(f'Frozen source SHA mismatch/missing: {file}')
    for name in required:
        if name not in mapping:
            die(f'Essential script not in frozen hash manifest: {name}')
    return plan


def marker_valid(path):
    if not path.is_file():return False
    try:
        value=load(path)
        return bool(value.get('checkpoint_sha256')) and (path.parent/'best_model.pt').is_file()
    except (OSError,ValueError,AttributeError):return False


def verified_checkpoint_marker(path):
    if not marker_valid(path):return False
    marker=load(path)
    return sha(path.parent/'best_model.pt') == marker['checkpoint_sha256']


def verify_common(p):
    if not all(f.is_dir() for f in (p['base'],p['original'],p['followup'],p['supervision'])):
        die('Expected original/followup/supervision dirs missing; wrong remote layout')
    check_sha(p['original']/'sweep_plan.json',p['night_code'],['corrector_night_sweep.py'])
    fp=check_sha(p['followup']/'followup_plan.json',p['action_code'],
                 ['corrector_followup_queue.py','corrector_night_sweep.py'])
    x=fp.get('settings',{})
    if x.get('models')!=ACTION_MODELS or x.get('seeds')!=ACTION_SEEDS:
        die(f'Frozen action models/seeds differ from requested six: {x.get("models")}, {x.get("seeds")}')
    if str(Path(x.get('output','')).resolve()) != str(p['followup']):
        die('Frozen followup output differs from known output')
    if not (p['guardian_code']/'guardian_watch.sh').is_file():
        die('Frozen guardian watcher missing')
    gp=load(p['supervision']/'guardian_plan.json')
    guardian_source=p['guardian_code']/'corrector_guardian.py'
    if not guardian_source.is_file() or sha(guardian_source)!=gp.get('source_sha256'):
        die('Frozen guardian source SHA mismatch')
    return x


def arm(p,a):
    if not a.confirm:
        die('For arm, explicitly pass --confirm. inspect is always nonmutating.')
    verify_common(p)
    current=statuses(p)
    orig=current['original']
    if not orig or orig.get('status')!='running' or (orig.get('job') or {}).get('name')!=TARGET:
        die('Target h3_s43 is not recorded as current running job. Refusing to arm.')
    if p['marker'].exists():die('Target JOB_COMPLETE already exists: watcher cannot guarantee intended gate')
    if any(f.exists() for _,f in stops(p)):die('STOP already exists: an operator may own it')
    if (p['control']/'handoff.json').exists():die('Handoff already recorded')
    if not lock_busy(locks(p)['original']):die('Original queue lock not currently held: cannot establish live owner')
    if deadline(p)-time.time()<120:die('Deadline too close for safe handoff')
    control=p['control'];control.mkdir(parents=True,exist_ok=True)
    log=control/'watcher-state.json'
    save(log,{'status':'armed','target':TARGET,'started_utc':datetime.now(timezone.utc).isoformat()})
    print('ARMED: will ONLY create STOP at the three known levels AFTER target JOB_COMPLETE. Do not launch another queue yet.',flush=True)
    while True:
        if marker_valid(p['marker']):break
        if p['marker'].is_file():die('Target marker exists but lacks expected fields/best weights')
        if any(f.exists() for _,f in stops(p)):die('Another operator added STOP during watch; abort without modifying it')
        st=statuses(p)['original']
        if not st or st.get('status')!='running' or (st.get('job') or {}).get('name')!=TARGET:
            die('Original job changed before expected completion marker; no STOP created')
        if deadline(p)-time.time()<5:
            die('Budget ended before target completed; no STOP created')
        time.sleep(0.4)
    # JOB_COMPLETE is the existing frozen scheduler's durable completion signal.
    made=[]
    for name,file in stops(p):
        try:
            with file.open('x') as f:
                f.write('Priority switch after '+TARGET+' JOB_COMPLETE; see '+str(control)+'\n')
            made.append(str(file))
        except FileExistsError:
            save(control/'partial_stop_collision.json',{'created_by_this_script':made,'collision':str(file)})
            die('STOP collision: no files removed; operator must inspect partial creation')
    record={'target':TARGET,'marker':str(p['marker']),
        'marker_sha256':sha(p['marker']),'created_stop_files':made,
        'event_utc':datetime.now(timezone.utc).isoformat(),
        'warning':'Boundary stop requested, NOT evidence all old processes have exited'}
    save(control/'handoff.json',record)
    save(log,{'status':'stop_requested','event_utc':record['event_utc']})
    print('THREE LEVEL STOP REQUESTED AFTER TARGET JOB_COMPLETE:',json.dumps(record,ensure_ascii=False),flush=True)
    print('ACTION NOT STARTED. Wait and run inspect, then start-actions --confirm.',flush=True)


def check_stopped(p):
    if not verified_checkpoint_marker(p['marker']):die('Target JOB_COMPLETE/best_model.pt SHA mismatch or absent')
    j=load(p['control']/'handoff.json')
    if set(j['created_stop_files'])!={str(f) for _,f in stops(p)}:
        die('Handoff record STOP set mismatch')
    for name,file in stops(p):
        if not file.is_file():die(f'STOP missing: {name}')
    st=statuses(p)
    for name in ('guardian','followup','original'):
        if not st[name] or st[name].get('status') not in ('paused','completed'):
            die(f'{name} status is not paused/completed: {st[name]}')
    b={name:lock_busy(file) for name,file in locks(p).items()}
    if any(b.values()):die(f'Queue locks are still active: {b}')
    processes=active_processes(p['base'])
    if processes:die(f'Training/supervision processes still alive: {processes}')
    if (p['action']/'STOP').exists():die('Actions STOP already present')
    if st['action'] and st['action'].get('status') in ('error','running'):
        die(f'Actions status conflicts: {st["action"]}')


def raise_nofile():
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft<32768:
        try:resource.setrlimit(resource.RLIMIT_NOFILE,(32768,max(hard,32768)))
        except (OSError,ValueError,PermissionError):
            try:resource.setrlimit(resource.RLIMIT_NOFILE,(min(32768,hard),hard))
            except (OSError,ValueError,PermissionError):pass
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft<8192:die(f'RLIMIT_NOFILE={soft}/{hard}, need >=8192, abort before spawning')
    return soft,hard


def action_cmd(p,settings,hours):
    script=p['action_code']/'corrector_night_sweep.py'
    if not script.is_file():die('Frozen action training scheduler missing')
    bootstrap=p['base']/'bootstrap/bin/python'
    if not bootstrap.is_file():die('Bootstrap Python missing')
    args=[str(bootstrap),'-u',str(script),'run',
        '--archive',str(settings['archive']),
        '--runtime',str(settings['runtime']),
        '--source-root',str(settings['source_root']),
        '--output',str(p['action']),
        '--models',*settings['models'],
        '--seeds',*map(str,settings['seeds']),
        '--eval-training-seeds',*map(str,settings['eval_training_seeds']),
        '--epochs',str(settings['epochs']),
        '--patience',str(settings['patience']),
        '--snapshot-every',str(settings['snapshot_every']),
        '--predictor-precision',str(settings['predictor_precision']),
        '--max-compute-hours',str(hours),
        '--min-free-gib',str(settings['min_free_gib'])]
    if settings['no_eval']:args.append('--no-eval')
    return args


def start_actions(p,a):
    if not a.confirm:die('Use --confirm after inspect verified old chain fully stopped')
    settings=verify_common(p)
    check_stopped(p)
    if (p['control']/'actions_started.json').exists():die('Action queue start already recorded; do not launch twice')
    free=shutil.disk_usage(p['base']).free/GIB
    sys_free=shutil.disk_usage(p['supervision']).free/GIB
    if free<5 or sys_free<8:die(f'Low disk free data={free:.2f}, system={sys_free:.2f} GiB')
    remaining=(deadline(p)-time.time()-60)/3600
    if remaining<0.25:die(f'Remaining time {remaining:.3f}h insufficient; do not silently extend budget')
    soft,hard=raise_nofile()
    cmd=action_cmd(p,settings,remaining)
    log=p['control']/'actions.log';log.parent.mkdir(parents=True,exist_ok=True)
    print('FROZEN ACTION CMD:',subprocess.list2cmdline(cmd),flush=True)
    print('RLIMIT_NOFILE:',soft,hard,'deadline unchanged',flush=True)
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,cwd=p['base'])
    save(p['control']/'actions_started.json',{
        'pid':proc.pid,'argv':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),
        'nofile_soft_hard':[soft,hard],'log':str(log),'deadline_unix':deadline(p)})
    time.sleep(1)
    if proc.poll() is not None:
        die(f'Action child exited rapidly with code {proc.returncode}. Inspect {log}. Old STOPs remain.')
    print(f'ACTION STARTED pid={proc.pid}, log={log}; old supervisor remains STOPPED.',flush=True)


def resume_original(p,a):
    if not a.confirm:die('Use --confirm only after six actions + evaluations completed')
    verify_common(p)
    j=load(p['control']/'handoff.json')
    if set(j['created_stop_files'])!={str(f) for _,f in stops(p)}:
        die('STOP record mismatch')
    if (p['control']/'original_resumed.json').exists():die('Already resumed once; inspect state first')
    if not (p['control']/'actions_started.json').exists():die('Actions were not launched by this switch')
    st=statuses(p)
    action_status=(st['action'] or {}).get('status')
    if action_status == 'completed':
        for m in ACTION_MODELS:
            for seed in ACTION_SEEDS:
                path=p['action']/'runs'/f'{m}_s{seed}'/'JOB_COMPLETE.json'
                if not verified_checkpoint_marker(path):die(f'Action job completion/best SHA mismatch: {path}')
    elif a.allow_partial_actions and action_status=='paused':
        print('EXPLICIT PARTIAL OVERRIDE: incomplete action experiments remain resumable in their original output.',flush=True)
    else:
        die(f'Action queue status is {action_status}; expected completed, or paused with --allow-partial-actions.')
    if any(lock_busy(file) for file in locks(p).values()):die('Some queue lock is held')
    if active_processes(p['base']):die('A relevant process is still running')
    if any(not f.exists() for _,f in stops(p)):
        die('A STOP was removed or modified outside this script; inspect before proceeding')
    if deadline(p)-time.time()<60:
        die('Existing 36-hour deadline expired: NOT extending automatically; consult before resuming')
    soft,hard=raise_nofile()
    gp=load(p['supervision']/'guardian_plan.json')['settings']
    watch=p['guardian_code']/'guardian_watch.sh'
    cmd=['bash',str(watch)]
    for key,value in gp.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:cmd.append(flag)
        elif isinstance(value,list):cmd.extend([flag,*map(str,value)])
        elif value is not None:cmd.extend([flag,str(value)])
    # Explicitly never extend the deadline.
    if '--extend-hours' in cmd:die('Unexpected guardian extension in frozen settings')
    for _,file in stops(p):file.unlink()
    log=p['control']/'resume_guardian.log'
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,cwd=p['base'])
    save(p['control']/'original_resumed.json',{'pid':proc.pid,'argv':cmd,
         'nofile_soft_hard':[soft,hard],'utc':datetime.now(timezone.utc).isoformat(),'log':str(log),
         'action_status_at_handoff':action_status,'partial_override':bool(a.allow_partial_actions)})
    time.sleep(1)
    if proc.poll() is not None:die(f'Guardian watcher exited fast code {proc.returncode}; inspect log and status')
    print(f'ORIGINAL PIPELINE RESUME REQUESTED pid={proc.pid}, log={log}; original/plan SHA unchanged',flush=True)



def rollback(p,a):
    """Cancel a fully paused handoff *before* any action queue has started."""
    if not a.confirm:die('Use --confirm after checking inspect')
    verify_common(p)
    check_stopped(p)
    if (p['control']/'actions_started.json').exists():
        die('Actions have already been started; rollback is not allowed; use resume-original instead')
    if (p['control']/'rollback.json').exists():die('Rollback already recorded')
    if deadline(p)-time.time()<60:die('Original deadline has expired; no automatic extension')
    soft,hard=raise_nofile()
    gp=load(p['supervision']/'guardian_plan.json')['settings']
    cmd=['bash',str(p['guardian_code']/'guardian_watch.sh')]
    for key,value in gp.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:cmd.append(flag)
        elif isinstance(value,list):cmd.extend([flag,*map(str,value)])
        elif value is not None:cmd.extend([flag,str(value)])
    for _,file in stops(p):file.unlink()
    log=p['control']/'rollback_guardian.log'
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,cwd=p['base'])
    save(p['control']/'rollback.json',{'pid':proc.pid,'argv':cmd,
         'nofile_soft_hard':[soft,hard],'utc':datetime.now(timezone.utc).isoformat(),'log':str(log)})
    time.sleep(1)
    if proc.poll() is not None:die(f'Rollback watcher exited early; inspect {log}')
    print(f'RESTORED ORIGINAL ORDER pid={proc.pid}, log={log}; new action queue was never started',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['inspect','arm','start-actions','resume-original','rollback'])
    ap.add_argument('--base',default='/root/autodl-tmp/corrector')
    ap.add_argument('--supervision',default='/root/corrector_supervision_36h')
    ap.add_argument('--confirm',action='store_true',help='explicitly authorize this one state-changing phase')
    ap.add_argument('--allow-partial-actions',action='store_true',
        help='only for resume-original: allow boundary-paused, incomplete action experiments to be resumed later')
    a=ap.parse_args();p=cfg(a)
    if a.allow_partial_actions and a.mode!='resume-original':die('--allow-partial-actions applies only to resume-original')
    if a.mode=='inspect':inspect(p)
    elif a.mode=='arm':arm(p,a)
    elif a.mode=='start-actions':start_actions(p,a)
    elif a.mode=='resume-original':resume_original(p,a)
    elif a.mode=='rollback':rollback(p,a)

if __name__=='__main__':
    try:main()
    except (RuntimeError,KeyError,FileNotFoundError,PermissionError,ValueError) as e:
        print('SAFETY BLOCK:',e,file=sys.stderr)
        sys.exit(2)
