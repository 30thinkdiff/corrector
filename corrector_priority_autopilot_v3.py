#!/usr/bin/env python3
"""AutoDL Object10 priority autopilot v3 (stdlib only).

Run 'preflight' before 'launch'. The program never edits frozen source, model
checkpoints or frozen experiment plans. It requests cooperative STOP, waits for
all previous supervisors to exit, executes the six A10 jobs with frozen CLI,
and, only after all six are verified complete, directly resumes the original frozen 9-job queue.

It is intentionally fail-closed on unknown errors. This script is an untested
live-host integration: review preflight on the actual server before launch.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode = True

BASE = Path('/root/autodl-tmp/corrector')
SUP = Path('/root/corrector_supervision_36h')
WORK = SUP / 'priority_autopilot_v3'
ORIGINAL = BASE / 'object10/night_20261109_001'
FOLLOW = BASE / 'object10/action_followup_20261109_001'
ACTION = FOLLOW / 'experiments'
NIGHT_CODE = BASE / 'night-code-1e80242789'
ACTION_CODE = BASE / 'action-code-9274415dd6'
GUARD_CODE = BASE / 'guardian-code-2263c95a33'
MODELS = ['transformer-a10-h1', 'transformer-a10-h3']
SEEDS = [42,43,44]
# 2026-11-11 15:30 China Standard Time; 14 minutes before the user's reset.
CUTOFF = datetime.fromisoformat('2026-11-11T15:30:00+08:00').timestamp()
REASONABLE_MIN_REMAINING = 15 * 60
GIB = 2**30
SCRIPTS = ('corrector_guardian.py','corrector_followup_queue.py',
           'corrector_night_sweep.py','corrector_train.py',
           'corrector_closed_loop_eval.py','corrector_prediction_report.py',
           'guardian_watch.sh','corrector_recover_nofile.py',
           'corrector_priority_switch.py arm','corrector_priority_safe_v2.py arm')
EXPECTED_ACTION_SHA = {
    'corrector_followup_queue.py': '29dee8098944776cdb7466f62ff719f05752ad69ec9622ee49fae59a977a252f',
    'corrector_night_sweep.py': 'eed20430c73974f8b99f9cef4ad0937bffe07042beb69def3b12f71572f5ff4a',
    'corrector_train.py': 'b2e06f181bd3dc4beee1bbe8fb215100d4664d0566a581fe2a864a15b6b496ca',
}
EXPECTED_GUARD_SHA = '9bf64e2a5197de76e0ec6c3aff65e75b044d4edb9cb83c097818700b81919503'
STOP_PATHS = (ORIGINAL/'STOP', FOLLOW/'STOP', SUP/'STOP')
LOCK_PATHS = (ORIGINAL/'.sweep.lock', FOLLOW/'.followup.lock',
              ACTION/'.sweep.lock', SUP/'.guardian.lock')
STOP_TOKEN_PREFIX = 'autopilot-v3|'


def fail(message): raise RuntimeError(message)
def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    target=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    try:
        with target.open('x') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(target,path)
    finally: target.unlink(missing_ok=True)
def emit(msg,**fields):
    print(json.dumps({'utc':now(),'event':msg,**fields},ensure_ascii=False),flush=True)
def state_file(name):
    return {'guardian':SUP/'guardian_status.json','followup':FOLLOW/'followup_status.json',
            'original':ORIGINAL/'sweep_status.json','action':ACTION/'sweep_status.json'}[name]
def get_state(name):
    p=state_file(name)
    return read(p) if p.is_file() else None

def lock_busy(path):
    if not path.exists(): return False
    with path.open('rb') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(f,fcntl.LOCK_UN)
    return False

def proc_list():
    found=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        try:cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,ValueError):continue
        if str(BASE) in cmd and any(s in cmd for s in SCRIPTS):
            found.append((int(d.name),cmd[:440]))
    return sorted(found)
def running_old():
    # All frozen worker scripts and watchdog are represented, including worker subprocesses.
    return bool(proc_list()) or any(lock_busy(x) for x in LOCK_PATHS)

def verify_plan_code(plan_path, code):
    plan=read(plan_path);hashes=plan.get('code_sha256',{})
    if not hashes:fail('Missing frozen source SHA list: '+str(plan_path))
    for file,expected in hashes.items():
        target=code/file
        if not target.is_file() or digest(target)!=expected:
            fail('Frozen source SHA mismatch: '+str(target))
    return plan

def verified_context():
    night=verify_plan_code(ORIGINAL/'sweep_plan.json',NIGHT_CODE)
    follow=verify_plan_code(FOLLOW/'followup_plan.json',ACTION_CODE)
    for filename,sha in EXPECTED_ACTION_SHA.items():
        if digest(ACTION_CODE/filename)!=sha:fail('Unexpected action bundle version: '+filename)
    gp=read(SUP/'guardian_plan.json')
    if digest(GUARD_CODE/'corrector_guardian.py')!=gp['source_sha256'] or gp['source_sha256']!=EXPECTED_GUARD_SHA:
        fail('Guardian source SHA differs from frozen bundle')
    gsettings=gp['settings']
    if (Path(gsettings['original'])!=ORIGINAL or Path(gsettings['followup'])!=FOLLOW
            or Path(gsettings['original_code'])!=NIGHT_CODE or Path(gsettings['followup_code'])!=ACTION_CODE):
        fail('Guardian source/output paths are unexpected')
    s=follow['settings']
    if s['models']!=MODELS or s['seeds']!=SEEDS or Path(s['output'])!=FOLLOW:
        fail('Action model/seed/output mismatch')
    if Path(follow['predecessor_output'])!=ORIGINAL or Path(follow['predecessor_code'])!=NIGHT_CODE:
        fail('Followup predecessor mismatch')
    if len(night['jobs'])!=9 or [j['name'] for j in night['jobs']] != [f'{m}_s{i}' for i in SEEDS for m in ('transformer-h1','mlp','transformer-h3')]:
        fail('Original 9-job schedule changed')
    orig_deadline=float(gp['deadline_unix'])
    if abs(orig_deadline-datetime.fromisoformat('2026-11-11T11:36:44.262306+08:00').timestamp())>3:
        fail('Original Guardian deadline was modified or is unexpected')
    if CUTOFF<=orig_deadline:fail('New cutoff must exceed previous deadline')
    return night,follow,gp

def inherited_nofile():
    before=resource.getrlimit(resource.RLIMIT_NOFILE)
    soft,hard=before
    target=32768
    try:
        if hard!=resource.RLIM_INFINITY and hard<target:
            # Same approach as the supplied NOFILE recovery bundle; only the
            # current supervisor's limits change, not any existing process.
            resource.setrlimit(resource.RLIMIT_NOFILE,(min(target,max(soft,8192)),target))
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        wanted=target if hard==resource.RLIM_INFINITY else min(target,hard)
        if wanted>=8192 and soft!=resource.RLIM_INFINITY and soft<wanted:
            resource.setrlimit(resource.RLIMIT_NOFILE,(wanted,hard))
    except (OSError,PermissionError,ValueError):pass
    current=resource.getrlimit(resource.RLIMIT_NOFILE)
    if current[0]!=resource.RLIM_INFINITY and current[0]<8192:
        fail(f'RLIMIT_NOFILE only {current}; need >=8192')
    child=json.loads(subprocess.check_output([sys.executable,'-c',
        'import resource,json;print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))'],text=True))
    if child[0]!=resource.RLIM_INFINITY and child[0]<8192:
        fail('Child processes do not inherit sufficient NOFILE')
    return list(current)


def can_raise_nofile_without_touching_parent():
    # Preflight runs a fresh child process; no limit changes to caller or GPU.
    code='import resource\ns,h=resource.getrlimit(resource.RLIMIT_NOFILE)\ntry:\n if h!=resource.RLIM_INFINITY and h<32768:resource.setrlimit(resource.RLIMIT_NOFILE,(min(32768,max(s,8192)),32768))\n s,h=resource.getrlimit(resource.RLIMIT_NOFILE)\n w=32768 if h==resource.RLIM_INFINITY else min(32768,h)\n if w>=8192 and s!=resource.RLIM_INFINITY and s<w:resource.setrlimit(resource.RLIMIT_NOFILE,(w,h))\nexcept (ValueError,OSError,PermissionError):pass\ns,h=resource.getrlimit(resource.RLIMIT_NOFILE)\nraise SystemExit(0 if s==resource.RLIM_INFINITY or s>=8192 else 2)'
    check=subprocess.run([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return check.returncode==0

def free_gib(root):return round(shutil.disk_usage(root).free/GIB,2)

def marker_valid(run):
    marker=run/'JOB_COMPLETE.json';best=run/'best_model.pt'
    if not marker.is_file() or not best.is_file():return False
    return read(marker).get('checkpoint_sha256')==digest(best)

def existing_switch_interference():
    for root in (SUP/'priority_v2_20261110',SUP/'priority_switch_20261110'):
        if root.exists() and any(x.name not in ('.control.lock',) for x in root.iterdir()):
            fail('Previous priority switch journal exists, inspect first: '+str(root))
    for pid,cmd in proc_list():
        if 'corrector_priority_switch.py arm' in cmd or 'corrector_priority_safe_v2.py arm' in cmd:
            fail(f'Existing priority watcher active: PID {pid}')


def safe_initial_check():
    verified_context();existing_switch_interference()
    if WORK.exists():fail('Autopilot journal already exists. Use status or resume; not launch.')
    if CUTOFF-time.time()<REASONABLE_MIN_REMAINING:fail('Too close to fixed cutoff')
    if any(x.exists() for x in (*STOP_PATHS,ACTION/'STOP')):
        fail('Pre-existing STOP; owner unknown, refuse automatic handoff')
    states={k:get_state(k) for k in ('guardian','followup','original','action')}
    for name in ('guardian','followup','original'):
        if (states[name] or {}).get('status')!='running':
            fail(name+' not running at preflight')
    if states['action'] is not None:fail('Action queue already has status, no duplicate launch')
    if ACTION.exists() and any(x.name!='.sweep.lock' for x in ACTION.iterdir()):
        fail('Action output already has files, investigate; no overwrite')
    if not all(lock_busy(x) for x in (LOCK_PATHS[0],LOCK_PATHS[1],LOCK_PATHS[3])):
        fail('Original three locks not all held')
    if lock_busy(LOCK_PATHS[2]):fail('Action queue lock is already held')
    if len(proc_list())<3:fail('Expected old supervisor process tree not found')
    if free_gib(BASE)<7 or free_gib(SUP)<9:fail('Data/system disk space too low')
    if not can_raise_nofile_without_touching_parent():
        fail('Cannot achieve NOFILE >=8192 in a fresh child; fix before requesting old STOP')
    original=states['original'];job=(original.get('job') or {}).get('name')
    known={x['name'] for x in read(ORIGINAL/'sweep_plan.json')['jobs']}
    if job not in known:fail('Current original job unknown or missing')
    if original.get('phase','').endswith('_training'):
        run=ORIGINAL/'runs'/job
        if not (run/'protocol.json').is_file() or not (run/'latest.pt').is_file() or (run/'latest.pt').stat().st_size<1024*1024:
            fail('Current training protocol/latest missing; do not request boundary stop')
    return states


def show(preflight=False):
    print('UTC',now())
    print('NEW_CUTOFF_CHINA',datetime.fromtimestamp(CUTOFF,timezone.utc).astimezone(
        timezone(__import__('datetime').timedelta(hours=8))).isoformat())
    print('REMAINING_HOURS',round((CUTOFF-time.time())/3600,3))
    try:
        verified_context();print('FROZEN_SHA_AND_PLANS PASS')
    except Exception as e:print('FROZEN_SHA_AND_PLANS FAIL:',e)
    for k in ('guardian','followup','original','action'):
        s=get_state(k)
        print('STATE',k,{x:s.get(x) for x in ('status','phase','job','reason','error','pid','child_pid')} if s else None)
    print('STOPS',[str(x) for x in (*STOP_PATHS,ACTION/'STOP') if x.exists()])
    print('LOCKS',[lock_busy(x) for x in LOCK_PATHS])
    print('PROCESSES',proc_list())
    print('FREE_GIB data/system',free_gib(BASE),free_gib(SUP))
    print('DONE_OLD',sum((ORIGINAL/'runs'/j['name']/'JOB_COMPLETE.json').is_file() for j in read(ORIGINAL/'sweep_plan.json')['jobs']),'/9')
    print('DONE_ACTION',sum((ACTION/'runs'/f'{m}_s{i}'/'JOB_COMPLETE.json').is_file() for i in SEEDS for m in MODELS),'/6')
    if (WORK/'state.json').exists():print('AUTOPILOT',read(WORK/'state.json'))
    if preflight:
        try:safe_initial_check();print('AUTOPILOT_PREFLIGHT PASS')
        except Exception as e:print('AUTOPILOT_PREFLIGHT FAIL:',e);raise


def save_stage(stage,**fields):
    previous=read(WORK/'state.json') if (WORK/'state.json').exists() else {}
    value={**previous,'stage':stage,'updated_utc':now(),**fields}
    write(WORK/'state.json',value)
    emit('stage',stage=stage,**fields)
    return value


def stop_token():
    s=read(WORK/'state.json')
    return s['stop_token']


def issue_stops():
    s=read(WORK/'state.json')
    token=s['stop_token']
    if s['stage']!='stop_request_intent':
        save_stage('stop_request_intent',reason='ownership journal created before touching STOP files')
    for path in STOP_PATHS:
        if path.exists():
            if not path.is_file() or path.read_text()!=token:
                fail('STOP has foreign content: '+str(path))
            continue
        # Exclusive creation prevents truncation of operator STOP in race.
        try:
            with path.open('x') as f:
                f.write(token);f.flush();os.fsync(f.fileno())
        except FileExistsError:fail('STOP collision: '+str(path))
    save_stage('stop_requested',stop_files=[str(p) for p in STOP_PATHS])


def verify_owned_stops():
    tok=stop_token()
    for p in STOP_PATHS:
        if not p.is_file() or p.read_text()!=tok:
            fail('Expected owned STOP missing/altered: '+str(p))


def quiesced():
    statuses={n:get_state(n) for n in ('guardian','followup','original')}
    if any((s or {}).get('status')=='error' for s in statuses.values()):
        fail('Old supervisor entered error during stop; refuse to start A10')
    if any((s or {}).get('status') not in ('paused','completed') for s in statuses.values()):
        return False
    if running_old():return False
    # Epoch-boundary save must have completed if our original task was fitting.
    s=read(WORK/'state.json')
    if s.get('old_phase','').endswith('_training'):
        r=ORIGINAL/'runs'/s['old_job'];training=r/'training_status.json'
        if training.is_file():
            status=read(training).get('status')
            if status=='running':return False
            if status not in ('paused','completed'):fail('Unexpected training status: '+str(status))
        if not (r/'latest.pt').is_file():fail('Original checkpoint missing after stop')
    return True


def import_frozen_module(file,name):
    spec=importlib.util.spec_from_file_location(name,file)
    if spec is None or spec.loader is None:fail('Frozen source import unavailable')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def action_command(hours):
    follow=read(FOLLOW/'followup_plan.json')['settings']
    sys.path.insert(0,str(ACTION_CODE))
    f=import_frozen_module(ACTION_CODE/'corrector_followup_queue.py','frozen_a10_followup_v3')
    d=dict(follow)
    for k in ('archive','runtime','source_root','output'):d[k]=Path(d[k])
    cmd=f.new_command(SimpleNamespace(**d),hours)
    if len(cmd)<6 or Path(cmd[2])!=ACTION_CODE/'corrector_night_sweep.py':fail('Unexpected A10 entrypoint')
    if '--output' not in cmd or cmd[cmd.index('--output')+1]!=str(ACTION):fail('Unexpected A10 output')
    if cmd.count('--max-compute-hours')!=1:fail('Duplicate A10 budget')
    return cmd


def job_is_active(name):
    p=state_file(name);s=read(p) if p.is_file() else {}
    return (s.get('status')=='running' or lock_busy(LOCK_PATHS[2] if name=='action' else LOCK_PATHS[3]))


def launch_action():
    verified_context();verify_owned_stops()
    if not quiesced():fail('Old process not quiesced yet')
    if (ACTION/'STOP').exists():fail('Unknown A10 STOP exists')
    if CUTOFF-time.time()<600:fail('Not enough time to begin A10 safely')
    limits=inherited_nofile()
    stage=read(WORK/'state.json')
    attempts=stage.get('action_attempts',0)
    if attempts>1:fail('Action launch/retry limit reached')
    if ACTION.exists() and (ACTION/'sweep_status.json').exists():
        s=get_state('action')
        if s['status']=='error':
            error=str(s.get('error','')).lower()
            # Do NOT reinterpret OOM, NaNs, protocol or code errors as transient.
            if not any(x in error for x in ('退出码-9','退出码-15','退出码137','exit code 137')):
                fail('Non-transient action error; preserve failure evidence')
            phase=s.get('phase','')
            job=(s.get('job') or {}).get('name')
            if phase.endswith('_training') and job:
                run=ACTION/'runs'/job
                if (run/'protocol.json').exists() and not (run/'latest.pt').is_file():
                    fail('A10 protocol exists without a resumable checkpoint')
            previous=WORK/'action_error_before_retry.json'
            if not previous.exists():write(previous,s)
        elif s['status'] not in ('paused','running','completed'):
            fail('Unrecognized action state: '+str(s['status']))
        if s['status']=='running' and not lock_busy(LOCK_PATHS[2]):
            if not stage.get('orphan_proven'):
                fail('Action says running but lock is free: possible crashed state; inspect')
            if any(str(ACTION) in cmd for _,cmd in proc_list()):
                fail('Orphan worker still active; cannot launch another queue')
    if lock_busy(LOCK_PATHS[2]):fail('Action queue lock is occupied')
    if any(str(ACTION_CODE/'corrector_night_sweep.py') in cmd for _,cmd in proc_list()):
        fail('An A10 process exists; cannot start second queue')
    # Launch intent journaled first; interrupted intent is re-evaluated conservatively.
    hours=max(0,(CUTOFF-time.time()-20)/3600)
    cmd=action_command(hours)
    save_stage('action_launch_intent',action_command=cmd,action_attempts=attempts+1,nofile=limits)
    with (WORK/'actions.log').open('a') as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           cwd=BASE,start_new_session=True)
    save_stage('action_running',action_pid=p.pid)
    return p.pid


def maybe_archive():
    # Guardian is paused during independent A10; apply its EXACT previously tested
    # lossless move only to finished runs, after best SHA verification.
    if free_gib(BASE)>=10:return
    if free_gib(SUP)<12:return
    gp=verified_context()[2]
    sys.path.insert(0,str(GUARD_CODE))
    module=import_frozen_module(GUARD_CODE/'corrector_guardian.py','frozen_guard_archive_v3')
    for run in sorted((ACTION/'runs').glob('*')):
        if free_gib(BASE)>=10 or free_gib(SUP)<12:break
        if not marker_valid(run):continue
        dest=SUP/'priority_autopilot_v3_archive'/run.name/'latest.pt'
        journal=WORK/'archive'/f'{run.name}.json'
        if module.tier_checkpoint(run,dest,journal):emit('lossless_checkpoint_archive',run=run.name)


def request_action_stop(reason):
    path=ACTION/'STOP'
    s=read(WORK/'state.json')
    value=s['stop_token']+'ACTION_STOP:'+reason+'\n'
    if path.exists():
        if not path.is_file() or path.read_text()!=value:
            fail('Unknown A10 STOP; refuse overwrite')
    else:
        with path.open('x') as f:
            f.write(value);f.flush();os.fsync(f.fileno())
    save_stage('action_stopping',action_stop_reason=reason)


def rescue_orphan_action():
    """Request supported boundary stop if the A10 QUEUE dies but a child lives.

    No SIGKILL. A restart is allowed only after every old A10 child exits and
    no lock is held, with an intact per-run latest checkpoint when required.
    """
    s=get_state('action') or {}
    job=(s.get('job') or {}).get('name')
    phase=str(s.get('phase') or '')
    workers=[(pid,cmd) for pid,cmd in proc_list() if str(ACTION) in cmd]
    if workers:
        if phase.endswith('_training') and job:
            stop=ACTION/'runs'/job/'TRAIN_STOP'
        elif 'forced_check' in phase or 'paired' in phase or 'evaluation' in phase:
            stop=ACTION/'STOP'
        else:
            fail('Orphan worker exists in an unrecognized A10 phase: '+phase)
        marker=stop_token()+'ORPHAN_BOUNDARY_STOP\n'
        stop.parent.mkdir(parents=True,exist_ok=True)
        if not stop.exists():
            with stop.open('x') as f:f.write(marker);f.flush();os.fsync(f.fileno())
        elif stop.read_text()!=marker:
            fail('Orphan STOP already exists with different owner')
        save_stage('orphan_quiescing',orphan_stop=str(stop),orphan_job=job)
        return
    # The queue and child have both vanished. Never restart without
    # proving its last training checkpoint can be resumed.
    if phase.endswith('_training') and job:
        run=ACTION/'runs'/job
        if (run/'protocol.json').is_file() and (not (run/'latest.pt').is_file() or
                (run/'latest.pt').stat().st_size<1024*1024):
            fail('Orphan training has protocol but no intact-looking latest')
    if read(WORK/'state.json').get('action_attempts',0)>=2:
        fail('Orphan recovery retry limit reached')
    save_stage('old_quiet',orphan_proven=True,orphan_observed=phase)


def actions_complete():
    s=get_state('action')
    if not s or s.get('status')!='completed':return False
    for i in SEEDS:
        for m in MODELS:
            run=ACTION/'runs'/f'{m}_s{i}'
            if not marker_valid(run):fail('A10 best SHA or JOB_COMPLETE missing: '+str(run))
    return True


def direct_original_command():
    # Frozen Followup already provides the exact original-command reconstruction.
    # Running the remaining old 9-job queue directly avoids executing A10 a
    # second time through the Followup after all old jobs finish.
    verified_context()
    sys.path.insert(0,str(ACTION_CODE))
    f=import_frozen_module(ACTION_CODE/'corrector_followup_queue.py','frozen_return_command_v3')
    hours=(CUTOFF-time.time()-20)/3600
    if hours<=0:fail('No remaining supervised compute time')
    cmd=f.frozen_command(NIGHT_CODE,ORIGINAL,hours)
    if Path(cmd[2])!=NIGHT_CODE/'corrector_night_sweep.py':fail('Original frozen runner path changed')
    if '--output' not in cmd or cmd[cmd.index('--output')+1]!=str(ORIGINAL):
        fail('Original output path changed')
    return cmd


def resume_original_direct():
    if not actions_complete():fail('A10 has not completed all six')
    if CUTOFF-time.time()<600:
        save_stage('done',result='six_actions_complete; original_still_paused');return
    if running_old():fail('A frozen script or lock is still active')
    verify_owned_stops()
    limits=inherited_nofile()
    cmd=direct_original_command()
    # Keep Guardian and Followup stopped, to prevent them auto-dispatching a
    # second A10 sweep. The direct original sweep uses existing frozen plan.
    save_stage('original_launch_intent',original_command=cmd,original_nofile=limits)
    if (ORIGINAL/'STOP').read_text()!=stop_token():fail('Original STOP changed during return')
    (ORIGINAL/'STOP').unlink()
    with (WORK/'original_return.log').open('a') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              cwd=BASE,start_new_session=True)
    save_stage('original_running',original_pid=proc.pid)


def owner_token_from_state():
    s=read(WORK/'state.json')
    return s.get('stop_token','')


def telemetry():
    mem=0.0
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):mem=int(line.split()[1])*1024/GIB
    except (OSError,ValueError):pass
    info={'utc':now(),'phase':read(WORK/'state.json')['stage'],
          'data_free_gib':free_gib(BASE),'system_free_gib':free_gib(SUP),
          'ram_available_gib':round(mem,2),'remaining_hours':round((CUTOFF-time.time())/3600,3)}
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True)
        info['gpu']=r.stdout.strip()
    except (OSError,subprocess.SubprocessError) as e:info['gpu_error']=str(e)
    with (WORK/'telemetry.jsonl').open('a') as f:f.write(json.dumps(info,ensure_ascii=False)+'\n')
    return info


def process_autopilot():
    # Called under a durable flock. Restart after SIGKILL can re-attach, not duplicate.
    while True:
        s=read(WORK/'state.json');stage=s['stage']
        if stage=='prepared':
            # The initial launch already performed full preflight. Still validate prior
            # to the first mutation; if a small race occurred fail closed.
            verified_context()
            if any(p.exists() for p in STOP_PATHS):fail('External STOP appeared before first request')
            issue_stops();continue
        if stage=='stop_request_intent':
            issue_stops();continue
        if stage=='stop_requested':
            verify_owned_stops()
            if quiesced():
                save_stage('old_quiet');continue
            if time.time()>=CUTOFF:
                save_stage('failed_deadline_while_stopping',reason='Old processes did not quiesce by cutoff');return
            telemetry();time.sleep(15);continue
        if stage=='old_quiet':
            launch_action();continue
        if stage=='action_launch_intent':
            # Could be interrupted between Popen and PID receipt: never auto start
            # a second queue unless new output is untouched and no process exists.
            action_s=get_state('action')
            if action_s and action_s.get('status')=='running' and lock_busy(LOCK_PATHS[2]):
                save_stage('action_running',adopted_after_interruption=True);continue
            if lock_busy(LOCK_PATHS[2]) or any(str(ACTION_CODE/'corrector_night_sweep.py') in x for _,x in proc_list()):
                save_stage('action_running',adopted_after_interruption=True);continue
            if action_s is None and (not ACTION.exists() or not any(ACTION.iterdir())):
                # No evidence of any started process; require operator confirmation
                # rather than guessing and risking duplicate launches.
                fail('Launch intent exists without process receipt; human review required')
            if action_s and action_s['status']=='completed':
                save_stage('action_finished');continue
            if action_s and action_s['status']=='paused':
                save_stage('action_paused');continue
            fail('Ambiguous interrupted action launch; preserve journal')
        if stage=='action_running':
            action_s=get_state('action')
            if time.time()>=CUTOFF and not (ACTION/'STOP').exists() and not actions_complete():
                request_action_stop('cutoff');continue
            t=telemetry()
            maybe_archive()
            if t['data_free_gib']<6.2 or t['system_free_gib']<9:
                request_action_stop('disk_reserve');continue
            if action_s and action_s.get('status')=='error':
                # A failed sweep may still have a detached train/eval child.
                # Stop that child at a supported boundary before considering retry.
                if lock_busy(LOCK_PATHS[2]):time.sleep(15);continue
                if any(str(ACTION) in cmd for _,cmd in proc_list()):
                    rescue_orphan_action();continue
                err=str(action_s.get('error','')).lower()
                if s.get('action_attempts',0)<2 and any(x in err for x in ('退出码-9','退出码-15','退出码137','exit code 137')) and CUTOFF-time.time()>900:
                    save_stage('old_quiet',action_error=err)
                    time.sleep(5);continue
                fail('A10 error; refusing unlimited retry: '+str(action_s.get('error')))
            if action_s and action_s.get('status')=='completed':
                if lock_busy(LOCK_PATHS[2]):time.sleep(10);continue
                if actions_complete():save_stage('action_finished');continue
            if action_s and action_s.get('status')=='paused':
                if lock_busy(LOCK_PATHS[2]):time.sleep(10);continue
                save_stage('action_paused',reason=action_s.get('reason'));continue
            if action_s is None and not lock_busy(LOCK_PATHS[2]):
                # The queue takes a moment to create initial status; after that
                # a missing status is a launch failure and must not loop forever.
                born=datetime.fromisoformat(s['updated_utc']).timestamp()
                if time.time()-born>90:
                    fail('A10 created no status/lock within 90s after launch')
            # If queue PID disappeared but status remains running and lock is free,
            # the failure is ambiguous: do not launch duplicates.
            if action_s and action_s.get('status')=='running' and not lock_busy(LOCK_PATHS[2]):
                # Short grace for atomic status update after process exit.
                time.sleep(5)
                if (get_state('action') or {}).get('status')=='running' and not lock_busy(LOCK_PATHS[2]):
                    rescue_orphan_action();continue
            time.sleep(20);continue
        if stage=='orphan_quiescing':
            # A detached train/eval child may still be persisting a checkpoint.
            # Keep its STOP intact until the child is gone; don't start another.
            telemetry()
            workers=[x for x in proc_list() if str(ACTION) in x[1]]
            if workers or lock_busy(LOCK_PATHS[2]):
                time.sleep(20);continue
            p=Path(s['orphan_stop'])
            expected=stop_token()+'ORPHAN_BOUNDARY_STOP\n'
            if not p.is_file() or p.read_text()!=expected:
                fail('Orphan STOP changed unexpectedly; cannot resume')
            if s.get('orphan_job'):
                run=ACTION/'runs'/s['orphan_job']
                if (run/'protocol.json').is_file() and not (run/'latest.pt').is_file():
                    fail('Orphan save is not resumable')
            p.unlink()  # Only our own unique stop marker, after child exit.
            save_stage('old_quiet',orphan_proven=True,orphan_observed='worker_clean_exit')
            continue
        if stage=='action_stopping':
            path=ACTION/'STOP'
            if not path.is_file() or not path.read_text().startswith(stop_token()):fail('Action STOP was modified')
            if lock_busy(LOCK_PATHS[2]):telemetry();time.sleep(20);continue
            st=get_state('action')
            if st and st.get('status') in ('paused','completed'):
                save_stage('action_paused',reason='requested_safe_stop');continue
            fail('Action exited without a proper paused/completed state')
        if stage=='action_finished':
            if not actions_complete():fail('A10 completion check failed after exit')
            if time.time()>=CUTOFF-600:
                save_stage('done',result='six_actions_complete; original_still_paused');return
            resume_original_direct();continue
        if stage=='original_launch_intent':
            # If the owner died just after launching the frozen old queue,
            # adopt it; if uncertain, never launch a second copy.
            o=get_state('original')
            if lock_busy(LOCK_PATHS[0]) and any(str(NIGHT_CODE/'corrector_night_sweep.py') in x for _,x in proc_list()):
                save_stage('original_running',adopted_after_interruption=True);continue
            fail('Original queue launch receipt ambiguous; require operator inspection')
        if stage=='original_running':
            telemetry()
            o=get_state('original')
            if lock_busy(LOCK_PATHS[0]):
                time.sleep(30);continue
            if o and o.get('status') in ('completed','paused'):
                save_stage('done',result='original_'+o['status']);return
            fail('Direct original queue vanished or reported error; preserve state')
        if stage in ('action_paused','done','done_no_return','failed_deadline_while_stopping'):
            emit('finished',stage=stage);return
        if stage=='failed':return
        fail('Unknown autopilot stage: '+str(stage))


def run():
    with (WORK/'.runner.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:fail('Another autopilot runner is active')
        inherited_nofile()
        try:
            process_autopilot()
        except Exception as e:
            # A daemon crash may leave A10 running; do not kill children.
            emit('SAFETY_BLOCK',error=str(e))
            save_stage('failed',error=f'{type(e).__name__}: {e}',failed_from=read(WORK/'state.json').get('stage'))
            raise


def watch():
    # Limited daemon restart after abrupt signals, never after safety failures.
    for attempt in range(1,4):
        p=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'_run'],
            stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,cwd=BASE)
        if p.returncode==0:return
        s=read(WORK/'state.json') if (WORK/'state.json').exists() else {}
        if s.get('stage')=='failed':return
        if p.returncode not in (-9,137,139) or attempt==3:return
        emit('watch_restart',attempt=attempt,rc=p.returncode)
        time.sleep(30)


def launch():
    safe_initial_check()
    # Validate NOFILE before any STOP request and before creating journal.
    inherited_nofile()
    # Atomic mkdir is the single-use launch barrier: prevents two launchers.
    WORK.mkdir(parents=True,exist_ok=False)
    token=STOP_TOKEN_PREFIX+str(os.getpid())+'|'+str(time.time_ns())+'\n'
    original=get_state('original')
    write(WORK/'state.json',{
        'stage':'prepared','created_utc':now(),'updated_utc':now(),
        'old_job':(original.get('job') or {}).get('name'),'old_phase':original.get('phase'),
        'stop_token':token,'original_guardian_deadline':read(SUP/'guardian_plan.json')['deadline_unix'],
        'cutoff_unix':CUTOFF,'action_attempts':0})
    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'_watch'],
        stdin=subprocess.DEVNULL,stdout=(WORK/'autopilot.log').open('a'),
        stderr=subprocess.STDOUT,cwd=BASE,start_new_session=True)
    emit('watchdog_launched',pid=p.pid,log=str(WORK/'autopilot.log'))
    time.sleep(2)
    if p.poll() is not None:fail('Watchdog exited immediately; inspect '+str(WORK/'autopilot.log'))
    print('RUNNING: watch with `python',__file__,'status`')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('preflight','launch','status','resume','_watch','_run'))
    args=p.parse_args()
    if args.mode=='status':show();return
    if args.mode=='preflight':show(preflight=True);return
    if args.mode=='launch':launch();return
    if args.mode=='resume':
        if not (WORK/'state.json').is_file():fail('No previous autopilot journal')
        s=read(WORK/'state.json')
        if s['stage'] in ('failed','done','action_paused'):
            fail('Autopilot is terminal/paused: inspect; do not restart blindly')
        f=WORK/'.runner.lock'
        if lock_busy(f):fail('Autopilot still active')
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'_watch'],
            stdin=subprocess.DEVNULL,stdout=(WORK/'autopilot.log').open('a'),stderr=subprocess.STDOUT,
            cwd=BASE,start_new_session=True)
        print('RESUME_ATTEMPT_PID',proc.pid)
        return
    if args.mode=='_watch':watch();return
    if args.mode=='_run':run();return

if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,KeyError,ValueError,TypeError,ImportError,AssertionError) as e:
        print('SAFETY BLOCK:',e,file=sys.stderr);sys.exit(2)
