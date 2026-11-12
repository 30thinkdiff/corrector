"""Small CPU-only supervisor liveness guard. No blind program-error retries."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def alive(pid):
 try:return bool(pid) and Path(f'/proc/{int(pid)}/stat').read_text().split()[2]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False

def may_restart(state,pid_alive,stop_exists,retries):
 return state.get('status') in ('running','waiting_resources') and not pid_alive and not stop_exists and retries<2

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--primary',type=Path,required=True)
 p.add_argument('--extension',type=Path,required=True);p.add_argument('--extension-script',type=Path,required=True);p.add_argument('--code',type=Path,required=True)
 args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
 with (root/'lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=read(args.primary/'plan.json');deadline=plan['deadline_epoch'];python='/root/autodl-tmp/corrector/runtime/venv/bin/python'
  manifest=read(args.code/'DEPLOYMENT_MANIFEST.json')
  frozen={'deadline_epoch':deadline,'source':manifest,'primary':str(args.primary),'extension':str(args.extension),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'extension_script_sha256':hashlib.sha256(args.extension_script.read_bytes()).hexdigest()}
  if (root/'plan.json').exists() and read(root/'plan.json')!=frozen:raise ValueError('watchdog frozen plan changed')
  write(root/'plan.json',frozen)
  state=read(root/'status.json') if (root/'status.json').exists() else {'restarts':{}}
  state.update(status='running',pid=os.getpid())
  jobs={
   'primary':(args.primary,[python,'-u',str(args.code/'scripts/corrector_j36m_longrun.py'),'resume','--root',str(args.primary),'--data-root',plan['storage']['seed42_43'],'--code',str(args.code),'--smoke-root',plan['smoke_root'],'--hours',str(plan['budget_hours'])]),
   'extension':(args.extension,[python,'-u',str(args.extension_script),'--primary',str(args.primary),'--root',str(args.extension),'--data-root','/root/autodl-tmp/corrector/j36m_extension_20261113_001','--code',str(args.code)])}
  def stop(*_):(args.primary/'STOP').touch();(args.extension/'STOP').touch()
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
  children=[]
  while time.time()<deadline and not (args.primary/'STOP').exists():
   reports={}
   for name,(folder,command) in jobs.items():
    path=folder/'status.json'
    if not path.exists():raise FileNotFoundError('supervisor state missing; no blind start: '+str(path))
    item=read(path);living=alive(item.get('pid'));count=state['restarts'].get(name,0)
    reports[name]={'status':item['status'],'pid':item.get('pid'),'alive':living,'phase':item.get('phase')}
    pending=state.get('pending_restarts',{}).get(name)
    if pending and alive(pending):reports[name]['pending_restart_pid']=pending
    if may_restart(item,living or alive(pending),(folder/'STOP').exists(),count):
     for relative,expected in manifest['files'].items():
      if hashlib.sha256((args.code/relative).read_bytes()).hexdigest()!=expected:raise ValueError('source changed, restart refused')
     if hashlib.sha256(args.extension_script.read_bytes()).hexdigest()!=frozen['extension_script_sha256']:raise ValueError('followup source changed')
     with (root/(name+'_restart.log')).open('a') as log:child=subprocess.Popen(command,cwd=args.code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     children.append(child);state['restarts'][name]=count+1;reports[name]['restarted_pid']=child.pid
     state.setdefault('pending_restarts',{})[name]=child.pid
     time.sleep(5)
   state.update(supervisors=reports,remaining_hours=max(0,(deadline-time.time())/3600));write(root/'status.json',state)
   for child in children:child.poll()
   if all(r['status']=='completed' for r in reports.values()):state.update(status='completed');write(root/'status.json',state);return
   if any(r['status'] in ('error','blocked') for r in reports.values()):state.update(status='operator_attention',reason='program errors are not automatically restarted');write(root/'status.json',state);return
   time.sleep(30)
  stop();state.update(status='stopped',reason='operator_STOP_or_original_deadline');write(root/'status.json',state)

if __name__=='__main__':main()
