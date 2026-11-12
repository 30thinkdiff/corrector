"""Frozen follow-up: JC ablation; optional seed45 replication if <22h elapsed.

Waits for primary successful completion. Shares its original 32h deadline; never
starts after STOP, error, paused primary or any live primary GPU child.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def proposed(code,root,data,python,primary):
 cfg=code/'studies/policy_generalization/corrector/j36m';audit=primary/'audit70/feature_scale_audit.json'
 common=['--receipts',str(root/'receipts'),'--stop-file',str(root/'STOP'),'--data-min-free-gib','5','--system-min-free-gib','8']
 result=[]
 def add(name,mode,out,extra,cfgname='data70.json',group='JC'):
  result.append({'name':name,'mode':mode,'output':str(out),'group':group,'command':[python,'-u',str(code/'scripts/corrector_j36m.py'),mode,'--output',str(out),'--config',str(cfg/cfgname),*common,*extra]})
 add('gpu_JC','gpu-smoke',data/'smoke_JC',['--variant','JC','--microbatch','8','--audit',str(primary/'audit5/feature_scale_audit.json')],cfgname='data5.json')
 for variant,seed,group in [('JC',s,'JC') for s in (42,43,44)]+[(v,45,'seed45') for v in ('J','C0','J_equal','S2','S8')]:
  add(f'{variant}_s{seed}','train',data/'training'/f'{variant}_s{seed}',
    ['--variant',variant,'--audit',str(audit),'--epochs','60','--patience','0','--microbatch','8','--accumulate','4','--seed',str(seed),'--precision','fp32'],group=group)
 return result

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--primary',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--code',type=Path,required=True)
 args=p.parse_args();root=args.root;primary=args.primary;code=args.code
 root.mkdir(parents=True,exist_ok=True)
 with (root/'lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=read(primary/'plan.json');deadline=plan['deadline_epoch'];start=deadline-plan['budget_hours']*3600
  frozen={'source_manifest_sha256':sha(code/'DEPLOYMENT_MANIFEST.json'),'script_sha256':sha(__file__),'deadline_epoch':deadline,'primary_plan_sha256':sha(primary/'plan.json'),
          'jobs':proposed(code,root,args.data_root,sys.executable,primary),'seed45_start_rule':'only if less than22h since original start when JC complete; never select by validation performance'}
  if (root/'plan.json').exists() and read(root/'plan.json')!=frozen:raise ValueError('follow-up frozen plan changed')
  write(root/'plan.json',frozen)
  state=read(root/'status.json') if (root/'status.json').exists() else {'stages':{}}
  state.update(status='running',phase='waiting_primary_completion',pid=os.getpid(),deadline_epoch=deadline)
  def save():state['remaining_hours']=max(0,(deadline-time.time())/3600);write(root/'status.json',state)
  def stopped():
   if (primary/'STOP').exists() or (root/'STOP').exists() or time.time()>=deadline:(root/'STOP').touch();return True
   return False
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:(root/'STOP').touch())
  while True:
   if stopped():state.update(status='paused',reason='STOP_or_original_deadline');save();return
   current=read(primary/'status.json')
   if current['status']=='completed':break
   if current['status'] in ('error','paused'):state.update(status='blocked',reason='primary_not_completed');save();return
   state['primary_phase']=current.get('phase');save();time.sleep(30)
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LEROBOT_SRC='/root/autodl-tmp/corrector/runtime/lerobot/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  sys.path.insert(0,str(code/'scripts'));import corrector_j36m as cli
  from corrector_j36m_longrun import verify_code,artifact_valid,alive
  from types import SimpleNamespace
  verify_code(code,read(code/'DEPLOYMENT_MANIFEST.json'))
  for name in ('cpu','station','gpu_J','gpu_C0','gpu_S2','gpu_S8','gpu_J_equal','data70'):
   write(root/'receipts'/(name+'.json'),cli.receipt(SimpleNamespace(receipts=primary/'receipts'),name))
  seed45=None
  for job in frozen['jobs']:
   if job['group']=='seed45':
    if seed45 is None:
     seed45=state.get('seed45_enabled',time.time()-start<22*3600);state['seed45_enabled']=seed45;save()
    if not seed45:state['stages'][job['name']]={'status':'not_needed','reason':'22h useful primary/JC window already met'};continue
   name=job['name'];artifact=Path(job['output'])/('smoke.json' if job['mode']=='gpu-smoke' else 'result.json');check=job|{'artifact':str(artifact)}
   if artifact_valid(check):state['stages'][name]={'status':'completed','artifact_sha256':sha(artifact)};continue
   if stopped():state.update(status='paused',reason='STOP_or_original_deadline');save();return
   old=state['stages'].get(name,{});pid=old.get('child_pid');child=None
   command=job['command']+['--max-hours',str(max(.001,(deadline-time.time())/3600))]
   if job['mode']=='train' and (Path(job['output'])/'latest.pt').exists():command+=['--resume']
   log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
   if pid and alive(pid):
    if job['output'] not in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace'):raise RuntimeError('PID reused')
   else:
    if job['mode']=='gpu-smoke' and Path(job['output']).exists():raise RuntimeError('partial GPU diagnostic retained; requires review')
    with log.open('a') as f:child=subprocess.Popen(command,cwd=code,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    pid=child.pid
   state.update(status='running',phase=name,child_pid=pid,log=str(log));state['stages'][name]={'status':'running','child_pid':pid,'command':command};save()
   while child.poll() is None if child else alive(pid):
    stopped();save();time.sleep(20)
   state['child_pid']=None
   if not artifact_valid(check):
    state.update(status='paused' if stopped() else 'error',reason='incomplete_followup_artifact');save();return
   state['stages'][name]={'status':'completed','artifact_sha256':sha(artifact)};save()
  state.update(status='completed',phase='done');save()

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if '--root' in sys.argv:
   path=Path(sys.argv[sys.argv.index('--root')+1])/'status.json'
   if path.exists():
    state=read(path);state.update(status='error',error=f'{type(exc).__name__}: {exc}');write(path,state)
  raise
