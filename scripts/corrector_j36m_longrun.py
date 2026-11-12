"""Bounded unattended J36M studies; consumes verified smoke, never changes core code.

60 epochs x 5 fixed architectures x seeds42/43/44, same 70-episode split.
Useful completion can precede the wall budget: no dummy computation to burn time.
"""
import argparse,datetime,fcntl,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial');partial.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');partial.replace(path)
def alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def classify(text):
    if 'ResourcePause' in text and any(x in text for x in ('free below','RAM available','GPU occupied')):return 'resources'
    if any(x in text for x in ('Input/output error','[Errno 24] Too many open files','Stale file handle')):return 'retry_checkpoint'
    return 'stop_unknown_error'

def verify_code(code,expected):
    marker=read(code/'DEPLOYMENT_MANIFEST.json')
    if marker!=expected:raise ValueError('deployment manifest changed')
    for name,value in marker['files'].items():
        if digest(code/name)!=value:raise ValueError('deployed source changed: '+name)

def jobs(root,code,python,data_root):
    cli=str(code/'scripts/corrector_j36m.py');config=str(code/'studies/policy_generalization/corrector/j36m/data70.json')
    common=['--receipts',str(root/'receipts'),'--stop-file',str(root/'STOP'),
            '--data-min-free-gib','5','--system-min-free-gib','8']
    result=[]
    def add(name,mode,out,extra=(),artifact=None,cfg=config):
        result.append({'name':name,'mode':mode,'output':str(out),'command':[python,'-u',cli,mode,'--output',str(out),'--config',cfg,*common,*extra],
                       'artifact':str(out/(artifact or 'result.json'))})
    for variant in ('J','C0','S2','S8','J_equal'):
        add('batch8_'+variant,'gpu-smoke',data_root/'batch8_smoke'/variant,
            ['--variant',variant,'--microbatch','8','--audit',str(root/'audit5/feature_scale_audit.json')],
            artifact='smoke.json',cfg=str(code/'studies/policy_generalization/corrector/j36m/data5.json'))
    add('extract70','extract70',data_root/'extraction',artifact='status.json')
    add('verify70','verify70',root/'verified70',artifact='verify.json')
    add('audit70','cache-audit',root/'audit70',artifact='feature_scale_audit.json')
    for seed in (42,43,44):
        for variant in ('J','C0','J_equal','S2','S8'):
            storage=root if seed==44 else data_root
            add(f'{variant}_s{seed}','train',storage/'training'/f'{variant}_s{seed}',
                ['--variant',variant,'--audit',str(root/'audit70/feature_scale_audit.json'),'--epochs','60','--patience','0',
                 '--microbatch','8','--accumulate','4','--seed',str(seed),'--precision','fp32'])
    return result

def artifact_valid(job):
    path=Path(job['artifact'])
    if not path.is_file():return False
    data=read(path)
    if job['mode']=='train':return data.get('status')=='completed' and data.get('checkpoint_reload_verified') and data.get('epochs_completed')==60
    if job['mode']=='gpu-smoke':return data.get('status')=='passed' and data.get('checkpoint_reload_verified') and bool(data.get('shape')) and all(s[0]==8 for s in data['shape'].values())
    if job['mode']=='extract70':return data.get('status')=='completed' and len(data.get('completed_episodes',[]))==70
    if job['mode']=='verify70':return data.get('status')=='passed' and len(data.get('data',[]))==70
    return len(data.get('source',[]))==70 and set(data.get('train_only_delta_rms',{}))=={'delta_block_02','delta_block_08','delta_connector'}

def resources():
    import shutil
    result={'system_free_gib':shutil.disk_usage('/root').free/2**30,'data_free_gib':shutil.disk_usage('/root/autodl-tmp').free/2**30}
    try:result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):result['gpu']=None
    return result

def summary(root,plan):
    entries=[]
    for job in plan['jobs']:
        if job['mode']=='train' and artifact_valid(job):
            data=read(job['artifact']);protocol=read(Path(job['output'])/'protocol.json')
            entries.append({'name':job['name'],'variant':protocol['variant'],'seed':protocol['seed'],'best_epoch':data['best_epoch'],
                'best_validation':data['best_validation'],'parameters':data['parameters'],'epoch_seconds':data['epoch_seconds'],
                'optimizer_steps':data['optimizer_steps'],'result':job['artifact']})
    report={'completed_models':len(entries),'planned_models':15,'models':entries,'scope':'offline 70-episode predictor study; no claimed online success-rate improvement',
            'online_evaluation':{'status':'blocked_missing_verified_Plus_environment','ordinary_Object10_fallback':False}}
    write(root/'analysis/summary.json',report)


def run(args):
    root=args.root;code=args.code;python=args.python
    if args.mode=='stop':
        root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();return
    if args.mode=='status':
        print(json.dumps(read(root/'status.json'),ensure_ascii=False,indent=2));return
    if not (code/'DEPLOYMENT_MANIFEST.json').is_file():raise FileNotFoundError('verified source deployment required')
    if root.exists() and not (root/'plan.json').exists():raise FileExistsError('unknown existing run directory')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        now=time.time();plan_path=root/'plan.json'
        spec={'format':'j36m_longrun_v1','source':read(code/'DEPLOYMENT_MANIFEST.json'),'supervisor_sha256':digest(__file__),
              'smoke_root':str(args.smoke_root),'jobs':jobs(root,code,python,args.data_root),'epochs':60,'seeds':[42,43,44],
              'microbatch':8,'accumulate':4,'precision':'fp32','no_downloads':True,'no_retraining_frozen_weights':True,
              'disk_hard_floors_gib':{'system':8,'data':5},'storage':{'seed42_43':str(args.data_root),'seed44_logs_audit':str(root)}}
        if plan_path.exists():
            if args.mode!='resume':raise FileExistsError('existing plan; use resume')
            plan=read(plan_path)
            if any(plan[k]!=v for k,v in spec.items()):raise ValueError('frozen longrun plan/code changed')
            if (root/'STOP').exists():raise RuntimeError('STOP still present; inspect before removing your own stop request')
        else:
            if args.mode=='resume':raise FileNotFoundError('no plan')
            plan=spec|{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':now+args.hours*3600,'budget_hours':args.hours}
            write(plan_path,plan)
        state=read(root/'status.json') if (root/'status.json').exists() else {'stages':{},'retries':{}}
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                          LEROBOT_SRC='/root/autodl-tmp/corrector/runtime/lerobot/src',
                          OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        state.update(status='running',pid=os.getpid(),phase='waiting_verified_smoke')
        def stop(*_):(root/'STOP').touch()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard, max(soft,8192)) if hard!=resource.RLIM_INFINITY else max(soft,8192),hard))
        state['open_file_limit']=resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        def save():
            state.update(remaining_hours=max(0,(plan['deadline_epoch']-time.time())/3600),deadline_utc=datetime.datetime.fromtimestamp(plan['deadline_epoch'],datetime.timezone.utc).isoformat(),resources=resources())
            write(root/'status.json',state)
        def stopped():
            if (root/'STOP').exists() or time.time()>=plan['deadline_epoch']:
                (root/'STOP').touch();return True
            return False
        save();verify_code(code,plan['source'])
        # The smoke supervisor is a separate explicit prerequisite, not a fake receipt.
        while True:
            current=read(args.smoke_root/'pipeline_status.json') if (args.smoke_root/'pipeline_status.json').exists() else {}
            if current.get('status')=='completed_requested_scope':break
            if current.get('status') in ('paused_or_error','blocked','error','paused'):raise RuntimeError('smoke did not pass; inspect '+str(args.smoke_root/'launch.log'))
            if current.get('pid') and not alive(current['pid']):raise RuntimeError('smoke supervisor disappeared; will not bypass GPU gates')
            if stopped():
                (args.smoke_root/'STOP').touch();state.update(status='paused',phase='waiting_verified_smoke');save();return
            state['smoke_phase']=current.get('phase');save();time.sleep(20)
        sys.path.insert(0,str(code/'scripts'));import corrector_j36m as cli
        from types import SimpleNamespace
        gate_args=SimpleNamespace(receipts=args.smoke_root/'receipts')
        for name in ('gpu_J','gpu_C0','gpu_S2','gpu_S8','gpu_J_equal'):cli.receipt(gate_args,name)
        for name in ('cpu','station'):
            value=cli.receipt(gate_args,name);target=root/'receipts'/(name+'.json')
            if target.exists() and read(target)!=value:raise ValueError('copied gate conflict')
            write(target,value)
        old_audit=args.smoke_root/'audit5/feature_scale_audit.json'
        target=root/'audit5/feature_scale_audit.json'
        if target.exists() and digest(target)!=digest(old_audit):raise ValueError('five-episode audit changed')
        if not target.exists():write(target,read(old_audit))
        # A complete task/epoch/trial is the only automatic recovery boundary.
        for job in plan['jobs']:
            verify_code(code,plan['source']);name=job['name']
            previous=state['stages'].get(name,{})
            if previous.get('status')=='completed':
                if not artifact_valid(job) or digest(job['artifact'])!=previous['artifact_sha256']:raise ValueError('completed result changed: '+name)
                continue
            state.update(phase=name,child_pid=None)
            while True:
                if stopped():state.update(status='paused',reason='STOP_or_wall_budget');save();summary(root,plan);return
                command=job['command']+['--max-hours',str(max(.001,(plan['deadline_epoch']-time.time())/3600))]
                if job['mode']=='train' and (Path(job['output'])/'latest.pt').exists():command+=['--resume']
                # Resume after supervisor interruption must adopt the still-running child.
                old_pid=previous.get('child_pid') if previous.get('status')=='running' else None
                log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
                if old_pid and alive(old_pid):
                    raw=Path(f'/proc/{old_pid}/cmdline').read_bytes().decode(errors='replace')
                    if job['output'] not in raw or str(code) not in raw:raise RuntimeError('PID reused by unrelated process')
                    child=None;pid=old_pid;state['adopted_child']=pid
                else:
                    if artifact_valid(job):
                        state['stages'][name]={'status':'completed','artifact_sha256':digest(job['artifact']),'recovered_from_complete_artifact':True};save();break
                    if job['mode']=='cache-audit' and Path(job['output']).exists():raise RuntimeError('partial diagnostic retained; requires fresh explicitly reviewed diagnostic output')
                    with log.open('a') as stream:
                        child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    pid=child.pid
                state.update(status='running',child_pid=pid,command=command,log=str(log))
                state['stages'][name]={'status':'running','child_pid':pid,'command':command,'started_epoch':time.time()};save()
                while (child.poll() is None if child else alive(pid)):
                    if stopped():(root/'STOP').touch()
                    save();time.sleep(20)
                code_value=child.returncode if child else None;state['child_pid']=None
                if artifact_valid(job):
                    state['stages'][name]={'status':'completed','artifact_sha256':digest(job['artifact']),'returncode':code_value};save();summary(root,plan);break
                if stopped():state.update(status='paused',reason='STOP_or_wall_budget');save();summary(root,plan);return
                error=log.read_text(errors='replace')[-12000:] if log.exists() else ''
                classification=classify(error);count=state['retries'].get(name,0)
                state['stages'][name]={'status':'interrupted_or_error','returncode':code_value,'classification':classification,'log':str(log)}
                if classification=='resources':
                    state.update(status='waiting_resources');save();time.sleep(60)
                    previous={};continue
                if classification=='retry_checkpoint' and count<2 and job['mode'] in ('train','extract70'):
                    state['retries'][name]=count+1;save();time.sleep(30);previous={};continue
                state.update(status='error',reason=classification);save();summary(root,plan);raise RuntimeError('task stopped: '+name+'; inspect '+str(log))
        state.update(status='completed',phase='all_meaningful_jobs_completed',child_pid=None);save();summary(root,plan)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['run','resume','status','stop']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True);p.add_argument('--smoke-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--python',default='/root/autodl-tmp/corrector/runtime/venv/bin/python');p.add_argument('--hours',type=float,default=32.)
    args=p.parse_args()
    if args.hours<22 or args.hours>48:p.error('wall budget must be 22..48 hours')
    try:run(args)
    except Exception as exc:
        path=args.root/'status.json'
        if path.exists():
            state=read(path);state.update(status='error',error=f'{type(exc).__name__}: {exc}');write(path,state)
        raise
if __name__=='__main__':main()
