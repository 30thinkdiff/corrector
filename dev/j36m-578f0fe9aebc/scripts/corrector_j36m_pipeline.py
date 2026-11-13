"""Explicit bounded stage queue. Default ends after smoke; no automatic expansion."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from corrector_data_workbench import write_json,digest
from j36m.training import source_hash


def plan(root,base,allow_download=False,proxy_url=None):
    cli=str(base/'scripts/corrector_j36m.py');conf=base/'studies/policy_generalization/corrector/j36m'
    receipts=root/'receipts';jobs=[]
    common=['--receipts',str(receipts),'--stop-file',str(root/'STOP')]
    def add(name,mode,output,extra=(),cfg='data5.json',group='smoke'):
        jobs.append({'name':name,'mode':mode,'group':group,'output':str(output),
                     'command':[('/root/autodl-tmp/corrector/runtime/sim-venv/bin/python' if mode=='plus' else sys.executable),'-u',cli,mode,'--output',str(output),'--config',str(conf/cfg),*common,*extra]})
    add('preflight','preflight',root/'preflight')
    add('cpu','cpu-smoke',root/'cpu')
    add('audit5','cache-audit',root/'audit5')
    add('station','station-audit',root/'station')
    for kind in ('J','C0','S2','S8','J_equal'):
        add('gpu_'+kind,'gpu-smoke',root/('smoke_'+kind),['--variant',kind,'--audit',str(root/'audit5/feature_scale_audit.json')])
    extraction_flags=(['--allow-source-download'] if allow_download else [])+(['--proxy-url',proxy_url] if proxy_url else [])
    add('extract70','extract70',root/'extraction',extraction_flags,cfg='data70.json',group='extract')
    add('verify70','verify70',root/'verified70',cfg='data70.json',group='extract')
    add('audit70','cache-audit',root/'audit70',cfg='data70.json',group='extract')
    for kind in ('S2','S8','C0','J','J_equal'):
        add('train_'+kind,'train',root/('train_'+kind),['--variant',kind,'--audit',str(root/'audit70/feature_scale_audit.json'),
            '--epochs','30','--patience','0','--microbatch','1','--accumulate','32','--seed','42'],cfg='data70.json',group='train')
    checkpoint=root/'train_J/best.pt'
    add('cal_template','calibration-template',root/'calibration_template.json',['--checkpoint',str(checkpoint)],cfg='data70.json',group='development')
    extra=['--checkpoint',str(checkpoint),'--calibration',str(root/'calibration_template.json'),'--phase','calibration']
    add('calibration_runs','plus',root/'calibration_runs',extra,cfg='data70.json',group='development')
    add('fit_calibration','calibrate',root/'calibration_runs',['--checkpoint',str(checkpoint)],cfg='data70.json',group='development')
    cal=root/'calibration_runs/frozen_calibration.json'
    for phase in ('online-smoke','development','heldout'):
        extra=['--checkpoint',str(checkpoint),'--calibration',str(cal),'--phase',phase]
        if phase=='heldout':extra+=['--development-receipt',str(root/'plus_development/frozen_receipt.json'),
                                    '--physical-reference',str(root/'plus_development/physical_reference.pt')]
        add('plus_'+phase,'plus',root/('plus_'+phase),extra,cfg='data70.json',group='heldout' if phase=='heldout' else 'development')
    return {'format':'j36m_stage_plan_v1','source':source_hash(),
            'config_files':{str(conf/n):digest(conf/n) for n in ('data5.json','data70.json','manifest70.json')},'jobs':jobs,
            'note':'30 fixed epochs each, matched effective batch32, seed42; estimate GPU duration after smoke before long queue'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['plan','run','resume','status','stop'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--until',choices=['smoke','extract','train','development','heldout'],default='smoke')
    p.add_argument('--max-hours',type=float,default=12.);p.add_argument('--allow-source-download',action='store_true');p.add_argument('--proxy-url')
    args=p.parse_args();root=args.root
    if args.mode=='status':
        for name in ['pipeline_status.json','experiment_index.json']:
            f=root/name
            if f.exists():print(f.read_text())
        return
    if args.mode=='stop':root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();print(root/'STOP');return
    frozen=plan(root,Path(__file__).resolve().parents[1],args.allow_source_download,args.proxy_url)
    if root.exists() and not (root/'experiment_index.json').exists():raise FileExistsError('existing unrecognized run root')
    root.mkdir(parents=True,exist_ok=True)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:(root/'STOP').touch(exist_ok=True))
    index=root/'experiment_index.json'
    if index.exists():
        if json.loads(index.read_text())!=frozen:raise ValueError('frozen command/code plan changed; no silent retuning')
    else:write_json(index,frozen)
    if args.mode=='plan':print(json.dumps(frozen,indent=2));return
    if args.mode=='resume' and (root/'STOP').exists():raise RuntimeError('STOP still present; remove only your own request to resume')
    status_path=root/'pipeline_status.json'
    state=json.loads(status_path.read_text()) if status_path.exists() else {'stages':{}}
    if state.get('status')=='running' and state.get('pid')!=os.getpid():
        try:os.kill(state['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('previous pipeline process still alive')
    import fcntl
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        state.update(status='running',pid=os.getpid(),child_pid=None,until=args.until)
        order=['smoke','extract','train','development','heldout'];limit=order.index(args.until)
        # Each resume grants an explicit bounded new invocation; elapsed prior time is retained in stage records.
        deadline=time.monotonic()+args.max_hours*3600
        for job in frozen['jobs']:
            if any(digest(Path(n))!=value for n,value in frozen['config_files'].items()):raise RuntimeError('frozen data selection changed during run')
            if order.index(job['group'])>limit:break
            if state['stages'].get(job['name'],{}).get('status')=='completed':continue
            if (root/'STOP').exists() or time.monotonic()>=deadline:
                state.update(status='paused',reason='STOP_or_budget');write_json(status_path,state);return
            command=job['command']+['--max-hours',str(max(.001,(deadline-time.monotonic())/3600))]
            out=Path(job['output'])
            if job['mode']=='train' and (out/'latest.pt').exists():command+=['--resume']
            if job['mode'] in ('cpu-smoke','gpu-smoke','cache-audit','station-audit') and out.exists():
                state.update(status='blocked',reason='incomplete diagnostic output retained; use fresh explicit stage output, inspect log first')
                write_json(status_path,state);raise FileExistsError(out)
            log=root/(job['name']+'.log');tick=time.monotonic()
            state.update(phase=job['name'],command=command,log=str(log));write_json(status_path,state)
            with log.open('a') as f:
                child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state['child_pid']=child.pid;write_json(status_path,state)
                while child.poll() is None:
                    if time.monotonic()>=deadline:(root/'STOP').touch(exist_ok=True)
                    time.sleep(5);state['phase_seconds']=time.monotonic()-tick;write_json(status_path,state)
            row={'status':'completed' if child.returncode==0 else 'paused_or_error','returncode':child.returncode,
                 'seconds':time.monotonic()-tick,'log':str(log),'command':command}
            state['stages'][job['name']]=row;state['child_pid']=None
            if child.returncode:
                state.update(status='paused_or_error');write_json(status_path,state);return
            write_json(status_path,state)
        from j36m.reporting import summarize
        summarize(root)
        state.update(status='completed_requested_scope',phase='finished');write_json(status_path,state)

if __name__=='__main__':main()
