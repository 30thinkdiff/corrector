"""独立36小时标准库监督器：现有冻结任务→追加初始化评测；不下载/改模型。"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE=Path('/root/autodl-tmp/corrector')


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()
@contextmanager
def lock(p):
    with Path(p).open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('已有监督器或任务持有锁：'+str(p))
        yield

def active(root,kind):
    if not root.exists():return False
    try:
        with lock(root/('.followup.lock' if kind=='followup' else '.sweep.lock')):pass
    except RuntimeError:return True
    status=root/('followup_status.json' if kind=='followup' else 'sweep_status.json')
    if status.exists():
        state=read(status)
        for key in ('pid','child_pid'):
            pid=state.get(key)
            if isinstance(pid,int):
                try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
                except (OSError,UnicodeError):continue
                if 'corrector_' in cmd and str(BASE) in cmd:return True
    return False

def reconstruct(code,root,hours,followup=True):
    plan=read(root/('followup_plan.json' if followup else 'sweep_plan.json'))
    for name,sha in plan['code_sha256'].items():
        if digest(code/name)!=sha:raise RuntimeError('冻结代码校验失败：'+str(code/name))
    cmd=[sys.executable,'-u',str(code/('corrector_followup_queue.py' if followup else 'corrector_night_sweep.py')),'run']
    for k,v in plan['settings'].items():
        flag='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v:cmd.append(flag)
        elif isinstance(v,list):cmd += [flag,*map(str,v)]
        elif v is not None:cmd += [flag,str(v)]
    return [*cmd,'--max-compute-hours',str(hours)]


def tier_checkpoint(run,destination,journal):
    """仅已完整结束的JOB；逐字节校验后原子换成指向无损副本的链接。"""
    latest=run/'latest.pt'
    if latest.is_symlink():return False
    if not all((run/n).is_file() for n in ('JOB_COMPLETE.json','result.json','training_status.json','best_model.pt','latest.pt')):return False
    if read(run/'training_status.json').get('status')!='completed' or not read(run/'result.json').get('checkpoint_reload_verified'):return False
    if read(run/'JOB_COMPLETE.json').get('checkpoint_sha256')!=digest(run/'best_model.pt'):raise RuntimeError('已完成JOB的best发生变化：'+str(run))
    before=latest.stat();sha=digest(latest)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if digest(destination)!=sha:raise RuntimeError('归档目标已有不同内容，不覆盖')
    else:
        partial=destination.with_suffix('.partial');shutil.copy2(latest,partial)
        if digest(partial)!=sha:raise RuntimeError('续训状态转存校验失败；原文件未替换')
        partial.replace(destination)
    after=latest.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or digest(latest)!=sha:
        raise RuntimeError('转存时源文件变化；不替换活动checkpoint')
    value={'source':str(latest),'destination':str(destination),'sha256':sha,'bytes':before.st_size,'status':'verified_copy'}
    write(journal,value)
    link=latest.with_name('latest.pt.placement-link');link.unlink(missing_ok=True)
    link.symlink_to(destination.resolve());os.replace(link,latest)
    value['status']='linked';write(journal,value)
    return True


def paired_summary(root):
    outcomes={}
    for path in (root/'trials').glob('*.json'):
        record=read(path)
        if record.get('status')!='completed':continue
        case=record['case'];key=(case['condition'],case['seed'])
        outcomes.setdefault(key,{})[case['method']]=record['result']['success']
    result={}
    for method in ('H50','H20','lvm'):
        pairs=[v for v in outcomes.values() if method in v and 'ogg' in v]
        wins=sum(v['ogg'] and not v[method] for v in pairs);losses=sum(v[method] and not v['ogg'] for v in pairs)
        n=wins+losses
        exact=min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.
        result[method+'_vs_ogg']={'matched':len(pairs),'ogg_only':wins,'baseline_only':losses,
            'both_success':sum(v[method] and v['ogg'] for v in pairs),'both_failure':sum(not v[method] and not v['ogg'] for v in pairs),
            'exact_mcnemar_two_sided_unadjusted':exact}
    write(root/'analysis/paired_discordance.json',{'pairs':result,'scope':'same suite/new initializations; unadjusted exploratory comparisons; seeds/initializations may be dependent; not new-task generalization'})


class Guardian:
    def __init__(self,args):
        self.args=args;self.stopping=False;self.child_process=None;self.telemetry_failures=0
        self.state={'status':'running','pid':os.getpid(),'child_pid':None,
            'retries':read(args.output/'retry_counts.json') if (args.output/'retry_counts.json').exists() else {}}
    def update(self,**values):
        self.state.update(values);write(self.args.output/'guardian_status.json',self.state)
        print(json.dumps(values,ensure_ascii=False),flush=True)
    def remaining(self):return max(0.,self.deadline-time.time())
    def stopped(self):return self.stopping or (self.args.output/'STOP').exists()
    def rebalance(self):
        args=self.args
        if shutil.disk_usage(args.data_root).free>=args.rebalance_below_gib*2**30:return
        if os.stat(args.data_root).st_dev==os.stat(args.output).st_dev:return
        for family,root in [('original',args.original),('actions',args.followup/'experiments')]:
            for run in sorted((root/'runs').glob('*')):
                latest=run/'latest.pt'
                if latest.is_symlink() or not latest.is_file():continue
                if shutil.disk_usage(args.output).free < latest.stat().st_size+(args.system_reserve_gib+1)*2**30:return
                target=args.output/'checkpoint_archive'/family/run.name/'latest.pt'
                if tier_checkpoint(run,target,args.output/'placements'/f'{family}_{run.name}.json'):
                    self.update(placement={'run':run.name,'destination':str(target),'lossless':True})
                if shutil.disk_usage(args.data_root).free>=args.rebalance_below_gib*2**30:return
    def monitor(self):
        self.rebalance()
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        record={'utc':datetime.now(timezone.utc).isoformat(),'remaining_hours':self.remaining()/3600,
            'data_free_gib':shutil.disk_usage(self.args.data_root).free/2**30,
            'system_free_gib':shutil.disk_usage(self.args.output).free/2**30,
            'available_ram_gib':mem.get('MemAvailable',0)/2**30,'load_average':os.getloadavg()}
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
            record['gpu']=r.stdout.strip();self.telemetry_failures=0
        except (OSError,subprocess.SubprocessError) as e:
            self.telemetry_failures+=1;record['gpu_query_error']=str(e)
        with (self.args.output/'resources.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        self.update(resources=record,remaining_hours=self.remaining()/3600)
        healthy=(record['data_free_gib']>=self.args.min_free_gib and record['system_free_gib']>=self.args.system_reserve_gib
                 and record['available_ram_gib']>=self.args.min_ram_gib and self.telemetry_failures<3)
        return healthy
    def child(self,cmd,label,stop_file=None):
        cmd=list(cmd)
        for option in ('--max-compute-hours','--max-wall-hours'):
            if option in cmd:
                index=cmd.index(option)+1;cmd[index]=str(min(float(cmd[index]),self.remaining()/3600))
        path=self.args.output/'logs'/f'{label}.log';path.parent.mkdir(exist_ok=True)
        self.update(phase=label,log=str(path))
        with path.open('a') as file:
            process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
            self.child_process=process;self.update(child_pid=process.pid)
            requested=False
            while process.poll() is None:
                healthy=self.monitor()
                if self.stopped() and not requested:
                    if stop_file is not None:
                        stop_file.parent.mkdir(parents=True,exist_ok=True)
                        stop_file.touch();write(self.args.output/'OWNED_EVAL_STOP.json',{'path':str(stop_file)})
                    else:
                        try:process.send_signal(signal.SIGTERM)
                        except ProcessLookupError:pass
                    requested=True;self.update(boundary_stop_requested=True)
                time.sleep(min(30,self.args.poll_seconds))
        self.child_process=None;self.update(child_pid=None,boundary_stop_requested=False)
        if requested:return False
        if process.returncode:
            tail=path.read_text(errors='replace')[-32768:]
            fatal=any(s in tail.lower() for s in ('modulenotfounderror','nonfinite','nan','protocol','协议','out of memory','valueerror','shape mismatch'))
            retryable=(process.returncode in (-9,-15,137) or 'resource temporarily unavailable' in tail.lower()) and not fatal
            if not retryable and label=='continue_frozen_pipeline':retryable=self.recover_interrupted_pipeline()
            count=self.state.get('retries',{}).get(label,0)
            if retryable and count<2 and self.remaining()>120:
                self.state.setdefault('retries',{})[label]=count+1
                write(self.args.output/'retry_counts.json',self.state['retries'])
                self.update(retry=label,attempt=count+1)
                time.sleep(60)
                while label=='continue_frozen_pipeline' and any((active(self.args.followup,'followup'),
                        active(self.args.original,'sweep'),active(self.args.followup/'experiments','sweep'))):
                    if self.remaining()<=0 or self.stopped():return False
                    self.monitor();time.sleep(self.args.poll_seconds)
                if self.remaining()<=0 or self.stopped():return False
                return self.child(cmd,label,stop_file)
            raise RuntimeError(f'{label}退出码{process.returncode}，保留现场：{path}')
        return True
    def recover_interrupted_pipeline(self):
        """仅已有完整latest的SIGKILL/SIGTERM中断；保存原错误证据，绝不改协议/超参。"""
        args=self.args
        if active(args.followup,'followup') or active(args.original,'sweep') or active(args.followup/'experiments','sweep'):return False
        recovered=False
        for role,root,code in [('original',args.original,args.original_code),('actions',args.followup/'experiments',args.followup_code)]:
            path=root/'sweep_status.json'
            if not path.exists():continue
            state=read(path);error=str(state.get('error','')).lower()
            if state.get('status')!='error':continue
            if not any(s in error for s in ('退出码-9','退出码-15','exit code 137','exitcode -9')):continue
            run=root/'runs'/state.get('job',{}).get('name','')
            if not (run/'latest.pt').is_file():continue
            reconstruct(code,root,max(self.remaining()/3600,.001),followup=False)
            key='recover_'+role;count=self.state['retries'].get(key,0)
            if count>=2:return False
            write(args.output/'recovery_events'/f'{key}_{count+1}.json',{'previous_state':state,'run':str(run),'latest_sha256':digest(run/'latest.pt')})
            state.update(status='paused',reason='guardian_recoverable_process_interruption')
            write(path,state);self.state['retries'][key]=count+1
            write(args.output/'retry_counts.json',self.state['retries']);recovered=True
        status_path=args.followup/'followup_status.json'
        if recovered and status_path.exists() and read(status_path).get('status')=='error':
            state=read(status_path)
            write(args.output/'recovery_events'/'followup_previous_error.json',state)
            state.update(status='paused',reason='guardian_recovered_nested_process_interruption');write(status_path,state)
        return recovered

    def select_models(self):
        args=self.args;selected=[]
        families=[('mlp',args.original,args.original_code,['mlp_s42']),
            ('transformer',args.original,args.original_code,['transformer-h1_s42','transformer-h3_s42']),
            ('action_sequence',args.followup/'experiments',args.followup_code,['transformer-a10-h1_s42','transformer-a10-h3_s42'])]
        for family,root,code,names in families:
            reconstruct(code,root,max(self.remaining()/3600,.001),followup=False)
            candidates=[]
            for name in names:
                run=root/'runs'/name;marker=read(run/'JOB_COMPLETE.json');sha=marker['checkpoint_sha256']
                if digest(run/'best_model.pt')!=sha:raise RuntimeError('best SHA发生变化')
                report=read(root/'reports'/name/sha[:16]/'summary.json')
                if report['checkpoint_sha256']!=sha:raise RuntimeError('离线报告SHA不同')
                candidates.append({'family':family,'run':name,'checkpoint':str(run/'best_model.pt'),
                    'sha256':sha,'code':str(code),'development_mse':report['metrics']['mse']})
            selected.append(min(candidates,key=lambda r:r['development_mse']))
        plan={'selected':selected,'initial_states':list(range(3,10)),'seeds':[1000,1001],
            'methods':['H50','H20','lvm','ogg'],'trials_per_model':560,
            'scope':'fixed selection by development MSE within each family; action-sequence conditioning has extra information'}
        path=args.output/'evaluation_plan.json'
        if path.exists() and read(path)!=plan:raise RuntimeError('追加评测选优计划不同，不覆盖原结果')
        write(path,plan);return selected
    def run(self):
        args=self.args
        plan_path=args.output/'guardian_plan.json'
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('mode','extend_hours')}
        if plan_path.exists():
            plan=read(plan_path)
            if plan['settings']!=settings or plan['source_sha256']!=digest(Path(__file__)):raise RuntimeError('监督计划/代码不同，保留现场')
        else:
            plan={'settings':settings,'source_sha256':digest(Path(__file__)),'created_utc':datetime.now(timezone.utc).isoformat(),
                  'deadline_unix':time.time()+args.hours*3600}
        if args.extend_hours:plan['deadline_unix']=max(time.time(),plan['deadline_unix'])+args.extend_hours*3600
        write(plan_path,plan);self.deadline=plan['deadline_unix']
        self.update(deadline_utc=datetime.fromtimestamp(self.deadline,timezone.utc).isoformat())
        while self.remaining()>0 and not self.stopped():
            if not self.monitor():
                self.update(phase='waiting_resources',reason='resource_limit; no new GPU task launched')
                time.sleep(args.poll_seconds);continue
            busy=(active(args.followup,'followup') or active(args.original,'sweep') or active(args.followup/'experiments','sweep'))
            if busy:
                self.update(phase='waiting_existing_pipeline');time.sleep(args.poll_seconds);continue
            status=read(args.followup/'followup_status.json')
            if status['status']=='completed':break
            if status['status']=='error':
                if self.recover_interrupted_pipeline():continue
                raise RuntimeError('既有调度器报告程序错误，需要查看原日志；不无限重启')
            if (args.followup/'STOP').exists():self.update(status='paused',reason='existing_operator_STOP');return
            cmd=reconstruct(args.followup_code,args.followup,self.remaining()/3600)
            if not self.child(cmd,'continue_frozen_pipeline'):
                self.update(status='paused',reason='resource_or_operator_stop');return
            if read(args.followup/'followup_status.json')['status']!='completed':
                paused=read(args.followup/'followup_status.json')
                if 'operator_STOP' in str(paused.get('reason','')):
                    self.update(status='paused',reason='existing_operator_STOP');return
                if self.remaining()<=0:break
                self.update(phase='pipeline_boundary_pause',reason=paused.get('reason'))
                time.sleep(args.poll_seconds);continue
        if self.remaining()<=0 or self.stopped():self.update(status='paused',reason='deadline_or_STOP');return
        self.update(phase='all_15_training_jobs_finished')
        for model in self.select_models():
            if self.remaining()<=0 or self.stopped() or not self.monitor():
                self.update(status='paused',reason='deadline_STOP_or_resource');return
            output=args.output/'evaluation'/model['family'];state=output/'execution.json'
            if state.exists() and read(state)['status']=='completed':paired_summary(output);continue
            stop=output/'STOP';owned=args.output/'OWNED_EVAL_STOP.json'
            if owned.exists() and read(owned).get('path')==str(stop):stop.unlink(missing_ok=True);owned.unlink()
            if stop.exists():self.update(status='paused',reason='evaluation_operator_STOP');return
            cmd=[str(args.runtime/'sim-venv/bin/python'),'-u',str(Path(model['code'])/'corrector_closed_loop_eval.py'),
                '--workdir',str(args.runtime),'--output',str(output),'--suite','libero_object','--task-ids',*map(str,range(10)),
                '--initial-state-ids',*map(str,range(3,10)),'--seeds','1000','1001','--methods','H50','H20','lvm','ogg',
                '--corrector-checkpoint',model['checkpoint'],'--source-root',str(args.runtime/'vla-corrector-reference'),
                '--control-implementation','reference','--video-policy','failures','--audit-mode','failures','--trace-mode','all',
                '--fps','20','--min-free-gib',str(args.system_reserve_gib),'--max-wall-hours',str(self.remaining()/3600),'--stop-file',str(stop)]
            if not self.child(cmd,'extra_'+model['family'],stop):self.update(status='paused',reason='evaluation_resource_or_STOP');return
            paired_summary(output)
            if read(state).get('errors'):raise RuntimeError('追加评测存在程序错误')
            if read(state)['status']!='completed':self.update(status='paused',reason='evaluation_budget');return
        self.update(status='completed',phase='finite_plan_finished',reason='no repeated filler work; instance not shut down')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','status','plan'])
    p.add_argument('--output',type=Path,default=Path('/root/corrector_supervision_36h'))
    p.add_argument('--data-root',type=Path,default=BASE)
    p.add_argument('--original',type=Path,default=BASE/'object10/night_20261109_001')
    p.add_argument('--original-code',type=Path,default=BASE/'night-code-1e80242789')
    p.add_argument('--followup',type=Path,default=BASE/'object10/action_followup_20261109_001')
    p.add_argument('--followup-code',type=Path,default=BASE/'action-code-9274415dd6')
    p.add_argument('--runtime',type=Path,default=BASE/'runtime')
    p.add_argument('--hours',type=float,default=36)
    p.add_argument('--extend-hours',type=float,default=0)
    p.add_argument('--min-free-gib',type=float,default=5)
    p.add_argument('--system-reserve-gib',type=float,default=8)
    p.add_argument('--rebalance-below-gib',type=float,default=10)
    p.add_argument('--min-ram-gib',type=float,default=4)
    p.add_argument('--poll-seconds',type=float,default=30)
    args=p.parse_args()
    if any(not math.isfinite(v) or v<=0 for v in (args.hours,args.min_free_gib,args.system_reserve_gib,args.min_ram_gib,args.poll_seconds,args.rebalance_below_gib)) or not math.isfinite(args.extend_hours) or args.extend_hours<0 or args.rebalance_below_gib<=args.min_free_gib:
        raise ValueError('预算/资源阈值无效')
    if args.mode=='status':
        state=read(args.output/'guardian_status.json')
        try:state['process_alive']=str(Path(__file__).resolve()) in Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode()
        except OSError:state['process_alive']=False
        for name,path in [('followup',args.followup/'followup_status.json'),('original',args.original/'sweep_status.json'),
                          ('actions',args.followup/'experiments/sweep_status.json')]:
            if path.exists():
                value=read(path);state[name]={k:value.get(k) for k in ('status','phase','job','reason','error')}
        print(json.dumps(state,ensure_ascii=False,indent=2));return
    if args.mode=='plan':
        print(json.dumps({'hours':args.hours,'existing_training_jobs':15,'additional_eval_models':3,'additional_trials':1680,
            'no_downloads':True,'immutable_checkpoint_tiering':'completed JOB latest only; SHA verified copy+atomic original-path symlink',
            'persistent_deadline':True,'no_auto_shutdown':True},ensure_ascii=False,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    with lock(args.output/'.guardian.lock'):
        guardian=Guardian(args)
        def stop(signum,frame):guardian.stopping=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:guardian.run()
        except BaseException as e:guardian.update(status='error',error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':main()
