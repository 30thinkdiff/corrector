"""等待并续完冻结9组夜间队列，再运行动作序列条件预测器；共用可恢复墙钟预算。"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from corrector_training_queue import read, write, digest, lock

HERE=Path(__file__).resolve().parent


def frozen_command(code, output, hours):
    """仅重建原冻结计划的CLI，不用当前新版训练器加载旧checkpoint。"""
    protocol=read(output/'sweep_plan.json')
    for name,sha in protocol['code_sha256'].items():
        if digest(code/name)!=sha: raise RuntimeError('前序冻结源码改变，拒绝续训：'+name)
    command=[sys.executable,'-u',str(code/'corrector_night_sweep.py'),'run']
    for key,value in protocol['settings'].items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value: command.append(flag)
        elif isinstance(value,list): command.extend([flag,*map(str,value)])
        elif value is not None: command.extend([flag,str(value)])
    return [*command,'--max-compute-hours',str(hours)]


def process_matches(pid, code):
    if not isinstance(pid,int) or pid<=0:return False
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (FileNotFoundError,PermissionError):return False
    return str(code.resolve())+'/' in command


def predecessor_busy(output, code):
    try:
        with lock(output/'.sweep.lock'): pass
    except RuntimeError:return True
    state=read(output/'sweep_status.json')
    return any(process_matches(state.get(k),code) for k in ('pid','child_pid'))


def predecessor_complete(output):
    state=read(output/'sweep_status.json')
    if state.get('status')!='completed':return False
    jobs=read(output/'sweep_plan.json')['jobs']
    if not all((output/'runs'/job['name']/'JOB_COMPLETE.json').is_file() for job in jobs):
        raise RuntimeError('前序称已完成，但缺少逐任务完成记录')
    return True


def new_command(args,hours):
    return [sys.executable,'-u',str(HERE/'corrector_night_sweep.py'),'run',
            '--archive',str(args.archive),'--runtime',str(args.runtime),'--source-root',str(args.source_root),
            '--output',str(args.output/'experiments'),'--models',*args.models,
            '--seeds',*map(str,args.seeds),'--eval-training-seeds',*map(str,args.eval_training_seeds),
            '--epochs',str(args.epochs),'--patience',str(args.patience),'--snapshot-every',str(args.snapshot_every),
            '--predictor-precision',args.predictor_precision,'--max-compute-hours',str(hours),
            '--min-free-gib',str(args.min_free_gib),*(['--no-eval'] if args.no_eval else [])]


class Followup:
    def __init__(self,args):
        self.args=args;self.started=time.monotonic();self.stopping=False
        self.state={'status':'running','pid':os.getpid(),'child_pid':None}

    def update(self,**values):
        self.state.update(values)
        write(self.args.output/'followup_status.json',self.state)
        print(json.dumps(values,ensure_ascii=False),flush=True)

    def remaining(self):return max(0.,self.args.max_compute_hours*3600-(time.monotonic()-self.started))
    def stop(self):return self.stopping or (self.args.output/'STOP').exists()

    def child(self,command,label):
        path=self.args.output/(label+'.log')
        self.update(phase=label,command=command,log=str(path))
        with path.open('a') as file:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
            self.update(child_pid=process.pid)
            requested=False
            while process.poll() is None:
                if self.stop() and not requested:
                    # 仅请求我们启动的调度器在epoch/回合边界暂停，不杀现有外部任务。
                    if label=='resume_original_queue' and not (self.args.predecessor_output/'STOP').exists():
                        write(self.args.output/'OWNED_PREDECESSOR_STOP.json',{'output':str(self.args.predecessor_output)})
                    try: process.send_signal(signal.SIGTERM)
                    except ProcessLookupError: pass
                    requested=True
                    self.update(boundary_stop_requested=True)
                time.sleep(min(5,self.args.poll_seconds))
        self.update(child_pid=None,boundary_stop_requested=False)
        marker=self.args.output/'OWNED_PREDECESSOR_STOP.json'
        if label=='resume_original_queue' and marker.exists() and not (self.args.predecessor_output/'STOP').exists():
            marker.unlink()  # 训练阶段只会留下TRAIN_STOP，不留下可能误认未来外部STOP的标记。
        if process.returncode:raise RuntimeError(f'{label}退出码{process.returncode}；查看{path}')

    def run(self):
        args=self.args
        plan={'predecessor_output':str(args.predecessor_output),'predecessor_code':str(args.predecessor_code),
              'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                          if k not in {'mode','max_compute_hours','poll_seconds'}},
              'code_sha256':{p.name:digest(p) for p in HERE.glob('*.py')}}
        frozen=args.output/'followup_plan.json'
        if frozen.exists() and read(frozen)!=plan:raise RuntimeError('后续计划已冻结，换设置请另起output')
        write(frozen,plan)
        self.update(budget_hours=args.max_compute_hours,budget_includes_predecessor_wait=True)
        while not self.stop() and self.remaining()>0:
            if predecessor_busy(args.predecessor_output,args.predecessor_code):
                self.update(phase='waiting_original_queue',remaining_hours=self.remaining()/3600)
                time.sleep(args.poll_seconds);continue
            if predecessor_complete(args.predecessor_output):break
            state=read(args.predecessor_output/'sweep_status.json')
            if state.get('status')=='error':raise RuntimeError('前序任务程序错误，保留现场，不能自动当作完成')
            owned=args.output/'OWNED_PREDECESSOR_STOP.json'
            if owned.exists() and read(owned).get('output')==str(args.predecessor_output):
                (args.predecessor_output/'STOP').unlink(missing_ok=True)
                owned.unlink()
            if (args.predecessor_output/'STOP').exists():
                self.update(status='paused',reason='predecessor_has_operator_STOP');return
            self.child(frozen_command(args.predecessor_code,args.predecessor_output,self.remaining()/3600),
                       'resume_original_queue')
            if not predecessor_complete(args.predecessor_output):
                self.update(status='paused',reason='original_queue_not_finished_within_budget');return
        if self.stop() or self.remaining()<=0:
            self.update(status='paused',reason='budget_or_STOP_before_action_experiments');return
        self.update(phase='original_all_jobs_completed')
        self.child(new_command(args,self.remaining()/3600),'action_experiments')
        state=read(args.output/'experiments/sweep_status.json')
        self.update(status=state['status'],phase='action_experiments_'+state['status'],
                    reason=state.get('reason'),remaining_hours=self.remaining()/3600)


def parser():
    base=Path('/root/autodl-tmp/corrector')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','plan','status'])
    p.add_argument('--predecessor-output',type=Path,default=base/'object10/night_20261109_001')
    p.add_argument('--predecessor-code',type=Path,default=base/'night-code-1e80242789')
    p.add_argument('--output',type=Path,default=base/'object10/action_followup_20261109_001')
    p.add_argument('--archive',type=Path,default=base/'object10')
    p.add_argument('--runtime',type=Path,default=base/'runtime')
    p.add_argument('--source-root',type=Path,default=base/'runtime/vla-corrector-reference')
    p.add_argument('--models',nargs='+',choices=['transformer-a10-h1','transformer-a10-h3'],
                   default=['transformer-a10-h1','transformer-a10-h3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--eval-training-seeds',nargs='+',type=int,default=[42])
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--snapshot-every',type=int,default=0)
    p.add_argument('--predictor-precision',choices=['float32','amp-bfloat16'],default='amp-bfloat16')
    p.add_argument('--max-compute-hours',type=float,default=20)
    p.add_argument('--min-free-gib',type=float,default=5)
    p.add_argument('--poll-seconds',type=float,default=30)
    p.add_argument('--no-eval',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if any(not math.isfinite(x) or x<=0 for x in (args.max_compute_hours,args.poll_seconds,args.min_free_gib)):
        raise ValueError('预算/轮询/剩余磁盘须正数')
    if args.mode=='status':
        state=read(args.output/'followup_status.json')
        summary={k:state.get(k) for k in ('status','phase','pid','child_pid','remaining_hours','reason','error','log')}
        summary['process_alive']=process_matches(state.get('pid'),HERE)
        for name,path in [('original_queue',args.predecessor_output/'sweep_status.json'),
                          ('action_queue',args.output/'experiments/sweep_status.json')]:
            if path.exists():
                value=read(path)
                summary[name]={k:value.get(k) for k in ('status','phase','job','reason','error','child_pid')}
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    if args.mode=='plan':
        print(json.dumps({'predecessor':'finish all original 9 jobs using frozen old code; never overwrite running code',
            'models':args.models,'seeds':args.seeds,'new_training_jobs':len(args.models)*len(args.seeds),
            'hours':args.max_compute_hours,'wait_counts_in_budget':True,'offline':True,
            'note':'budget may pause before new experiments start; resume same command'},ensure_ascii=False,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    with lock(args.output/'.followup.lock'):
        runner=Followup(args)
        def stop(signum,frame):runner.stopping=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:runner.run()
        except BaseException as error:
            runner.update(status='error',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
