"""绝对截止、有限恢复和双盘资源保护；单GPU串行完成冻结扩展计划。"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

BASE=Path('/root/autodl-tmp/corrector')
ROOT=Path('/root/corrector_overnight_20261111')
HERE=Path(__file__).resolve().parent


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def resources():
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[0] in ['MemAvailable:']}
    available=min(info.get('MemAvailable',0),120*2**30);limit=None;usage=None
    for maximum,current in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                            ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            value=Path(maximum).read_text().strip()
            if value!='max':
                limit=int(value);usage=int(Path(current).read_text());available=min(available,max(0,limit-usage))
            break
        except (OSError,ValueError):pass
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    return {'utc':datetime.now(timezone.utc).isoformat(),'system_free_gib':shutil.disk_usage(ROOT).free/2**30,
        'data_free_gib':shutil.disk_usage(BASE).free/2**30,'available_ram_gib':available/2**30,
        'cgroup_limit_gib':limit/2**30 if limit is not None else None,'cgroup_used_gib':usage/2**30 if usage is not None else None,
        'effective_ram_ceiling_gib':120,
        'gpu':gpu.stdout.strip(),'gpu_query_error':gpu.stderr.strip() if gpu.returncode else None}


def run(args):
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(32768,hard) if hard!=-1 else 32768,hard))
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline<=datetime.now(timezone.utc):raise RuntimeError('截止时间已到，恢复时须显式指定新的截止时间')
    if (ROOT/'results/STOP').exists():raise RuntimeError('仍有STOP，确认暂停后显式移除才可续接')
    state={'status':'running','pid':os.getpid(),'child_pid':None,'deadline':deadline.isoformat(),'retries':0,'phase':'starting'}
    write(ROOT/'queue_status.json',state)
    def pause(*a):
        (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/STOP').touch()
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    from analyse_corrector_overnight import analyse
    try:
        while True:
            if datetime.now(timezone.utc)>=deadline:state.update(status='paused',reason='deadline');break
            env=os.environ.copy();env['CORRECTOR_OVERNIGHT_SUPERVISOR_PID']=str(os.getpid())
            cmd=[str(BASE/'runtime/sim-venv/bin/python'),'-u',str(HERE/'corrector_overnight_eval.py'),
                '--output',str(ROOT/'results'),'--deadline',deadline.isoformat()]
            if (ROOT/'results/plan.json').exists():cmd.append('--resume')
            with (ROOT/'worker.log').open('a') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                state.update(child_pid=child.pid,phase='expanded_trials_and_prediction_audit',command=cmd);write(ROOT/'queue_status.json',state)
                while child.poll() is None:
                    metrics=resources();state['resources']=metrics
                    with (ROOT/'resources.jsonl').open('a') as file:file.write(json.dumps(metrics)+'\n')
                    if metrics['system_free_gib']<5 or metrics['data_free_gib']<3 or metrics['available_ram_gib']<6:
                        state['resource_pause_requested']=True;pause()
                    path=ROOT/'results/execution.json'
                    if path.exists():state['execution']=json.loads(path.read_text())
                    if (ROOT/'results/trials').is_dir():analyse(ROOT/'results')
                    write(ROOT/'queue_status.json',state)
                    # Sleeps in short intervals so terminal signals remain responsive.
                    for _ in range(6):
                        if child.poll() is not None:break
                        time.sleep(5)
            state['child_pid']=None
            if child.returncode==0:
                result=json.loads((ROOT/'results/pipeline_status.json').read_text());state.update(status=result['status'],phase='finished',result=result);break
            if child.returncode in [-9,-15,137,143] and state['retries']<2 and not (ROOT/'results/STOP').exists():
                state['retries']+=1;write(ROOT/'queue_status.json',state);time.sleep(5);continue
            raise RuntimeError(f'worker退出{child.returncode}；保留现场，不循环重启确定性错误，查看{ROOT}/worker.log')
        if (ROOT/'results/trials').is_dir():analyse(ROOT/'results')
    except BaseException as error:
        state.update(status='error',error=f'{type(error).__name__}: {error}')
        # A supervisor failure requests a boundary stop rather than abandoning a live worker.
        pause()
        raise
    finally:write(ROOT/'queue_status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',required=True);args=p.parse_args();run(args)
