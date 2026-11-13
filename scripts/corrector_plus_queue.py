"""Wait for ordinary evaluation/assets, then run immutable Plus adapter.

STOP requests propagate at episode boundaries. Resume reruns the same frozen
plan and verifies completed artifacts. Unknown failures stop, never blind retry.
"""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from corrector_j36m_object_eval import read,write,digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ordinary',type=Path,default=Path('/root/corrector_expanded_object_20261113_001'))
    p.add_argument('--assets',type=Path,default=Path('/root/autodl-tmp/corrector/plus_resources_20261113_001'))
    p.add_argument('--python',default='/root/autodl-tmp/corrector/runtime/sim-venv/bin/python')
    p.add_argument('--hours',type=float,default=18)
    args=p.parse_args()
    if not 0<args.hours<=24:raise ValueError('hours must be 0..24')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=args.output/'STOP';script=Path(__file__).with_name('corrector_expanded_eval.py')
    command=[args.python,'-u',str(script),'plus','--output',str(args.output/'results'),'--hours',str(args.hours)]
    frozen={'command':command,'adapter_sha256':digest(script),'queue_sha256':digest(__file__),
            'conditions_sha256':digest(script.with_name('plus_conditions.json')),'resume':'same source/plan; completed trials verified'}
    path=args.output/'queue_plan.json'
    if path.exists() and read(path)!=frozen:raise ValueError('queue plan/source differs; choose new output')
    if not path.exists():write(path,frozen)
    state={'status':'running','phase':'waiting_ordinary_and_assets','pid':os.getpid(),'child_pid':None}
    signal.signal(signal.SIGTERM,lambda *_:stop.touch());signal.signal(signal.SIGINT,lambda *_:stop.touch())
    child=None
    try:
        while True:
            if stop.exists():state.update(status='paused',reason='STOP');return
            ordinary=read(args.ordinary/'status.json');assets=read(args.assets/'status.json')
            state.update(ordinary_status=ordinary['status'],asset_status=assets['status']);write(args.output/'queue_status.json',state)
            if ordinary['status'] in ['error','paused']:raise RuntimeError('ordinary phase not completed; inspect original status')
            if assets['status']=='error':raise RuntimeError('asset preparation failed')
            if ordinary['status']=='completed' and assets['status']=='ready':break
            time.sleep(15)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if gpu.strip():raise RuntimeError('unowned GPU process; stop before Plus')
        state['phase']='plus_evaluation'
        with (args.output/'worker.log').open('ab') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=child.pid;write(args.output/'queue_status.json',state)
        while child.poll() is None:
            if stop.exists():(args.output/'results/STOP').touch()
            write(args.output/'queue_status.json',state);time.sleep(10)
        state.update(child_pid=None,returncode=child.returncode)
        result=read(args.output/'results/status.json')
        state.update(status=result['status'],phase=result['phase'],error=result.get('error'))
        if child.returncode and result['status']!='paused':raise RuntimeError('Plus failed; inspect worker.log, no automatic retry')
    except BaseException as e:
        state.update(status='error',error=f'{type(e).__name__}: {e}')
        if child and child.poll() is None:(args.output/'results/STOP').touch()
        raise
    finally:write(args.output/'queue_status.json',state)


if __name__=='__main__':main()
