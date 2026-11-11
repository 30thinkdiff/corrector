"""修复已确认Errno24启动环境；不改冻结训练代码/协议/权重/截止时间。"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

BASE=Path('/root/autodl-tmp/corrector')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def raise_nofile(target=32768):
    before=resource.getrlimit(resource.RLIMIT_NOFILE);soft,hard=before
    if target<8192:raise ValueError('目标至少8192，覆盖454条轨迹与worker的映射开销')
    infinity=resource.RLIM_INFINITY
    if hard!=infinity and hard<target:
        try:resource.setrlimit(resource.RLIMIT_NOFILE,(min(target,max(soft,8192)),target))
        except (PermissionError,ValueError,OSError):pass
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    requested=target if hard==infinity else min(target,hard)
    if requested<8192:raise RuntimeError(f'容器硬上限仅{hard}，无法提高到8192；尚未改变实验状态，需要调整容器启动限制')
    if soft!=infinity and soft<requested:resource.setrlimit(resource.RLIMIT_NOFILE,(requested,hard))
    after=resource.getrlimit(resource.RLIMIT_NOFILE)
    child=json.loads(subprocess.check_output([sys.executable,'-c',
        'import resource,json;print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))'],text=True))
    if child[0]!=infinity and child[0]<8192:raise RuntimeError('子进程未继承足够的文件额度，不启动任务')
    return {'before':list(before),'after':list(after),'child':child,'minimum':8192,'desired':target}


def validate_code(code,plan):
    for name,expected in plan['code_sha256'].items():
        if sha(code/name)!=expected:raise RuntimeError('冻结代码SHA不同：'+str(code/name))


def prepare(root,guardian_code,limits):
    plan=read(root/'guardian_plan.json');settings=plan['settings']
    script=guardian_code/'corrector_guardian.py'
    if sha(script)!=plan['source_sha256']:raise RuntimeError('监督器并非原冻结版本，拒绝修改已有计划')
    spec=importlib.util.spec_from_file_location('_frozen_guardian_nofile',script)
    guardian=importlib.util.module_from_spec(spec);spec.loader.exec_module(guardian)
    watch=guardian_code/'guardian_watch.sh'
    if not watch.is_file():raise RuntimeError('缺少原监督器watch脚本')
    original,followup=Path(settings['original']),Path(settings['followup'])
    roots=[(original,'sweep'),(followup,'followup'),(followup/'experiments','sweep')]
    # 同时核验锁，避免在活动训练或等待中的监督器旁边改状态。
    with (root/'.guardian.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('监督器仍在运行；不修改状态、不重复启动')
    if any(guardian.active(p,kind) for p,kind in roots):raise RuntimeError('仍有前序父/子进程，拒绝并行恢复')
    for path in [root/'STOP',original/'STOP',followup/'STOP',followup/'experiments/STOP']:
        if path.exists():raise RuntimeError('存在用户STOP，保留不删除：'+str(path))
    original_plan=read(original/'sweep_plan.json');followup_plan=read(followup/'followup_plan.json')
    validate_code(Path(settings['original_code']),original_plan)
    validate_code(Path(settings['followup_code']),followup_plan)
    status_path=original/'sweep_status.json';state=read(status_path)
    receipt=root/'nofile_recovery.json'
    if state.get('status')=='error':
        log=Path(state.get('log',''))
        if not log.is_file():raise RuntimeError('缺少训练错误日志，未修改状态')
        tail=log.read_text(errors='replace')[-65536:]
        if '[Errno 24] Too many open files' not in tail:raise RuntimeError('不是已确认的文件描述符错误，不通用清除错误状态')
        job=state.get('job',{}).get('name')
        if not job or '_training' not in state.get('phase',''):raise RuntimeError('不是训练加载阶段，保留现场')
        run=original/'runs'/job
        if (run/'protocol.json').exists() and not (run/'latest.pt').is_file():
            raise RuntimeError('已有协议但无完整latest；需单独检查，未改状态')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        backup=root/'recovery_events'/('nofile_'+stamp)
        paths=[status_path,followup/'followup_status.json',root/'guardian_status.json']
        states=[(p,read(p)) for p in paths if p.exists()]
        for index,(path,value) in enumerate(states):write(backup/f'{index}_{path.name}',value)
        write(backup/'environment.json',{'limits':limits,'run':str(run),'training_log':str(log),
            'latest_sha256':sha(run/'latest.pt') if (run/'latest.pt').exists() else None,
            'guardian_deadline_unix':plan['deadline_unix']})
        for path,value in states:
            if value.get('status')=='error':
                value.update(status='paused',reason='confirmed_nofile_environment_fixed',error=None)
                write(path,value)
        write(receipt,{'status':'prepared','limits':limits,'run':str(run),'backup':str(backup),
                       'deadline_unchanged':True,'code_and_weights_unchanged':True})
    elif not receipt.exists():raise RuntimeError('原队列不是本次已确认错误；未修改状态')
    watch=guardian_code/'guardian_watch.sh'
    if not watch.is_file():raise RuntimeError('缺少原监督器watch脚本')
    args=[]
    for k,v in settings.items():
        flag='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v:args.append(flag)
        elif isinstance(v,list):args.extend([flag,*map(str,v)])
        elif v is not None:args.extend([flag,str(v)])
    return ['bash',str(watch),*args]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guardian-root',type=Path,default=Path('/root/corrector_supervision_36h'))
    parser.add_argument('--guardian-code',type=Path,default=BASE/'guardian-code-2263c95a33')
    parser.add_argument('--nofile',type=int,default=32768)
    args=parser.parse_args();args.guardian_root.mkdir(parents=True,exist_ok=True)
    with (args.guardian_root/'.nofile-recovery.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('已有恢复入口，不重复启动')
        limits=raise_nofile(args.nofile)
        print(json.dumps({'limits':limits},ensure_ascii=False),flush=True)
        command=prepare(args.guardian_root,args.guardian_code,limits)
        print(json.dumps({'status':'resume_original_guardian','deadline':'unchanged','command':command},ensure_ascii=False),flush=True)
        os.execvp(command[0],command)

if __name__=='__main__':main()
