"""AutoDL Object10下载出口恢复；只重启匹配的prefetch，保留数据和GPU消费者。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.parse import urlsplit


def option(argv, name):
    return argv[argv.index(name)+1] if name in argv and argv.index(name)+1 < len(argv) else None


def is_target(argv, root):
    return (any(Path(arg).name == 'corrector_prefetch.py' for arg in argv)
            and option(argv, '--manifest') == str(root/'object10/manifest.json')
            and option(argv, '--local-dir') == str(root/'object10-work/source')
            and option(argv, '--max-new-files') == '0')


def targets(root):
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        if is_target(argv, root):
            found[int(proc.name)] = argv
    return found


def stop_target(pid, argv, root, grace=15.):
    # 每次发信号前重新核对完整命令，避免使用过期PID去停止其他实验。
    if targets(root).get(pid) != argv:
        raise RuntimeError('下载进程已改变，请重新核对；未停止任何进程')
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic()+grace
    while time.monotonic() < deadline:
        if pid not in targets(root):
            return False
        time.sleep(.5)
    if targets(root).get(pid) == argv:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
    for _ in range(20):
        if pid not in targets(root):
            return True
        time.sleep(.1)
    raise RuntimeError('旧下载器尚未退出，不启动第二个写入者')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/corrector'))
    parser.add_argument('--proxy-url', default='http://127.0.0.1:17891')
    args = parser.parse_args()
    proxy = urlsplit(args.proxy_url)
    if proxy.scheme != 'http' or proxy.hostname != '127.0.0.1' or not proxy.port or proxy.username:
        parser.error('本恢复入口仅接受无凭据的本机回环HTTP代理')
    root = args.root
    python = root/'runtime/venv/bin/python'
    script = root/'training-code-netfix/corrector_prefetch.py'
    manifest = root/'object10/manifest.json'
    if not all(p.is_file() for p in (python, script, manifest)):
        parser.error('缺少本次Object10已部署环境；不重新安装或下载代码')
    if (root/'object10-work/PREFETCH_STOP').exists():
        parser.error('存在PREFETCH_STOP人工暂停标记，先确认暂停意图；不自动删除')
    # 单独HEAD验证新出口，失败时不碰原下载器。只传输头，不下载训练数据。
    url = ('https://huggingface.co/datasets/HuggingFaceVLA/libero/resolve/'
           '86958911c0f959db2bbbdb107eb3e17c5f9c798e/data/chunk-000/file-299.parquet')
    probe = subprocess.run(['curl','-sS','-I','--proxy',args.proxy_url,'--connect-timeout','10',
                            '--max-time','25','-o','/dev/null','-w','%{http_code}',url],
                           capture_output=True,text=True)
    if probe.returncode or not probe.stdout.startswith(('2','3')):
        parser.error('新代理未通过HF文件HEAD检查，未停止原任务；请先恢复17891隧道')
    with (root/'.prefetch-recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old = targets(root)
        if len(old) > 1:
            parser.error('发现多个全量下载器，需先人工核对；未停止任何进程')
        forced = False
        if old:
            pid, command = next(iter(old.items()))
            print(json.dumps({'action':'stop_prefetch_only','pid':pid,'grace_seconds':15}),flush=True)
            forced = stop_target(pid,command,root)
        else:
            command = [str(python),'-u',str(script),'--manifest',str(manifest),
                       '--local-dir',str(root/'object10-work/source'),'--workers','3',
                       '--max-new-files','0','--min-free-gib','5']
        if targets(root):
            raise RuntimeError('仍有全量下载器，不启动第二个写入者')
        if '--proxy-url' in command:
            command[command.index('--proxy-url')+1] = args.proxy_url
        else:
            command += ['--proxy-url',args.proxy_url]
        env=os.environ.copy();env['HF_HOME']=str(root/'runtime/hf')
        env.pop('HF_HUB_OFFLINE',None);env.pop('TRANSFORMERS_OFFLINE',None)
        log=root/'object10-prefetch.log'
        with log.open('ab',buffering=0) as output:
            marker={'status':'network_recovery','proxy_port':proxy.port,'old_prefetch_pids':list(old),
                    'forced_stop_after_grace':forced,'started_unix':time.time()}
            output.write((json.dumps(marker)+'\n').encode())
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=output,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'status':'prefetch_restarted','pid':process.pid,'proxy_port':proxy.port,
                          'log':str(log),'gpu_consumer_untouched':True,'cache_preserved':True}),flush=True)


if __name__ == '__main__':
    main()
