"""在本地终端测SSH上传吞吐；远端仅丢弃数据，不写实验文件、不启动GPU。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-host', required=True, help='本地SSH配置中的远程别名')
    parser.add_argument('--file', type=Path, required=True, help='本地已有的真实分片')
    parser.add_argument('--sample-mib', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=60, help='认证后上传超时')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.ssh_host.startswith('-') or any(c.isspace() for c in args.ssh_host):
        parser.error('SSH别名不能以短横线开头或含空白')
    if not args.file.is_file() or args.sample_mib <= 0 or args.seconds <= 0:
        parser.error('需要存在的分片文件、正的样本大小和超时')
    expected = min(args.file.stat().st_size, args.sample_mib * 2**20)
    if expected == 0:
        parser.error('不能使用空文件测速')
    # 禁用SSH压缩，使用真实Parquet字节，避免可压缩填充数据产生虚假速度。
    command = ['ssh', '-T', '-o', 'ConnectTimeout=10', '-o', 'Compression=no',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
               args.ssh_host, "printf 'CORRECTOR_UPLOAD_READY\\n'; cat > /dev/null"]
    print('请在此终端完成SSH身份验证；上传计时从远端READY之后开始。', flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               bufsize=0)
    result = {'expected_bytes': expected, 'status': 'error', 'remote_writes': False}
    timer = None
    try:
        deadline = time.monotonic() + 90
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([process.stdout], [], [], left)[0]:
                raise TimeoutError('90秒内未完成SSH认证/收到READY')
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('SSH在READY前退出；检查终端中的认证/连接错误')
            if line.strip() == b'CORRECTOR_UPLOAD_READY':
                break
        timer = threading.Timer(args.seconds, process.kill)
        timer.start()
        began = time.perf_counter()
        submitted = 0
        with args.file.open('rb') as source:
            while submitted < expected:
                block = source.read(min(2**20, expected - submitted))
                if not block:
                    raise RuntimeError('读取期间本地文件被截短')
                view = memoryview(block)
                while view:
                    count = process.stdin.write(view)
                    if not count:
                        raise BrokenPipeError('SSH输入提前关闭')
                    submitted += count
                    view = view[count:]
        process.stdin.close()
        code = process.wait()
        elapsed = time.perf_counter() - began
        if code != 0:
            raise RuntimeError(f'SSH退出码{code}，未确认全部传输完成')
        result.update(status='completed', bytes=expected, seconds=elapsed,
                      mib_per_second=expected / elapsed / 2**20)
    except (OSError, RuntimeError, TimeoutError) as error:
        result['error'] = str(error)
    finally:
        if timer is not None:
            timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
