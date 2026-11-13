"""冻结清单的原始分片并发预下载；不加载模型，不提取特征，不训练。"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import time
from urllib.parse import urlsplit


def selected_files(manifest, local_dir, limit):
    files = {}
    for record in manifest['records']:
        for spec in record['files']:
            path = PurePosixPath(spec['path'])
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] != 'data' or path.suffix != '.parquet'):
                raise ValueError('清单包含不安全的原始分片路径')
            size = spec['bytes']
            if not isinstance(size, int) or size <= 0:
                raise ValueError('清单分片大小必须是正整数')
            if str(path) in files and files[str(path)] != size:
                raise ValueError('同一分片在清单中大小不一致')
            files[str(path)] = size
    chosen, new_count = [], 0
    root = local_dir.resolve()
    for name, size in files.items():
        path = local_dir / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('拒绝通过符号链接下载到原始目录之外')
        cached = path.is_file() and path.stat().st_size == size
        if not cached:
            if limit and new_count >= limit:
                continue
            new_count += 1
        chosen.append((name, size, cached))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--local-dir', type=Path, required=True,
                        help='现有提取workdir下的source目录')
    parser.add_argument('--proxy-url')
    parser.add_argument('--use-hf-token', action='store_true',
                        help='仅官方Hub：显式使用HF_TOKEN或已保存凭据；默认匿名，不接收明文token参数')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-new-files', type=int, default=3,
                        help='默认只测3个未完整缓存的分片；0才下载全部')
    parser.add_argument('--min-free-gib', type=float, default=5)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.max_new_files < 0 or args.min_free_gib < 0:
        parser.error('workers需为1～8，文件数量和磁盘余量不得为负')
    if args.proxy_url:
        proxy = urlsplit(args.proxy_url)
        if proxy.scheme not in {'http', 'https'} or not proxy.hostname:
            parser.error('需要HTTP(S)代理URL')
        os.environ.update(http_proxy=args.proxy_url, https_proxy=args.proxy_url,
                          HTTP_PROXY=args.proxy_url, HTTPS_PROXY=args.proxy_url,
                          no_proxy='localhost,127.0.0.1,::1', NO_PROXY='localhost,127.0.0.1,::1')
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    manifest = json.loads(args.manifest.read_text())
    if (manifest.get('dataset') != 'HuggingFaceVLA/libero'
            or not re.fullmatch('[0-9a-f]{40}', manifest.get('revision', ''))):
        parser.error('需要LIBERO完整固定revision的清单')
    args.local_dir.mkdir(parents=True, exist_ok=True)
    files = selected_files(manifest, args.local_dir, args.max_new_files)
    missing = sum(size for _, size, cached in files if not cached)
    # 同时保留全部待下载文件、部分文件的保守副本余量与用户保留空间。
    extra = args.workers * max((size for _, size, _ in files), default=0)
    if shutil.disk_usage(args.local_dir).free < missing + extra + args.min_free_gib * 2**30:
        parser.error('磁盘不足以容纳本次分片与预留空间；未启动下载')
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print(json.dumps({'status': 'stopping', 'reason': 'signal',
                          'note': '不再提交新文件，等待正在下载的文件结束'}), flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_file = args.stop_file or args.local_dir.parent / 'PREFETCH_STOP'
    began = time.perf_counter()
    print(json.dumps({'status': 'starting', 'workers': args.workers,
                      'selected_files': len(files), 'uncached_bytes_before': missing,
                      'revision': manifest['revision']}), flush=True)
    if stop_file.exists():
        print(json.dumps({'status': 'paused', 'reason': 'stop_file'}), flush=True)
        return 0
    # 多个SDK下载线程共用这个local_dir；勿与另一预下载/原始清理任务同时运行。
    from huggingface_hub import hf_hub_download
    if args.use_hf_token:
        endpoint = urlsplit(os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
        if endpoint.scheme != 'https' or endpoint.netloc != 'huggingface.co':
            parser.error('--use-hf-token仅用于https://huggingface.co；公开数据镜像保持匿名')
        from huggingface_hub import get_token
        if not get_token():
            parser.error('未找到HF_TOKEN或已保存的登录凭据；未启动下载，不回退匿名请求')

    def download(spec):
        name, size, cached = spec
        tick = time.perf_counter()
        file = Path(hf_hub_download(manifest['dataset'], name, repo_type='dataset',
                                   revision=manifest['revision'], local_dir=args.local_dir,
                                   token=True if args.use_hf_token else False))
        if file.stat().st_size != size:
            raise RuntimeError(f'分片大小不符：{name}')
        return {'file': name, 'bytes': size, 'existed_before': cached,
                'seconds': time.perf_counter() - tick}

    completed, errors, new_bytes = 0, 0, 0
    iterator = iter(files)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = set()
        while True:
            while not stop and not stop_file.exists() and len(active) < args.workers:
                spec = next(iterator, None)
                if spec is None:
                    break
                active.add(pool.submit(download, spec))
            if not active:
                break
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for task in done:
                try:
                    metrics = task.result()
                except Exception as error:
                    errors += 1
                    stop = True
                    print(json.dumps({'status': 'error', 'error_type': type(error).__name__,
                                      'note': '保留SDK缓存；停止提交新文件'}), flush=True)
                else:
                    completed += 1
                    if not metrics['existed_before']:
                        new_bytes += metrics['bytes']
                    print(json.dumps({'status': 'file_completed', **metrics}), flush=True)
    elapsed = time.perf_counter() - began
    status = 'error' if errors else ('completed' if completed == len(files) else 'paused')
    print(json.dumps({'status': status, 'completed_files': completed, 'errors': errors,
                      'wall_seconds': elapsed, 'new_completed_bytes': new_bytes,
                      'new_completed_mib_per_second': new_bytes / 2**20 / elapsed,
                      'rate_note': '完整新文件载荷/墙钟；若从partial恢复，会高于实际网络吞吐'}), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
