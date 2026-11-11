"""临时原始分片→逐轨迹特征持久化；先测速，再按冻结清单扩大规模。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import sys


def configure_proxy(url):
    """在导入Hub/HTTPX前显式配置本次进程；不打印凭据、不改变系统设置。"""
    if not url:
        return
    import os
    import socket
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('--proxy-url需要HTTP(S)代理地址')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=5):
            pass
    except OSError as error:
        raise ConnectionError('本次指定的代理端口不可达；先恢复SSH隧道，不进入模型加载。') from error
    os.environ.update(http_proxy=url, https_proxy=url, HTTP_PROXY=url, HTTPS_PROXY=url,
                      no_proxy='localhost,127.0.0.1,::1', NO_PROXY='localhost,127.0.0.1,::1')
    print(f'本次进程代理已显式设置：{parsed.scheme}://{parsed.hostname}:{port}', flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corrector_data_workbench as data


class BudgetStop(Exception):
    """仅在轨迹边界正常结束，不伪造全部完成。"""


def release_source_files(source_root, remaining, known_files):
    """只删除冻结清单内且再无未完成轨迹使用的分片；失败轨迹仍保留其原始文件。"""
    source_root = source_root.resolve()
    released = 0
    for relative in known_files:
        if remaining[relative]:
            continue
        path = source_root / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not Path(relative).parts or Path(relative).parts[0] != 'data'
                or path.is_symlink() or not path.resolve().is_relative_to(source_root)):
            raise ValueError('拒绝清理非本次数据分片路径')
        if path.is_file():
            released += path.stat().st_size
            path.unlink()
    return released


def choose_records(rows, task_ids, limit, seed=42):
    import random
    records = []
    for row in rows:
        low, high = row['stats/task_index/min'][0], row['stats/task_index/max'][0]
        if low != high or int(low) not in task_ids:
            continue
        if len(row['tasks']) != 1:
            raise ValueError('需要单任务轨迹')
        records.append({'episode': int(row['episode_index']), 'frames': int(row['length']),
                        'task_id': int(low), 'task': row['tasks'][0]})
    # 小测速集合含813/818/825；扩大时仍保留相同划分，不因抽样规模变化重新分配。
    splits = {}
    for task in sorted(task_ids):
        pool = sorted(r['episode'] for r in records if r['task_id'] == task)
        if len(pool) < 2:
            raise ValueError(f'task{task}缺少可独立验证的轨迹')
        reserved = [825] if 825 in pool else []
        candidates = [e for e in pool if e not in {813, 818, *reserved}]
        random.Random(seed + task).shuffle(candidates)
        count = min(len(pool) - 1, max(1, round(len(pool) * .2)))
        validation = set(reserved + candidates[:count - len(reserved)])
        splits.update({e: 'validation' if e in validation else 'train' for e in pool})
    records.sort(key=lambda r: ({813: -3, 818: -2, 825: -1}.get(r['episode'], r['episode'])))
    records = records[:limit] if limit else records
    for row in records:
        row['split'] = splits[row['episode']]
    if {r['split'] for r in records} != {'train', 'validation'}:
        raise ValueError('所选子集缺少训练或验证轨迹；扩大limit，不自动打散帧')
    return records


class FooterResolver:
    """用实际Parquet页脚定位；发布的episode data/file_index已知不可靠。"""
    def __init__(self, files, fs, cache, reader=None):
        self.files, self.fs, self.cache = files, fs, cache
        self.ranges = json.loads(cache.read_text()) if cache.exists() else {}
        self.reader = reader

    def bounds(self, i):
        path = self.files[i]['path']
        if path not in self.ranges:
            if self.reader:
                low, high = self.reader(path)
            else:
                import pyarrow.parquet as pq
                remote = f'datasets/{data.DATA_ID}@{data.DATA_SHA}/{path}'
                with self.fs.open(remote, 'rb', block_size=65536) as file:
                    meta = pq.ParquetFile(file).metadata
                    column = meta.schema.names.index('episode_index')
                    stats = [meta.row_group(j).column(column).statistics for j in range(meta.num_row_groups)]
                    if not stats or any(s is None or not s.has_min_max for s in stats):
                        raise RuntimeError(f'页脚缺少episode范围：{path}')
                    low, high = min(int(s.min) for s in stats), max(int(s.max) for s in stats)
            self.ranges[path] = [low, high]
            data.write_json(self.cache, self.ranges)
        return self.ranges[path]

    def locate(self, episode):
        lo, hi = 0, len(self.files) - 1
        found = None
        while lo <= hi:
            mid = (lo + hi) // 2
            low, high = self.bounds(mid)
            if episode < low: hi = mid - 1
            elif episode > high: lo = mid + 1
            else: found = mid; break
        if found is None:
            raise RuntimeError(f'没有定位到episode{episode}；不回退到错误元信息索引')
        selected = [found]
        for direction in (-1, 1):
            i = found + direction
            while 0 <= i < len(self.files):
                low, high = self.bounds(i)
                if not low <= episode <= high: break
                selected.append(i); i += direction
        # 只是候选，下载后还须核对完整帧索引和长度；不能仅凭min/max宣布已定位。
        return [self.files[i] for i in sorted(selected)]


def make_manifest(args):
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq
    path = args.archive / 'manifest.json'
    selection = {'task_ids': args.task_ids, 'limit': args.limit, 'seed': args.seed}
    if path.exists():
        old = json.loads(path.read_text())
        if old['selection'] != selection or old['revision'] != data.DATA_SHA:
            raise RuntimeError('冻结清单不同，使用新archive；不覆盖已完成缓存')
        if old['estimated_feature_bytes']>args.feature_budget_gib*2**30:raise ValueError('已有清单超出当前显式特征预算')
        return old
    if getattr(args,'manifest_template',None):
        template=json.loads(args.manifest_template.read_text())
        if (template['dataset']!=data.DATA_ID or template['revision']!=data.DATA_SHA
                or template['selection']!=selection):raise ValueError('预核验清单与所选数据版本/范围不一致')
        if template['estimated_feature_bytes']>args.feature_budget_gib*2**30:raise ValueError('预核验清单超出特征预算')
        for r in template['records']:
            for f in r['files']:
                candidate=Path(f['path'])
                if candidate.is_absolute() or '..' in candidate.parts or not candidate.parts or candidate.parts[0]!='data' or f['bytes']<=0:
                    raise ValueError('清单分片路径/大小不合法')
        data.write_json(path,template)
        return template
    began = time.perf_counter()
    info = HfApi().dataset_info(data.DATA_ID, revision=data.DATA_SHA, files_metadata=True, token=False)
    files = sorted([{'path': f.rfilename, 'bytes': f.size} for f in info.siblings
                    if f.rfilename.startswith('data/') and f.rfilename.endswith('.parquet')], key=lambda r: r['path'])
    if not files or any(f['bytes'] is None for f in files): raise RuntimeError('缺少实际分片大小')
    rows = []
    columns = ['episode_index','length','tasks','stats/task_index/min','stats/task_index/max']
    for f in info.siblings:
        if f.rfilename.startswith('meta/episodes/') and f.rfilename.endswith('.parquet'):
            local = hf_hub_download(data.DATA_ID, f.rfilename, repo_type='dataset', revision=data.DATA_SHA,
                                    local_dir=args.workdir / 'metadata', token=False)
            rows.extend(pq.read_table(local, columns=columns).to_pylist())
    records = choose_records(rows, set(args.task_ids), args.limit, args.seed)
    resolver = FooterResolver(files, HfFileSystem(token=False), args.archive / 'footer_ranges.json')
    for r in records:
        r['files'] = resolver.locate(r['episode'])
        print('定位', r['episode'], r['frames'], [f['path'] for f in r['files']], flush=True)
    frames = sum(r['frames'] for r in records)
    # 以已实测的双相机L=128/D=960量化布局估算，不含原图和checkpoint。
    estimate = frames * (128 * 960 + 128 * 2 + 7 * 2 + 7 * 4 + 4 + 8 + 8)
    if estimate > args.feature_budget_gib * 2**30:
        raise RuntimeError('预计特征超出显式feature-budget-gib，缩小清单或提高已核对预算')
    unique = {f['path']: f['bytes'] for r in records for f in r['files']}
    manifest = {'format': 1, 'dataset': data.DATA_ID, 'revision': data.DATA_SHA, 'selection': selection,
                'records': records, 'frames': frames, 'physical_download_bytes': sum(unique.values()),
                'estimated_feature_bytes': estimate, 'fps': 10,
                'split_scope': 'episode train/validation development split; no independent predictor test set'}
    data.write_json(path, manifest)
    data.write_json(args.archive / 'plan_timing.json', {'seconds': time.perf_counter()-began})
    return manifest


def validate_rows(rows, spec):
    rows = sorted(rows, key=lambda r: r['frame_index'])
    if (len(rows) != spec['frames'] or [r['frame_index'] for r in rows] != list(range(spec['frames']))
            or any(r['episode_index'] != spec['episode'] or r['task_index'] != spec['task_id']
                   or len(r['action']) != 7 for r in rows)):
        raise RuntimeError(f'实际原始帧与冻结清单不一致：episode{spec["episode"]}')
    return rows


def run_stream(args, manifest):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    specs = {r['episode']: r for r in manifest['records']}
    pending = []
    for ep, spec in specs.items():
        final = args.archive / 'features' / f'episode_{ep}'
        if data.verified(final):
            meta = json.loads((final/'metadata.json').read_text())
            if (meta['policy_revision'] != data.POLICY_SHA or meta['corrector_revision'] != data.CORRECTOR_SHA
                    or meta['split'] != spec['split'] or meta['num_frames'] != spec['frames']):
                raise RuntimeError('已完成特征与清单/编码器不一致')
        else: pending.append(ep)
    # 计数包含下次才处理的轨迹：跨运行仍要用的共享分片不重复删除/下载。
    remaining = Counter(f['path'] for ep in pending for f in specs[ep]['files'])
    known_files = {f['path'] for r in manifest['records'] for f in r['files']}
    if args.max_new_episodes: pending = pending[:args.max_new_episodes]
    started = time.perf_counter()
    deadline = started + args.minutes * 60 if args.minutes else float('inf')
    report_path = args.archive / 'stream_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'episodes': {}, 'attempts': []}
    attempt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'selected': pending, 'downloads': [],
               'stream_script_sha256':data.digest(Path(__file__)),'extract_script_sha256':data.digest(Path(data.__file__)),
               'status':'running','phase':'loading_encoder'}
    report['attempts'].append(attempt)
    raw_root = args.workdir / 'raw'; raw_root.mkdir(parents=True, exist_ok=True)
    source_root = args.workdir / 'source'; source_root.mkdir(exist_ok=True)
    stop_file = getattr(args, 'stop_file', None) or args.archive / 'STOP'
    min_free = getattr(args, 'min_free_gib', 5.) * 2**30
    attempt['released_source_bytes'] = 0

    def persist(): data.write_json(report_path, report)

    def prepare_episode(ep):
        if stop_file.exists(): raise BudgetStop('operator_stop_file')
        if time.perf_counter() >= deadline: raise BudgetStop()
        spec = specs[ep]; began = time.perf_counter(); tables = []; origins = []
        attempt.update(phase='preparing_episode', current_episode=ep)
        persist()
        print(f'开始准备episode{ep}原始数据，已完成特征将按SHA跳过。', flush=True)
        # 单轨迹raw、特征构建/发布的短暂副本也占磁盘，预留空间后才下载。
        feature_bytes = spec['frames'] * (128 * 960 + 128 * 2 + 7 * 2 + 7 * 4 + 4 + 8 + 8)
        raw_bytes = sum(f['bytes'] for f in spec['files'])
        if (shutil.disk_usage(args.workdir).free < min_free + 2 * raw_bytes + 2 * feature_bytes
                or shutil.disk_usage(args.archive).free < min_free + feature_bytes):
            raise BudgetStop('disk_space')
        for f in spec['files']:
            if shutil.disk_usage(args.workdir).free < f['bytes'] + 256 * 2**20:
                raise RuntimeError('临时盘空间不足；不会把GPU显存当作文件存储')
            cached = (source_root / f['path']).is_file(); tick = time.perf_counter()
            local = Path(hf_hub_download(data.DATA_ID, f['path'], repo_type='dataset', revision=data.DATA_SHA,
                                       local_dir=source_root, token=False))
            attempt['downloads'].append({'file': f['path'], 'bytes': f['bytes'], 'cache_hit': cached,
                                         'seconds': time.perf_counter()-tick})
            persist()
            table = pq.read_table(local, filters=[('episode_index','=',ep)])
            if table.num_rows: tables.append(table)
            origins.append({'file': f['path'], 'sha256': data.digest(local)})
        if not tables: raise RuntimeError(f'候选分片没有episode{ep}')
        table = pa.concat_tables(tables); rows = validate_rows(table.to_pylist(), spec)
        local = args.workdir / 'raw_build' / f'episode_{ep}'; local.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows,schema=table.schema), local/'frames.parquet')
        data.write_json(local/'provenance.json', {'dataset':data.DATA_ID,'revision':data.DATA_SHA,
             'episode':ep,'split':spec['split'],'num_frames':spec['frames'],'fps':10,
             'task':spec['task'],'task_id':spec['task_id'],'sources':origins,'manifest_sha256':data.digest(args.archive/'manifest.json')})
        final = raw_root / f'episode_{ep}'
        if not data.verified(final):
            if final.exists(): raise RuntimeError('发现不完整原始目录，保留现场')
            data.publish(local,final)
        report['episodes'][str(ep)] = {'prepare_seconds':time.perf_counter()-began}
        persist()

    def completed(ep, metrics):
        if not data.verified(args.archive / 'features' / f'episode_{ep}'):
            raise RuntimeError('特征尚未校验发布，拒绝清理原始分片')
        report['episodes'][str(ep)].update(metrics, completed=True)
        persist()  # 完成特征及测速记录先在Drive落盘，再清临时原图。
        if args.keep_raw:
            final = args.archive/'raw'/f'episode_{ep}'
            if not data.verified(final):
                local=args.workdir/'raw_export'/f'episode_{ep}';local.mkdir(parents=True,exist_ok=True)
                for p in (raw_root/f'episode_{ep}').iterdir():
                    if p.name!='COMPLETE.json': shutil.copy2(p,local/p.name)
                final.parent.mkdir(parents=True,exist_ok=True);data.publish(local,final)
                shutil.rmtree(local)
        for root in [raw_root,args.workdir/'raw_build',args.workdir/'features']:
            p=root/f'episode_{ep}'
            if p.exists():shutil.rmtree(p)
        for f in specs[ep]['files']:
            remaining[f['path']] -= 1
        attempt['released_source_bytes'] += release_source_files(source_root, remaining, known_files)
        persist()

    options = argparse.Namespace(output_root=args.archive,work_root=args.workdir,raw_root=raw_root,
        episodes=pending,device='cuda',allow_cpu_extraction=False,allow_model_download=True,
        source_root=args.source_root,before_episode=prepare_episode,on_complete=completed)
    # 处理上次“特征已发布、清理前断连”留下的文件；不影响未完成轨迹。
    attempt['released_source_bytes'] += release_source_files(source_root, remaining, known_files)
    persist()
    try:
        if stop_file.exists():
            raise BudgetStop('operator_stop_file')
        print(f'本次选择{len(pending)}条未完成轨迹；先加载冻结视觉编码器，再下载/提取。', flush=True)
        data.extract(options)
        attempt['status']='finished_selected_batch'
        attempt['phase']='idle'
    except BudgetStop as error:
        attempt['status']='stopped_at_episode_boundary'
        attempt['stop_reason']=str(error) or 'time_budget'
    except BaseException as error:
        attempt['status']='interrupted_or_error';attempt['error']=f'{type(error).__name__}: {error}'
        persist();raise
    finally:
        attempt['wall_seconds']=time.perf_counter()-started;persist()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','run','status'])
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--task-ids',type=int,nargs='+',default=[24])
    p.add_argument('--limit',type=int,default=44)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--feature-budget-gib',type=float,default=4.)
    p.add_argument('--manifest-template',type=Path,help='使用已核对的公开清单，省去远程重复扫描页脚')
    p.add_argument('--max-new-episodes',type=int,default=0)
    p.add_argument('--minutes',type=float,default=45.)
    p.add_argument('--keep-raw',action='store_true')
    p.add_argument('--stop-file',type=Path,help='默认archive/STOP；完整轨迹边界暂停')
    p.add_argument('--min-free-gib',type=float,default=5.,help='临时盘及归档盘的磁盘保留量')
    p.add_argument('--proxy-url',help='显式为本次Python下载设置HTTP(S)代理；不会自动创建SSH隧道')
    args=p.parse_args()
    import math
    if (args.limit<0 or args.max_new_episodes<0 or args.minutes<0 or args.feature_budget_gib<=0
            or not all(math.isfinite(x) and x>=0 for x in (args.minutes,args.min_free_gib,args.feature_budget_gib))):
        p.error('规模/预算参数不合法')
    configure_proxy(args.proxy_url)
    import os
    if str(args.archive).startswith('/content/drive/') and not os.path.ismount('/content/drive'):
        raise RuntimeError('Drive未挂载')
    args.archive.mkdir(parents=True,exist_ok=True);args.workdir.mkdir(parents=True,exist_ok=True)
    if args.mode=='plan':
        manifest=make_manifest(args);print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2))
    else:
        manifest=json.loads((args.archive/'manifest.json').read_text())
        if manifest['revision']!=data.DATA_SHA:raise RuntimeError('数据版本不一致')
        if args.mode=='run':run_stream(args,manifest)
        else:
            count=sum(data.verified(args.archive/'features'/f'episode_{r["episode"]}') for r in manifest['records'])
            print('有效特征轨迹：',count,'/',len(manifest['records']))


if __name__=='__main__':main()
