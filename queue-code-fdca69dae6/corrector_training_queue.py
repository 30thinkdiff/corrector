"""等待完整特征→可恢复预测器训练→GPU报告→有限LIBERO配对评测。

只复用已安装环境/模型；不安装、不下载、不删除缓存、不停止外部任务。
同一output只允许一个队列。断开SSH后请用nohup/screen保留本进程。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DATA_SHA = '86958911c0f959db2bbbdb107eb3e17c5f9c798e'
SOURCE_SHA = '927a06da8f00d7df483459e9039219e076c846de'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


@contextmanager
def lock(path):
    with Path(path).open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'已有进程持有锁，不重复启动：{path}') from error
        yield


def writer_busy(archive):
    try:
        with lock(archive / '.feature-writer.lock'):
            return False
    except RuntimeError:
        return True


def manifest_progress(archive, manifest):
    """轮询只读发布标记；真正训练前由训练器逐文件SHA和完整清单再核验。"""
    expected = {r['episode']: r for r in manifest['records']}
    published = {int(p.parent.name.split('_')[1]) for p in (archive/'features').glob('episode_*/COMPLETE.json')}
    if published - expected.keys():
        raise ValueError('特征归档含不在冻结清单中的额外episode')
    for ep in published:
        meta = read(archive/'features'/f'episode_{ep}'/'metadata.json')
        if (meta['num_frames'], meta['split']) != (expected[ep]['frames'], expected[ep]['split']):
            raise ValueError(f'特征与清单不同：episode{ep}')
    return len(published), len(expected)


def training_command(args, mode, *, check_resume=True):
    cmd = [str(args.runtime/'venv/bin/python'), '-u', str(HERE/'corrector_train.py'), mode,
           '--features', str(args.archive/'features'), '--manifest', str(args.archive/'manifest.json'),
           '--source-root', str(args.source_root), '--output', str(args.output/'predictor'),
           '--device', 'cuda', '--profile', 'paper-width', '--implementation', 'student',
           '--split-mode', 'episode', '--batch-size', str(args.batch_size), '--accumulate', str(args.accumulate),
           '--loader-workers', str(getattr(args, 'effective_loader_workers', args.loader_workers)), '--pin-memory', '--cpu-threads', '4',
           '--k-step', '10', '--epochs', str(args.epochs), '--patience', str(args.patience),
           '--snapshot-every', str(args.snapshot_every), '--lr', '0.0003', '--weight-decay', '0.0001',
           '--loss-type', args.loss_type, '--seed', '42', '--stop-file', str(args.output/'TRAIN_STOP')]
    if check_resume and mode == 'fit' and (args.output/'predictor/protocol.json').exists():
        if not (args.output/'predictor/latest.pt').is_file():
            raise RuntimeError('已有训练协议但尚无完整latest；请核查日志，不覆盖或伪造恢复')
        cmd.append('--resume')
    return cmd


def select_loader_workers(requested, batch_size, shared_free):
    # z_t和delta_z各为[B,128,960] FP32；默认每worker预取2批，再留主进程及余量。
    needed = batch_size*128*960*4*2*(2*requested+2)*1.5
    return requested if shared_free >= needed else 0


def eval_command(args, checkpoint, output, *, smoke=False, hours=0):
    return [str(args.runtime/'sim-venv/bin/python'), '-u', str(HERE/'corrector_closed_loop_eval.py'),
            '--workdir', str(args.runtime), '--output', str(output), '--suite', 'libero_object',
            '--task-ids', *map(str, [0] if smoke else range(10)),
            '--initial-state-ids', *map(str, [0, 1] if smoke else range(args.eval_initials)),
            '--seeds', *map(str, [1000] if smoke else range(1000, 1000+args.eval_seeds)),
            '--methods', 'H50', 'H20', 'monitor', 'lvm', 'ogg',
            '--corrector-checkpoint', str(checkpoint), '--source-root', str(args.source_root),
            '--control-implementation', 'student', '--video-policy', 'all' if smoke else 'failures',
            '--audit-mode', 'failures', '--trace-mode', 'all', '--fps', '20',
            '--min-free-gib', str(args.min_free_gib), '--max-wall-hours', str(hours),
            '--stop-file', str(args.output/'STOP')]


class Queue:
    def __init__(self, args):
        self.args, self.stopping, self.compute_started = args, False, None
        self.state = {'status': 'running', 'phase': 'starting', 'pid': os.getpid()}

    def update(self, **values):
        self.state.update(values)
        write(self.args.output/'queue_status.json', self.state)
        print(json.dumps(values, ensure_ascii=False), flush=True)

    def stop_requested(self):
        return self.stopping or (self.args.output/'STOP').exists()

    def remaining(self):
        return max(0., self.args.max_compute_hours*3600 - (time.monotonic()-self.compute_started))

    def child(self, command, label, *, stop_path=None, seconds=None):
        """stdout/stderr落盘；时间上限与信号请求边界暂停，不SIGKILL GPU进程。"""
        path = self.args.output/'logs'/f'{label}.log'
        path.parent.mkdir(exist_ok=True)
        self.update(phase=label, log=str(path), command=command)
        began, requested = time.monotonic(), False
        with path.open('a') as file:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=file, stderr=subprocess.STDOUT,
                                       env=self.env, start_new_session=True)
            self.update(child_pid=process.pid)
            while process.poll() is None:
                if stop_path and not requested and (self.stop_requested() or
                        (seconds is not None and time.monotonic()-began >= seconds)):
                    stop_path.touch()
                    requested = True
                    self.update(boundary_stop_requested=True)
                time.sleep(min(self.args.poll_seconds, 5))
        self.update(child_pid=None, boundary_stop_requested=False)
        if process.returncode:
            raise RuntimeError(f'{label}退出码{process.returncode}；查看{path}')
        return requested

    def wait_features(self, manifest):
        owned = None
        log = None
        try:
            while True:
                if self.stop_requested():
                    if owned and owned.poll() is None:
                        (self.args.output/'STOP').touch()
                        owned.wait()  # 我们启动的消费者在轨迹边界停止。
                    return False
                done, total = manifest_progress(self.args.archive, manifest)
                if owned and owned.poll() not in (None, 0):
                    raise RuntimeError('本队列启动的消费者失败；查看logs/feature_watch.log')
                busy = writer_busy(self.args.archive)
                self.update(phase='waiting_features', published=done, expected=total,
                            feature_writer_active=busy)
                if done == total and not busy:
                    if owned:
                        owned.wait()
                    return True
                if done < total and not busy and (owned is None or owned.poll() is not None):
                    if (self.args.archive/'STOP').exists():
                        raise RuntimeError('archive/STOP存在，保留人工暂停；不会擅自删除')
                    # 原消费者30分钟无数据会正常退出；本队列的替补无限等待已下载本地文件。
                    command = [str(self.args.runtime/'venv/bin/python'), '-u', str(HERE/'corrector_stream_workbench.py'),
                               'run', '--archive', str(self.args.archive), '--workdir', str(self.args.workdir),
                               '--source-root', str(self.args.source_root), '--watch-local', '--idle-minutes', '0',
                               '--poll-seconds', str(self.args.poll_seconds), '--max-new-episodes', '0', '--minutes', '0',
                               '--min-free-gib', str(self.args.min_free_gib), '--stop-file', str(self.args.output/'STOP')]
                    if log is None:
                        log = (self.args.output/'logs/feature_watch.log').open('a')
                    owned = subprocess.Popen(command, env=self.env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.update(replacement_consumer_pid=owned.pid)
                time.sleep(self.args.poll_seconds)
        finally:
            if log:
                log.close()

    def wait_gpu(self):
        while not self.stop_requested():
            probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True, timeout=20)
            if not probe.stdout.strip():
                return True
            self.update(phase='waiting_gpu_release', gpu_pids=probe.stdout.strip().splitlines())
            time.sleep(self.args.poll_seconds)
        return False

    def run(self):
        args = self.args
        manifest = read(args.archive/'manifest.json')
        if (manifest['dataset'] != 'HuggingFaceVLA/libero' or manifest['revision'] != DATA_SHA
                or len(manifest['records']) != 454 or manifest['frames'] != 66984
                or sorted({r['task_id'] for r in manifest['records']}) != list(range(20, 30))):
            raise ValueError('此队列限定已核对的Object十任务454条/66984帧，拒绝替换成小缓存')
        head = subprocess.check_output(['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip()
        if head != SOURCE_SHA:
            raise ValueError('固定参考源码尚未准备好；本队列不会联网补仓库')
        self.env = dict(os.environ, HF_HOME=str(args.runtime/'hf'), HF_HUB_OFFLINE='1',
                        TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false',
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        # 只冻结实验设置和代码；轮询/每次启动预算允许调整，不能篡改训练或配对集合。
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in {'mode', 'max_compute_hours', 'train_hours', 'eval_hours', 'poll_seconds'}}
        config['manifest_sha256'] = digest(args.archive/'manifest.json')
        config['code_sha256'] = {p.name: digest(p) for p in sorted(HERE.glob('*.py'))
                                 if p.name.startswith(('corrector_', 'smolvla_')) or p.name == 'batch_evaluation.py'}
        frozen = args.output/'queue_config.json'
        shared_free = shutil.disk_usage('/dev/shm').free
        args.effective_loader_workers = (read(frozen)['effective_loader_workers'] if frozen.exists() else
                                        select_loader_workers(args.loader_workers, args.batch_size, shared_free))
        if select_loader_workers(args.effective_loader_workers, args.batch_size, shared_free) != args.effective_loader_workers:
            raise RuntimeError('当前共享内存不足以复用原DataLoader协议；释放占用或使用新output')
        config['effective_loader_workers'] = args.effective_loader_workers
        self.update(resource_plan={'loader_workers_requested': args.loader_workers,
                    'loader_workers_selected': args.effective_loader_workers, 'shared_free_gib': shared_free/2**30,
                    'note': '共享内存不足时首跑用0 worker；batch和模型保持不变，选择写入冻结协议'})
        if frozen.exists() and read(frozen) != config:
            raise ValueError('已有队列设置/源码不同；请保持原部署包或使用新output')
        write(frozen, config)
        if not self.wait_features(manifest) or not self.wait_gpu():
            self.update(status='paused', reason='operator_stop_file')
            return
        self.compute_started = time.monotonic()
        if shutil.disk_usage(args.output).free < args.min_free_gib*2**30:
            raise RuntimeError('剩余磁盘小于安全余量；不删除已有数据')
        self.child(training_command(args, 'inspect'), 'inspect')
        if self.stop_requested() or self.remaining() <= 0:
            self.update(status='paused', reason='stop_or_compute_budget')
            return
        stop_path = args.output/'TRAIN_STOP'
        # 仅恢复队列自己上次预算/STOP请求留下的边界文件；人工用output/STOP控制整队列。
        if stop_path.exists():
            stop_path.unlink()
        self.child(training_command(args, 'fit'), 'training', stop_path=stop_path,
                   seconds=min(args.train_hours*3600, self.remaining()))
        if self.stop_requested():
            self.update(status='paused', reason='operator_stop_file')
            return
        result = args.output/'predictor/result.json'
        if not result.is_file() or not read(result).get('checkpoint_reload_verified'):
            raise RuntimeError('没有经过重载验证的最佳模型，不启动评测')
        checkpoint = args.output/'predictor/best_model.pt'
        sha = digest(checkpoint)
        report = args.output/'reports'/sha[:16]
        if self.remaining() <= 0:
            self.update(status='paused', reason='compute_budget', checkpoint_sha256=sha)
            return
        if not (report/'summary.json').exists():
            self.child([str(args.runtime/'venv/bin/python'), '-u', str(HERE/'corrector_prediction_report.py'),
                        'predict', '--run', str(args.output/'predictor'), '--features', str(args.archive/'features'),
                        '--source-root', str(args.source_root), '--checkpoint', str(checkpoint), '--output', str(report),
                        '--device', 'cuda', '--batch-size', str(args.batch_size),
                        '--loader-workers', str(args.effective_loader_workers), '--cpu-threads', '4'], 'prediction_report')
        elif read(report/'summary.json')['checkpoint_sha256'] != sha:
            raise RuntimeError('报告权重SHA不一致')
        self.child([str(args.runtime/'venv/bin/python'), str(HERE/'corrector_prediction_report.py'),
                    'plot', '--output', str(report)], 'plots')
        if args.no_eval:
            self.update(status='completed', phase='offline_report_ready', checkpoint_sha256=sha)
            return
        eval_started = time.monotonic()
        for smoke in (True, False):
            if self.stop_requested() or self.remaining() <= 0:
                self.update(status='paused', reason='stop_or_compute_budget')
                return
            available = min(self.remaining(), args.eval_hours*3600 - (time.monotonic()-eval_started))
            if available <= 0:
                self.update(status='paused', reason='evaluation_budget')
                return
            output = args.output/'evaluation'/sha[:16]/('smoke' if smoke else 'paired')
            self.child(eval_command(args, checkpoint, output, smoke=smoke, hours=available/3600),
                       'smoke' if smoke else 'paired', stop_path=args.output/'STOP')
            execution = read(output/'execution.json')
            if execution.get('errors', 0):
                raise RuntimeError('仿真出现程序错误，保留日志并停止后续阶段；机器人任务失败不算程序错误')
            if execution['status'] != 'completed':
                self.update(status='paused', reason=execution.get('reason', 'evaluation_paused'))
                return
        self.update(status='completed', phase='all_stages_finished', checkpoint_sha256=sha,
                    training_status=read(args.output/'predictor/training_status.json'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'plan', 'status'])
    base = Path('/root/autodl-tmp/corrector')
    p.add_argument('--archive', type=Path, default=base/'object10')
    p.add_argument('--workdir', type=Path, default=base/'object10-work')
    p.add_argument('--runtime', type=Path, default=base/'runtime')
    p.add_argument('--source-root', type=Path, default=base/'runtime/vla-corrector-reference')
    p.add_argument('--output', type=Path, default=base/'object10/queue_4090_001')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--loss-type', choices=['cosine', 'both', 'mse'], default='both')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulate', type=int, default=16)
    p.add_argument('--loader-workers', type=int, default=2)
    p.add_argument('--snapshot-every', type=int, default=25, help='0关闭额外快照；best/latest仍保留')
    p.add_argument('--train-hours', type=float, default=4)
    p.add_argument('--eval-hours', type=float, default=2)
    p.add_argument('--max-compute-hours', type=float, default=6, help='从完整特征就绪后计时；预算在epoch/回合边界生效')
    p.add_argument('--eval-initials', type=int, default=3)
    p.add_argument('--eval-seeds', type=int, default=2)
    p.add_argument('--no-eval', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--min-free-gib', type=float, default=5)
    return p


def main():
    args = parser().parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.train_hours, args.eval_hours, args.max_compute_hours, args.poll_seconds, args.min_free_gib)):
        raise ValueError('时间/轮询/磁盘余量须为正有限数')
    if (min(args.epochs, args.patience, args.batch_size, args.accumulate, args.eval_initials, args.eval_seeds) < 1
            or args.loader_workers < 0 or args.snapshot_every < 0 or args.eval_initials > 50):
        raise ValueError('训练/评测参数范围错误')
    if args.mode == 'status':
        print(json.dumps(read(args.output/'queue_status.json'), ensure_ascii=False, indent=2))
        return
    if args.mode == 'plan':
        print(json.dumps({'training_command': training_command(args, 'fit', check_resume=False),
                          'data': 'Object10: 454 episodes / 66984 frames; wait for complete cache',
                          'smoke_trials': 10, 'paired_trials': 10*args.eval_initials*args.eval_seeds*5,
                          'compute_hours': args.max_compute_hours, 'wait_not_in_compute_budget': True,
                          'notes': 'early stopping may finish early; no shutdown, download or auto expansion'},
                         ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'logs').mkdir(exist_ok=True)
    with lock(args.output/'.queue.lock'):
        queue = Queue(args)
        def stop(signum, frame):
            queue.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            queue.run()
        except BaseException as error:
            queue.update(status='error', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
