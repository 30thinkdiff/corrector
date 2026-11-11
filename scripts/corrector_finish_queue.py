"""接续已有最佳模型的绘图/闭环；复用原冻结部署，绝不启动训练或安装依赖。

独立上传到runtime，不能覆盖原queue-code的文件；使用bootstrap Python执行。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import time


def original_queue(code):
    spec = importlib.util.spec_from_file_location('_frozen_corrector_queue', code/'corrector_training_queue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_args(module, output, frozen):
    args = module.parser().parse_args(['run'])
    for key, value in frozen.items():
        if hasattr(args, key):
            previous = getattr(args, key)
            setattr(args, key, Path(value) if isinstance(previous, Path) else value)
    if args.output.resolve() != output.resolve():
        raise ValueError('冻结输出目录与请求目录不一致')
    args.effective_loader_workers = frozen['effective_loader_workers']
    return args


def validate(module, code, args, frozen):
    """只读已有结果；不把文件存在当成经过训练的证据。"""
    for name, sha in frozen['code_sha256'].items():
        if Path(name).name != name or module.digest(code/name) != sha:
            raise RuntimeError(f'原冻结部署已改变：{name}；不要用新版文件覆盖旧目录')
    if module.digest(args.archive/'manifest.json') != frozen['manifest_sha256']:
        raise RuntimeError('数据清单与原队列不同')
    result = module.read(args.output/'predictor/result.json')
    status = module.read(args.output/'predictor/training_status.json')
    if (result.get('checkpoint_reload_verified') is not True or result.get('optimizer_steps', 0) < 1
            or result.get('epochs_completed', 0) < 1 or status['status'] not in {'completed', 'paused'}):
        raise RuntimeError('缺少真实训练/最佳权重重载证据，拒绝跳过训练')
    checkpoint = args.output/'predictor/best_model.pt'
    sha = module.digest(checkpoint)
    report = args.output/'reports'/sha[:16]
    summary = module.read(report/'summary.json')
    if summary['checkpoint_sha256'] != sha or summary['sample_count'] < 1:
        raise RuntimeError('数值预测报告与当前best权重不一致')
    return checkpoint, report, sha, result, status


def inspect_run(output):
    """只输出关键证据，避免数百条episode/源文件摘要淹没报错。"""
    def optional(path):
        return json.loads(path.read_text()) if path.is_file() else {}
    result = optional(output/'predictor/result.json')
    status = optional(output/'predictor/training_status.json')
    history = optional(output/'predictor/history.json') or []
    initial = optional(output/'predictor/initial_validation.json')
    value = {'training_status': status,
             'training_result': {k: result.get(k) for k in
                ('epochs_completed', 'optimizer_steps', 'parameters', 'peak_cuda_gib',
                 'checkpoint_reload_verified', 'best_validation')},
             'split_samples': result.get('split', {}).get('samples'),
             'initial_validation': initial,
             'history_epochs': len(history)}
    if history:
        best = min(history, key=lambda row: row['validation']['loss'])
        seconds = sum(row['seconds'] for row in history)
        value.update(best_epoch=best['epoch'], epoch_compute_seconds=seconds,
                     mean_epoch_seconds=seconds/len(history), last_epoch=history[-1]['epoch'])
    summaries = sorted((output/'reports').glob('*/summary.json'), key=lambda p: p.stat().st_mtime)
    if summaries:
        summary = optional(summaries[-1])
        value['latest_prediction_report'] = {k: summary.get(k) for k in
            ('checkpoint_sha256', 'checkpoint_epoch', 'sample_count', 'metrics', 'task_macro_metrics', 'report_runtime')}
    log = output/'logs/plots.log'
    value['plots_log_tail'] = log.read_text(errors='replace').splitlines()[-20:] if log.is_file() else []
    print(json.dumps(value, ensure_ascii=False, indent=2))


def finish(args):
    module = original_queue(args.code_dir)
    frozen = module.read(args.output/'queue_config.json')
    saved = saved_args(module, args.output, frozen)
    with module.lock(args.output/'.queue.lock'):
        checkpoint, report, sha, result, training_status = validate(module, args.code_dir, saved, frozen)
        runner = module.Queue(saved)
        runner.env = dict(os.environ, HF_HOME=str(saved.runtime/'hf'), HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false',
                          MPLCONFIGDIR=str(saved.runtime/'matplotlib-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        runner.compute_started = time.monotonic()
        saved.max_compute_hours = args.eval_hours
        def stop(signum, frame):
            runner.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            runner.update(phase='resuming_after_training', checkpoint_sha256=sha,
                          epochs_completed=result['epochs_completed'], optimizer_steps=result['optimizer_steps'],
                          best_validation=result['best_validation'], training_status=training_status,
                          training_skipped='existing completed training and matching prediction report verified')
            if runner.stop_requested():
                runner.update(status='paused', reason='operator_stop_file')
                return
            # 使用已装matplotlib的sim-venv；原模型、训练、报告脚本保持字节不变。
            runner.child([str(saved.runtime/'sim-venv/bin/python'),
                          str(args.code_dir/'corrector_prediction_report.py'), 'plot', '--output', str(report)], 'plots')
            if args.plots_only or saved.no_eval:
                runner.update(status='completed', phase='offline_report_ready')
                return
            if not runner.wait_gpu():
                runner.update(status='paused', reason='operator_stop_file')
                return
            for smoke in (True, False):
                if runner.stop_requested() or runner.remaining() <= 0:
                    runner.update(status='paused', reason='stop_or_evaluation_budget')
                    return
                destination = args.output/'evaluation'/sha[:16]/('smoke' if smoke else 'paired')
                command = module.eval_command(saved, checkpoint, destination, smoke=smoke,
                                              hours=runner.remaining()/3600)
                runner.child(command, 'smoke' if smoke else 'paired', stop_path=args.output/'STOP')
                execution = module.read(destination/'execution.json')
                if execution.get('errors', 0):
                    raise RuntimeError('闭环程序错误；保留日志，不当作机器人任务失败')
                if execution['status'] != 'completed':
                    runner.update(status='paused', reason=execution.get('reason', 'evaluation_paused'))
                    return
            runner.update(status='completed', phase='all_stages_finished', checkpoint_sha256=sha,
                          training_status=training_status)
        except BaseException as error:
            runner.update(status='error', error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/root/autodl-tmp/corrector')
    parser.add_argument('--code-dir', type=Path, default=base/'queue-code-fdca69dae6')
    parser.add_argument('--output', type=Path, default=base/'object10/queue_4090_001')
    parser.add_argument('--eval-hours', type=float, default=2)
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--inspect', action='store_true', help='只读简短训练证据与绘图错误；不运行任何实验')
    args = parser.parse_args()
    import math
    if not math.isfinite(args.eval_hours) or args.eval_hours <= 0:
        parser.error('eval-hours应为正有限数')
    if args.inspect:
        inspect_run(args.output)
    else:
        finish(args)


if __name__ == '__main__':
    main()
