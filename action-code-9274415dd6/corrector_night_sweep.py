"""离线12小时研究队列：MLP/Transformer H1/H3、多种子、最终best报告与闭环。

复用已安装环境和454条缓存；不下载、不清理数据、不关机。旧实验目录不变。
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import statistics
import time

from corrector_training_queue import Queue, lock, read, write, digest, SOURCE_SHA

HERE = Path(__file__).resolve().parent


def specifications(args):
    variants = {'mlp': ('mlp', 1), 'transformer-h1': ('transformer', 1), 'transformer-h3': ('transformer', 3),
                'transformer-a10-h1': ('transformer-action', 1), 'transformer-a10-h3': ('transformer-action', 3)}
    return [{'name': f'{variant}_s{seed}', 'variant': variant, 'model_type': variants[variant][0],
             'h_window': variants[variant][1], 'seed': seed}
            for seed in args.seeds for variant in args.models]


def fit_command(args, job, resource, mode):
    output = args.output/'runs'/job['name']
    if mode == 'probe': output = args.output/'probes'/f"{job['variant']}_b{resource['batch']}"
    cmd = [str(args.runtime/'venv/bin/python'), '-u', str(HERE/'corrector_train.py'), mode,
           '--features', str(args.archive/'features'), '--manifest', str(args.archive/'manifest.json'),
           '--source-root', str(args.source_root), '--output', str(output), '--device', 'cuda',
           '--profile', 'paper-width', '--model-type', job['model_type'], '--h-window', str(job['h_window']),
           '--anchor-history', '3', '--transformer-dim', '768', '--transformer-heads', '12', '--transformer-layers', '5',
           '--implementation', 'reference', '--split-mode', 'episode', '--seed', str(job['seed']),
           '--predictor-precision', args.predictor_precision,
           '--batch-size', str(resource['batch']), '--accumulate', str(512//resource['batch']),
           '--loader-workers', str(resource['workers']), '--pin-memory', '--cpu-threads', '4',
           '--k-step', '10', '--epochs', str(args.epochs), '--patience', str(args.patience),
           '--lr', '0.0003', '--weight-decay', '0.0001', '--loss-type', 'both',
           '--snapshot-every', str(args.snapshot_every), '--stop-file', str(output/'TRAIN_STOP')]
    if job['model_type']=='transformer-action': cmd += ['--action-steps','10']
    if mode == 'fit' and (output/'protocol.json').is_file():
        if not (output/'latest.pt').is_file():
            raise RuntimeError(f'{output}有协议但没有完整latest，保留现场，不覆盖')
        cmd.append('--resume')
    return cmd


def evaluation_command(args, job, checkpoint, output, hours, smoke=False):
    cmd = [str(args.runtime/'sim-venv/bin/python'), '-u', str(HERE/'corrector_closed_loop_eval.py'),
           '--workdir', str(args.runtime), '--output', str(output), '--suite', 'libero_object',
           '--task-ids', *map(str, [0] if smoke else range(10)),
           '--initial-state-ids', *map(str, [0] if smoke else range(args.eval_initials)),
           '--seeds', *map(str, [1000] if smoke else args.eval_random_seeds),
           '--methods', *(['ogg'] if smoke else args.methods),
           '--corrector-checkpoint', str(checkpoint), '--source-root', str(args.source_root),
           '--control-implementation', 'reference', '--video-policy', 'all' if smoke else 'failures',
           '--audit-mode', 'failures', '--trace-mode', 'all', '--fps', '20',
           '--max-wall-hours', str(hours), '--min-free-gib', str(args.min_free_gib),
           '--stop-file', str(args.output/'STOP')]
    if smoke:
        cmd += ['--max-steps', '60', '--debug-force-step', '24']
    return cmd


def complete_training(output):
    status, result = output/'training_status.json', output/'result.json'
    return (status.is_file() and result.is_file() and read(status)['status'] == 'completed'
            and read(result).get('checkpoint_reload_verified') is True)


class Sweep(Queue):
    def update(self, **values):
        self.state.update(values)
        write(self.args.output/'sweep_status.json', self.state)
        print(json.dumps(values, ensure_ascii=False), flush=True)

    def available(self):
        return not self.stop_requested() and self.remaining() > 0

    def summarize(self):
        rows, index_hashes = [], set()
        for job in specifications(self.args):
            output = self.args.output/'runs'/job['name']
            if not complete_training(output): continue
            checkpoint = output/'best_model.pt'
            path = self.args.output/'reports'/job['name']/digest(checkpoint)[:16]/'summary.json'
            if not path.is_file(): continue
            report, result = read(path), read(output/'result.json')
            index_hashes.add(json.dumps(report['split']['index_sha256'],sort_keys=True))
            rows.append({**job, 'parameters':result['parameters'],'epochs_completed':result['epochs_completed'],
                         'best_epoch':report['checkpoint_epoch'],'cosine':report['metrics']['cosine'],
                         'mse':report['metrics']['mse'],'zero_mse':report['metrics']['zero_mse'],
                         'validation_pairs':report['sample_count'],'checkpoint_sha256':report['checkpoint_sha256']})
        if len(index_hashes)>1: raise RuntimeError('不同模型未使用相同有效锚点，拒绝混为架构对照')
        if rows:
            with (self.args.output/'comparison.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        groups=[]
        for variant in self.args.models:
            subset=[r for r in rows if r['variant']==variant]
            if not subset: continue
            groups.append({'variant':variant,'n_seeds':len(subset),
                           **{key+'_mean':statistics.mean(r[key] for r in subset) for key in ('cosine','mse')},
                           **{key+'_sample_std':statistics.stdev(r[key] for r in subset) if len(subset)>1 else None
                              for key in ('cosine','mse')}})
        write(self.args.output/'comparison.json',{'runs':rows,'groups':groups,
              'scope':'same-episode/common-anchor development validation; closed-loop results are separate'})

    def wait_gpu(self):
        while self.available():
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True, timeout=20)
            if not result.stdout.strip(): return True
            self.update(phase='waiting_gpu', gpu_pids=result.stdout.strip().splitlines())
            time.sleep(self.args.poll_seconds)
        return False

    def resources(self, job):
        path = self.args.output/'resources'/f"{job['variant']}.json"
        if path.is_file(): return read(path)
        # 先用真实128x960特征执行3次优化步骤再决定batch，正式fit会重新固定seed初始化。
        candidates = ([64,32,16,8,4,2,1] if job['model_type']=='transformer-action'
                      else [32,16,8,4,2,1] if job['model_type']=='mlp' else [8,4,2,1])
        for batch in candidates:
            if not self.available(): return None
            if not self.wait_gpu(): return None
            resource = {'batch': batch, 'workers': 0}
            label = f"probe_{job['variant']}_b{batch}"
            try:
                self.child(fit_command(self.args, job, resource, 'probe'), label)
            except RuntimeError:
                text = (self.args.output/'logs'/f'{label}.log').read_text(errors='replace')
                if 'out of memory' in text.lower():
                    self.update(probe_oom_batch=batch, next_action='retry smaller probe; no official fit created')
                    continue
                raise
            sample_bytes = batch*128*960*4*(2+job['h_window'])
            wanted = sample_bytes*(2*self.args.loader_workers+2)*1.5
            resource['workers'] = self.args.loader_workers if shutil.disk_usage('/dev/shm').free >= wanted else 0
            resource['probe'] = read(self.args.output/'probes'/f"{job['variant']}_b{batch}"/'probe.json')
            write(path, resource)
            self.update(selected_resource=resource)
            return resource
        raise RuntimeError('batch=1仍OOM；不改变模型/精度并冒充同一实验')

    def run(self):
        args = self.args
        self.compute_started = time.monotonic()
        self.env = dict(os.environ, HF_HOME=str(args.runtime/'hf'), HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(args.runtime/'matplotlib-cache'),
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        head = subprocess.check_output(['git','-C',str(args.source_root),'rev-parse','HEAD'],text=True).strip()
        if head != SOURCE_SHA: raise ValueError('参考源码版本不对；本队列不联网修复')
        manifest = read(args.archive/'manifest.json')
        if (len(manifest['records']) != 454 or manifest['frames'] != 66984
                or manifest['revision'] != '86958911c0f959db2bbbdb107eb3e17c5f9c798e'):
            raise ValueError('仅接受完整Object10冻结清单')
        jobs = specifications(args)
        self.update(resource_overview={'free_gib':shutil.disk_usage(args.output).free/2**30,
                    'estimated_all_checkpoint_gib':len(jobs)*.75,
                    'reserve_gib':args.min_free_gib,'snapshot_every':args.snapshot_every,
                    'precision':args.predictor_precision,'note':'no raw/feature copies or automatic deletion'})
        files = {p.name: digest(p) for p in HERE.glob('*.py')}
        protocol = {'jobs':jobs, 'manifest_sha256':digest(args.archive/'manifest.json'),
                    'code_sha256':files, 'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                                                   if k not in {'mode','max_compute_hours','poll_seconds'}}}
        frozen = args.output/'sweep_plan.json'
        if frozen.exists() and read(frozen) != protocol:
            raise ValueError('已有夜间计划/代码不同；换output，不篡改原实验')
        write(frozen,protocol)
        for job in jobs:
            self.update(job=job, phase='checking')
            if not self.available(): break
            if shutil.disk_usage(args.output).free < args.min_free_gib*2**30:
                self.update(status='paused', reason='disk_space'); return
            output = args.output/'runs'/job['name']
            output.mkdir(parents=True, exist_ok=True)
            if not self.wait_gpu(): break
            resource = self.resources(job)
            if resource is None or not self.available(): break
            if not complete_training(output):
                stop = output/'TRAIN_STOP'
                if stop.exists(): stop.unlink()  # 只清理由本队列管理的上次预算边界标记。
                self.child(fit_command(args,job,resource,'fit'), job['name']+'_training',
                           stop_path=stop, seconds=self.remaining())
                if not complete_training(output):
                    self.update(status='paused',reason='training_boundary_budget_or_stop',job=job)
                    return
            evidence = read(output/'result.json')
            self.update(training_summary={k:evidence.get(k) for k in
                        ('parameters','epochs_completed','optimizer_steps','best_validation','peak_cuda_gib')})
            checkpoint = output/'best_model.pt'; sha = digest(checkpoint)
            report = args.output/'reports'/job['name']/sha[:16]
            if not self.available(): break
            if not (report/'summary.json').is_file():
                self.child([str(args.runtime/'venv/bin/python'),'-u',str(HERE/'corrector_prediction_report.py'),'predict',
                            '--run',str(output),'--features',str(args.archive/'features'),'--source-root',str(args.source_root),
                            '--output',str(report),'--checkpoint',str(checkpoint),'--device','cuda',
                            '--batch-size',str(resource['batch']),'--loader-workers',str(resource['workers']),
                            '--cpu-threads','4'],job['name']+'_report')
            elif read(report/'summary.json')['checkpoint_sha256'] != sha:
                raise RuntimeError('报告来源SHA与best不同')
            self.summarize()
            if not self.available(): break
            self.child([str(args.runtime/'sim-venv/bin/python'),str(HERE/'corrector_prediction_report.py'),
                        'plot','--output',str(report)],job['name']+'_plots')
            # 每份最终best验证OGG可微调用；强制事件结果不计入正式成功率。
            if not self.available(): break
            smoke = args.output/'evaluation'/job['name']/sha[:16]/'forced_check'
            self.child(evaluation_command(args,job,checkpoint,smoke,self.remaining()/3600,True),
                       job['name']+'_forced_check',stop_path=args.output/'STOP')
            state = read(smoke/'execution.json')
            if state['status'] != 'completed':
                self.update(status='paused',reason='smoke_paused'); return
            records = [read(p) for p in (smoke/'trials').glob('*.json')]
            if state.get('errors') or not records or not all(r['result']['guidance_calls'] > 0 for r in records):
                raise RuntimeError('强制事件检查未证实OGG梯度调用；不继续无人值守评测')
            if not args.no_eval and job['seed'] in args.eval_training_seeds:
                if not self.available(): break
                paired = args.output/'evaluation'/job['name']/sha[:16]/'paired'
                self.child(evaluation_command(args,job,checkpoint,paired,self.remaining()/3600),
                           job['name']+'_paired',stop_path=args.output/'STOP')
                state = read(paired/'execution.json')
                if state.get('errors'): raise RuntimeError('正式评测程序错误；保留所有结果')
                if state['status'] != 'completed':
                    self.update(status='paused',reason='evaluation_budget_or_stop'); return
            write(output/'JOB_COMPLETE.json', {'checkpoint_sha256':sha, 'report':str(report),
                                              'paired_requested':not args.no_eval and job['seed'] in args.eval_training_seeds})
        else:
            self.update(status='completed',phase='all_jobs_finished',jobs=len(jobs))
            return
        self.update(status='paused',reason='wall_budget_or_stop')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path('/root/autodl-tmp/corrector')
    p.add_argument('mode',choices=['run','plan','status'])
    p.add_argument('--archive',type=Path,default=base/'object10')
    p.add_argument('--runtime',type=Path,default=base/'runtime')
    p.add_argument('--source-root',type=Path,default=base/'runtime/vla-corrector-reference')
    p.add_argument('--output',type=Path,default=base/'object10/night_20261109_001')
    p.add_argument('--models',nargs='+',choices=['mlp','transformer-h1','transformer-h3','transformer-a10-h1','transformer-a10-h3'],
                   default=['transformer-h1','mlp','transformer-h3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--eval-training-seeds',nargs='+',type=int,default=[42])
    p.add_argument('--methods',nargs='+',choices=['H50','H20','monitor','lvm','ogg'],default=['H50','H20','lvm','ogg'])
    p.add_argument('--eval-initials',type=int,default=3)
    p.add_argument('--eval-random-seeds',nargs='+',type=int,default=[1000,1001])
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--snapshot-every',type=int,default=0)
    p.add_argument('--predictor-precision',choices=['float32','amp-bfloat16'],default='amp-bfloat16')
    p.add_argument('--loader-workers',type=int,default=2)
    p.add_argument('--max-compute-hours',type=float,default=12)
    p.add_argument('--min-free-gib',type=float,default=5)
    p.add_argument('--poll-seconds',type=float,default=10)
    p.add_argument('--no-eval',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if (any(not math.isfinite(v) or v<=0 for v in (args.max_compute_hours,args.min_free_gib,args.poll_seconds))
            or min(args.epochs,args.patience,args.eval_initials)<1 or args.eval_initials>50
            or args.snapshot_every<0 or args.loader_workers<0 or any(s<0 for s in args.seeds+args.eval_random_seeds)
            or len(set(args.seeds))!=len(args.seeds) or len(set(args.models))!=len(args.models)
            or len(set(args.methods))!=len(args.methods) or len(set(args.eval_random_seeds))!=len(args.eval_random_seeds)):
        raise ValueError('预算、模型/种子列表或资源参数不合法')
    if args.mode=='status':
        print(json.dumps(read(args.output/'sweep_status.json'),ensure_ascii=False,indent=2));return
    if args.mode=='plan':
        print(json.dumps({'jobs':specifications(args),'hours':args.max_compute_hours,'common_anchor_history':3,
                          'effective_batch':512,'transformer':'C768/12heads/5layers; single action 37,316,544 params; sequence adds 7,680',
                          'predictor_precision':args.predictor_precision,
                          'formal_trials_per_selected_model':10*args.eval_initials*len(args.eval_random_seeds)*len(args.methods),
                          'eval_training_seeds':args.eval_training_seeds,'offline':True,
                          'note':'finite resumable queue; deadline may leave jobs pending; no shutdown or downloads'},
                         ensure_ascii=False,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'logs').mkdir(exist_ok=True)
    with lock(args.output/'.sweep.lock'):
        runner=Sweep(args)
        def stop(signum,frame): runner.stopping=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        try: runner.run()
        except BaseException as error:
            runner.update(status='error',error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__': main()
