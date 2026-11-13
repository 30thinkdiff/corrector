"""Unified multilayer MVP: explicit preflight / smoke / fit / paired / results.

No extractor, VLA loading, downloads, pip, or closed-loop control in this entry.
Reuse frozen training optimizer/loaders/update/atomic checkpoint utilities.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import csv
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from corrector_data_workbench import write_json
from corrector_train import atomic_save, reference_components, reference_loaders, reference_update
from multilayer_mvp.data import (MultiLayerTransitionDataset, fit_scales, load_config, validate)
from multilayer_mvp.metrics import Metrics, loss
from multilayer_mvp.model import ModelConfig, MultiLayerCorrector
from multilayer_mvp.runtime import ResourcePause, resources, safe_output, source_manifest

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({'stop_requested': True, 'boundary': 'complete epoch', 'signal': signum}), flush=True)


def move(batch, device):
    return {group: {k: v.to(device, non_blocking=True) for k, v in batch[group].items()}
            for group in ('inputs', 'targets')}


def autocast(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def evaluate(model, loader, scales, device, precision):
    metrics = Metrics(scales)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = move(batch, device)
            with autocast(device, precision):
                pred = model(x['inputs'])
            metrics.add(pred, x['targets'])
    return metrics.result()


def configure(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported; explicitly choose fp32 for a new run')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif args.precision != 'fp32':
        raise ValueError('CPU uses fp32')


def data_setup(cfg, records, variant):
    groups = {s: MultiLayerTransitionDataset(records, s, cfg['k_step'], variant == 'connector')
              for s in ('train', 'validation')}
    return groups


def write_history(output, history):
    write_json(output / 'history.json', history)
    tmp = output / 'metrics.csv.partial'
    fields = ['epoch', 'step', 'train_loss', 'seconds', 'layer', 'mse', 'normalized_mse',
              'zero_delta_mse', 'zero_delta_normalized_mse', 'relative_mse_vs_zero', 'cosine',
              'samples', 'cosine_valid', 'zero_targets', 'zero_predictions']
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            for layer, values in row['validation'].items():
                writer.writerow({k: row[k] for k in fields[:4]} | {'layer': layer} | values)
    tmp.replace(output / 'metrics.csv')


def make_protocol(args, cfg, records):
    return {'format': 'multilayer_training_mvp_v1', 'config': cfg,
            'source': source_manifest(), 'data': [r['provenance'] for r in records],
            'training': {k: getattr(args, k) for k in
                         ('variant', 'seed', 'batch_size', 'epochs', 'patience', 'lr', 'weight_decay',
                          'width', 'depth', 'precision', 'cpu_threads')},
            'environment': {'torch': str(torch.__version__), 'numpy': np.__version__, 'device': args.device},
            'selection_metric': 'validation delta_connector.normalized_mse for both variants',
            'loader_workers': 0, 'k_step': 10, 'data_fps': 10., 'prediction_seconds': 1.,
            'inputs': 'current visual features + normalized current 7D action only',
            'loss': 'mean of per-layer MSE / train-only RMS(delta)^2',
            'supervision': 'full raw [2,1024,768], Connector [128,960]; no target pooling',
            'context': 'global projected token means used only for cross-stream conditioning',
            'scope': '5-episode offline engineering MVP; no success-rate/generalization claim'}


def check_resume_protocol(old, new):
    # Runtime checkout SHA may be absent in ZIP deployment; actual file digests must match.
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    for p in (a, b):
        p['source'].pop('git_sha_at_execution', None)
    if a != b:
        raise ValueError('resume protocol/source/data mismatch; use a new output, do not overwrite')


def fit(args, cfg, records):
    output = safe_output(args.output, cfg)
    resources([output, cfg['features_root']], enforce=True, gpu=args.device == 'cuda')
    configure(args)
    protocol = make_protocol(args, cfg, records)
    if output.exists():
        if not args.resume or not (output / 'latest.pt').is_file():
            raise FileExistsError('existing output requires --resume and latest.pt; otherwise use a new path')
        check_resume_protocol(json.loads((output/'protocol.json').read_text()), protocol)
    else:
        if args.resume:
            raise FileNotFoundError('resume output/latest.pt absent')
        output.mkdir(parents=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fit_locked(args, cfg, records, output, protocol)


def fit_locked(args, cfg, records, output, protocol):
    write_json(output/'status.json', {'status': 'running', 'phase': 'initialization_or_resume'})
    groups = data_setup(cfg, records, args.variant)
    generator = torch.Generator().manual_seed(args.seed)
    train, val = reference_loaders(groups, args.batch_size, generator, num_workers=0,
                                   pin_memory=args.device == 'cuda')
    model_cfg = ModelConfig(args.variant, tuple(cfg['raw_shape']), tuple(cfg['connector_shape']),
                            args.width, args.depth)
    model, optimizer, scheduler = reference_components(MultiLayerCorrector, model_cfg, args.device,
                                                       args.lr, args.weight_decay, args.epochs)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    epoch, step, best_epoch, stale = 0, 0, 0, 0
    best, best_state, history = None, None, []
    # Complete epoch is the transactional recovery boundary. No partial optimizer state is published.
    if args.resume:
        state = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        check_resume_protocol(state['protocol'], protocol)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch, step, best_epoch, stale = (state[k] for k in ('epoch', 'step', 'best_epoch', 'stale'))
        scales, best, best_state, history = (state[k] for k in ('scales', 'best', 'best_state', 'history'))
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['loader_rng'])
        random.setstate(state['python_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        write_history(output, history)
    else:
        write_json(output / 'protocol.json', protocol)
        print(json.dumps({'phase': 'train_only_scales', 'samples': len(groups['train'])}), flush=True)
        scales = fit_scales(groups['train'])
        write_json(output / 'scales.json', {'scales': scales, 'fitted_split': 'train',
                    'episodes': [r['spec']['episode'] for r in groups['train'].records],
                    'samples': len(groups['train'])})
        write_json(output / 'initial_validation.json', evaluate(model, val, scales, args.device, args.precision))

    def save():
        atomic_save(output / 'latest.pt', {
            'protocol': protocol, 'model_cfg': model.config_dict(), 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': epoch, 'step': step, 'best_epoch': best_epoch, 'stale': stale,
            'scales': scales, 'best': best, 'best_state': best_state, 'history': history,
            'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []})
        if best_state is not None:
            atomic_save(output / 'best.pt', {'model_cfg': model.config_dict(), 'model': best_state,
                        'epoch': best_epoch, 'metrics': best, 'protocol': protocol, 'scales': scales})

    if not args.resume:
        save()
    invocation_utc = datetime.now(timezone.utc).isoformat()
    status = {'status': 'running', 'epoch': epoch, 'optimizer_steps': step, 'invocation_utc': invocation_utc}
    started = time.perf_counter()
    try:
        while epoch < args.epochs and not (args.patience > 0 and stale >= args.patience):
            if STOP_REQUESTED or (output/'STOP').exists() or (getattr(args, 'stop_file', None) is not None and args.stop_file.exists()):
                raise ResourcePause('STOP requested at complete epoch boundary')
            resources([output, cfg['features_root']], enforce=True)
            model.train()
            count, total = 0, 0.
            tick = time.perf_counter()
            for batch in train:
                if step % 20 == 0:
                    resources([output], enforce=True)
                x = move(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(args.device, args.precision):
                    pred = model(x['inputs'])
                    objective = loss(pred, x['targets'], scales)
                reference_update(model, optimizer, objective, True)
                count += len(batch['t'])
                total += float(objective.detach()) * len(batch['t'])
                step += 1
            metrics = evaluate(model, val, scales, args.device, args.precision)
            score = metrics['delta_connector']['normalized_mse']
            if best is None or score < best['delta_connector']['normalized_mse']:
                best, best_epoch, stale = metrics, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            scheduler.step()
            epoch += 1
            if args.device == 'cuda':
                torch.cuda.synchronize()
            history.append({'epoch': epoch, 'step': step, 'train_loss': total/count,
                            'validation': metrics, 'seconds': time.perf_counter() - tick})
            save()
            write_history(output, history)
            status.update(epoch=epoch, optimizer_steps=step, best_epoch=best_epoch)
            write_json(output/'status.json', status)
            print(json.dumps(history[-1], allow_nan=False), flush=True)
        # Best is reconstructed from the atomic latest checkpoint after recovery, never retrained.
        save()
        write_history(output, history)
        saved = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        reloaded = MultiLayerCorrector(model_cfg)
        reloaded.load_state_dict(saved['model'], strict=True)
        verified = all(torch.equal(v, reloaded.state_dict()[k]) for k, v in best_state.items())
        result = {'status': 'completed', 'epochs': epoch, 'optimizer_steps': step, 'best_epoch': best_epoch,
                  'best_validation': best, 'final_validation': history[-1]['validation'],
                  'selection_metric': protocol['selection_metric'], 'parameters': sum(p.numel() for p in model.parameters()),
                  'epoch_compute_seconds_total': sum(r['seconds'] for r in history),
                  'invocation_utc': invocation_utc, 'this_invocation_seconds': time.perf_counter()-started,
                  'peak_cuda_gib': torch.cuda.max_memory_allocated()/2**30 if args.device == 'cuda' else None,
                  'checkpoint_reload_verified': verified, 'split_samples': {k: len(v) for k, v in groups.items()},
                  'reason': 'early_stopping' if epoch < args.epochs else 'epoch_budget',
                  'scope': protocol['scope']}
        write_json(output/'result.json', result)
        status.update(status='completed', reason=result['reason'])
        write_json(output/'status.json', status)
        return result
    except BaseException as exc:
        paused = isinstance(exc, ResourcePause) or isinstance(exc, torch.cuda.OutOfMemoryError)
        status.update(status='paused' if paused else 'error',
                      error=f'{type(exc).__name__}: {exc}', recovery='resume from last complete epoch/latest.pt')
        write_json(output/'status.json', status)
        raise


def smoke(args, cfg, records):
    output = safe_output(args.output, cfg)
    if output.exists():
        raise FileExistsError('smoke output already exists; use a new directory')
    resources([output, cfg['features_root']], enforce=True, gpu=args.device == 'cuda')
    configure(args)
    output.mkdir(parents=True)
    model = MultiLayerCorrector(ModelConfig(args.variant, tuple(cfg['raw_shape']),
                                            tuple(cfg['connector_shape']), args.width, args.depth)).to(args.device)
    dataset = data_setup(cfg, records, args.variant)['train']
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    batches, gradients = [], {}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    first = None
    for batch in loader:
        x = move(batch, args.device)
        if first is None:
            first = x
            scales = {k: max(float(v.square().mean().sqrt()), 1e-6) for k, v in x['targets'].items()}
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device, args.precision):
            objective = loss(model(x['inputs']), x['targets'], scales)
        objective.backward()
        for key, p in model.named_parameters():
            if p.grad is not None:
                gradients[key] = float(p.grad.float().norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        batches.append(float(objective.detach()))
        if len(batches) == 2:
            break
    if len(batches) != 2:
        raise ValueError('need two real batches')
    changed = {k: not torch.equal(p.detach(), before[k]) for k, p in model.named_parameters()}
    for prefix in ('backbone.', *(f'heads.{k}.' for k in first['targets'])):
        if not any(changed[k] and gradients.get(k, 0) > 0 for k in changed if k.startswith(prefix)):
            raise RuntimeError('missing shared trunk / output head gradient and update: ' + prefix)
    with torch.no_grad(), autocast(args.device, args.precision):
        initial = float(loss(model(first['inputs']), first['targets'], scales))
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device, args.precision):
            objective = loss(model(first['inputs']), first['targets'], scales)
        reference_update(model, optimizer, objective, True)
    with torch.no_grad(), autocast(args.device, args.precision):
        final = float(loss(model(first['inputs']), first['targets'], scales))
    if final >= .95 * initial:
        raise RuntimeError(f'single-batch overfit failed: {initial} -> {final}')
    report = {'status': 'passed', 'device': args.device, 'two_batch_losses': batches,
              'overfit_initial': initial, 'overfit_final': final, 'overfit_steps': args.overfit_steps,
              'gradient_norms': gradients, 'parameters_changed': changed,
              'peak_cuda_gib': torch.cuda.max_memory_allocated()/2**30 if args.device == 'cuda' else None,
              'protocol': make_protocol(args, cfg, records),
              'note': 'discarded smoke model; no official predictor training result'}
    write_json(output/'smoke.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['preflight', 'smoke', 'fit', 'paired', 'results'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['connector', 'joint'], default='joint')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=0, help='0 gives matched A/B step budgets; >0 breaks this guarantee')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--weight-decay', type=float, default=.0001)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--overfit-steps', type=int, default=30)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-file', type=Path, help='external boundary-pause request; paired defaults to OUTPUT/STOP')
    return p


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.cpu_threads, args.overfit_steps) < 1 or args.patience < 0:
        raise ValueError('invalid training budget')
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, request_stop)
    cfg = load_config(args.config)
    if args.mode == 'results':
        for name in ('connector', 'joint'):
            path = args.output / name / 'result.json'
            print(json.dumps({'variant': name, 'result': json.loads(path.read_text()) if path.exists() else None},
                             ensure_ascii=False, allow_nan=False))
        return
    print(json.dumps({'phase': 'read_only_SHA_validation', 'episodes': len(cfg['episodes'])}), flush=True)
    records = validate(cfg)
    if args.mode == 'preflight':
        # No CUDA context/model creation; reports hardware through nvidia-smi only.
        samples = []
        for record in records:
            ds = MultiLayerTransitionDataset([record], record['spec']['split'])
            for index in (0, len(ds)-1):
                batch = ds[index]  # finite checks; only two transitions per episode
                samples.append({'episode': batch['episode'], 't': batch['t'],
                                'inputs': {k: list(v.shape) for k, v in batch['inputs'].items()},
                                'targets': {k: list(v.shape) for k, v in batch['targets'].items()}})
        report = {'episodes': [r['provenance'] for r in records], 'boundary_samples': samples,
                  'pairs': {s: len(MultiLayerTransitionDataset(records, s)) for s in ('train', 'validation')},
                  'resources': resources([args.output, cfg['features_root'], *(r['raw_path'] for r in cfg['episodes'])], gpu=True),
                  'source': source_manifest(), 'training_started': False}
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    elif args.mode == 'smoke':
        print(json.dumps(smoke(args, cfg, records), allow_nan=False), flush=True)
    elif args.mode == 'fit':
        print(json.dumps(fit(args, cfg, records), allow_nan=False), flush=True)
    else:
        if args.patience:
            raise ValueError('paired requires patience=0 to keep identical optimizer-step budgets')
        root = args.output
        if args.stop_file is None:
            args.stop_file = root / 'STOP'
        results = {}
        resume_requested = args.resume
        for variant in ('connector', 'joint'):
            if STOP_REQUESTED or args.stop_file.exists():
                raise ResourcePause('paired queue stopped before next variant')
            args.variant, args.output = variant, root / variant
            # --resume permits existing completed A plus pending B; new B starts explicitly fresh.
            args.resume = resume_requested and (args.output/'latest.pt').exists()
            results[variant] = fit(args, cfg, records)
        if results['connector']['optimizer_steps'] != results['joint']['optimizer_steps']:
            raise RuntimeError('A/B step budgets differ')
        write_json(root/'comparison.json', {'results': results, 'matched_steps': True,
                   'caveats': ['parameter counts differ', 'single validation episode, overlapping windows',
                               'offline prediction metrics are not robot success rates']})


if __name__ == '__main__':
    main()
