"""Formal training/checkpoint adapter; all source caches remain read-only."""
import copy,csv,hashlib,json,math,os,random,subprocess,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from corrector_data_workbench import digest,write_json
from corrector_train import atomic_save,reference_loaders,reference_update
from multilayer_mvp.data import MultiLayerTransitionDataset
from multilayer_mvp.metrics import Metrics
from .audit import frame
from .model import make_model,LAYERS
from .loss import objective,LossConfig,shared_gradient_audit


def deterministic():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def build_commit():
    root=Path(__file__).resolve().parents[2]
    marker=root/'DEPLOYMENT_MANIFEST.json'
    if marker.exists():return json.loads(marker.read_text())['git_commit']
    try:return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):return None


class TransitionDataset(MultiLayerTransitionDataset):
    def __init__(self,records,split,variant):
        super().__init__(records,split,10,variant=='C0');self.variant=variant
    def __getitem__(self,index):
        if self.variant not in ('S2','S8'):return super().__getitem__(index)
        r,t=self.locate(index);k={'S2':'block_02','S8':'block_08'}[self.variant]
        now=frame(r,k,t);future=frame(r,k,t+10)
        return {'inputs':{k:now,'action':r['arrays']['actions'][t].astype('float32')},
                'targets':{'delta_'+k:future-now},'episode':r['spec']['episode'],'t':t}


def scope_loss(variant):
    return LossConfig(weights=(1/3,1/3,1/3) if variant=='J_equal' else (0.,0.,1.) if variant=='JC' else (.25,.25,.5))


def source_hash():
    root=Path(__file__).parent
    scripts=root.parent
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix=='.py']
    paths += [scripts/name for name in ('corrector_j36m.py','corrector_j36m_pipeline.py','corrector_online_runtime.py','corrector_models.py','corrector_train.py','corrector_data_workbench.py','corrector_multilayer.py','corrector_multilayer_pilot.py')]
    paths += [scripts/name for name in ('corrector_prefetch.py','smolvla_libero_tools.py','smolvla_training_workbench.py')]
    paths += [scripts.parent/'studies/policy_generalization/corrector'/name for name in ('online_practice.py','batch_evaluation.py')]
    paths += list((scripts/'multilayer_mvp').glob('*.py'))
    paths += list((scripts.parent/'studies/policy_generalization/corrector/j36m').glob('data*.json'))
    paths += [scripts.parent/'studies/policy_generalization/corrector/j36m'/n for n in ('manifest70.json','historical_plus_reference.json')]
    files={str(p.relative_to(scripts.parent)):digest(p) for p in paths}
    return {'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def numpy_rng_state():
    name,keys,pos,has_gauss,cached_gaussian=np.random.get_state()
    return {'name':name,'keys':keys.tolist(),'pos':pos,'has_gauss':has_gauss,'cached_gaussian':cached_gaussian}


def restore_numpy_rng(state):
    np.random.set_state((state['name'],np.asarray(state['keys'],dtype=np.uint32),state['pos'],state['has_gauss'],state['cached_gaussian']))


def check_stats(audit,records):
    if audit['source']!=[r['provenance'] for r in records]:raise ValueError('scale audit/data content mismatch')
    return audit['train_only_input_rms'],audit['train_only_delta_rms']


def move(batch,device):
    return {kind:{k:v.to(device) for k,v in batch[kind].items()} for kind in ('inputs','targets')}


def amp(precision):
    return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


def evaluate(model,loader,scales,device,precision,records):
    effective={k:(v*v+1e-12)**.5 for k,v in scales.items()}
    total=Metrics(effective);episodes={};tasks={};mapping={r['spec']['episode']:r['spec']['task_id'] for r in records}
    model.eval()
    with torch.no_grad():
        for raw in loader:
            b=move(raw,device)
            with amp(precision):pred=model(b['inputs'])
            total.add(pred,b['targets'])
            for i,ep in enumerate(raw['episode'].tolist()):
                for key,book in ((ep,episodes),(mapping[ep],tasks)):
                    metric=book.setdefault(key,Metrics(effective))
                    metric.add({k:v[i:i+1] for k,v in pred.items()},{k:v[i:i+1] for k,v in b['targets'].items()})
    return {'overall':total.result(),'episodes':{str(k):v.result() for k,v in episodes.items()},
            'tasks':{str(k):v.result() for k,v in tasks.items()}}


def train(args,cfg,records,audit,guard):
    """Epoch-transaction recovery, explicit microbatch/accumulation, no automatic OOM shrink."""
    output=Path(args.output)
    input_rms,scales=check_stats(audit,records)
    frozen={'variant':args.variant,'epochs':args.epochs,'microbatch':args.microbatch,'accumulate':args.accumulate,
            'patience':args.patience,'seed':args.seed,'lr':args.lr,'weight_decay':1e-4,'precision':args.precision,
            'gradient_checkpointing':args.gradient_checkpointing,'loss':scope_loss(args.variant).serializable(),
            'config':cfg,'data':[r['provenance'] for r in records],'source':source_hash(),
            'input_rms':input_rms,'delta_rms':scales,'timebase':{'data_hz':10,'k':10,'online_hz':20,'due_steps':20},
            'environment':{'torch':str(torch.__version__),'numpy':np.__version__,'device':args.device},
            'build_git_commit':build_commit(),'deterministic':True,'tf32':False,
            'selection_metric':'validation normalized_mse of delta_'+({'S2':'block_02','S8':'block_08'}.get(args.variant,'connector')),
            'architecture_parameters':{'J':36793536,'J_equal':36793536,'JC':36793536,'C0':38052800,'S2':37266176,'S8':37266176}[args.variant]}
    if output.exists():
        if not args.resume or not (output/'latest.pt').is_file():raise FileExistsError('existing output requires complete latest and --resume')
        if json.loads((output/'protocol.json').read_text())!=json.loads(json.dumps(frozen)):raise ValueError('resume frozen protocol mismatch')
    else:
        if args.resume:raise FileNotFoundError('no checkpoint')
        output.mkdir(parents=True)
        write_json(output/'protocol.json',frozen)
    import fcntl
    with (output/'train.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(args,records,output,frozen,guard,input_rms,scales)


def _train(args,records,output,frozen,guard,input_rms,scales):
    deterministic()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    if args.device=='cuda':torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    generator=torch.Generator().manual_seed(args.seed)
    groups={s:TransitionDataset(records,s,args.variant) for s in ('train','validation')}
    tr,va=reference_loaders(groups,args.microbatch,generator,num_workers=0,pin_memory=args.device=='cuda')
    model=make_model(args.variant,input_rms,args.gradient_checkpointing).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.01)
    epoch=step=stale=0;peak_cuda=0.;best=None;best_weights=None;history=[];best_epoch=0
    if args.resume:
        state=torch.load(output/'latest.pt',map_location='cpu',weights_only=True)
        if state['protocol']!=json.loads(json.dumps(frozen)):raise ValueError('checkpoint protocol mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch,step,stale,best,best_epoch,best_weights,history=(state[k] for k in ('epoch','step','stale','best','best_epoch','best_weights','history'))
        peak_cuda=state.get('peak_cuda_allocated_gib',0.)
        generator.set_state(state['loader_rng']);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
        restore_numpy_rng(state['numpy_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    status={'status':'running','epochs_completed':epoch,'optimizer_steps':step,'variant':args.variant}
    def save():
        atomic_save(output/'latest.pt',{'protocol':json.loads(json.dumps(frozen)),'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,
                    'stale':stale,'best':best,'best_epoch':best_epoch,'best_weights':best_weights,'history':history,
                    'loader_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                    'numpy_rng':numpy_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else [],
                    'peak_cuda_allocated_gib':max(peak_cuda,torch.cuda.max_memory_allocated()/2**30) if args.device=='cuda' else None})
        if best_weights is not None:atomic_save(output/'best.pt',{'protocol':json.loads(json.dumps(frozen)),
                         'model':best_weights,'epoch':best_epoch,'validation':best,'input_rms':input_rms,'delta_rms':scales})
        write_json(output/'history.json',history)
        with (output/'metrics.csv.partial').open('w',newline='') as f:
            fields=['epoch','step','train_loss','seconds','layer','mse','normalized_mse','zero_delta_mse','zero_delta_normalized_mse','relative_mse_vs_zero','cosine','samples','cosine_valid','zero_targets','zero_predictions']
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for row in history:
                for layer,metric in row['validation']['overall'].items():writer.writerow({k:row[k] for k in fields[:4]}|{'layer':layer}|metric)
        (output/'metrics.csv.partial').replace(output/'metrics.csv')
    if not args.resume:save()
    start=time.monotonic()
    try:
        while epoch<args.epochs and not (args.patience>0 and stale>=args.patience):
            guard();tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            value=0.;samples=0;grad_audit=None
            for i,raw in enumerate(tr):
                if i%64==0 and hasattr(guard,'resources'):
                    guard.resources()
                    write_json(output/'status.json',dict(status,current_epoch=epoch+1,microbatch=i,checkpoint_boundary='last completed epoch'))
                b=move(raw,args.device)
                with amp(args.precision):
                    pred=model(b['inputs'])
                    train_pred={'delta_connector':pred['delta_connector']} if args.variant=='JC' else pred
                    loss,rows=objective(train_pred,b['targets'],scales,scope_loss(args.variant))
                if i==0 and args.variant.startswith('J'):grad_audit=shared_gradient_audit(rows,model)
                # Weight each microbatch by actual samples in this accumulation group, including final partial.
                group_start=(i//args.accumulate)*args.accumulate*args.microbatch
                group_samples=min(args.accumulate*args.microbatch,len(groups['train'])-group_start)
                should=(i+1)%args.accumulate==0 or i+1==len(tr)
                reference_update(model,optimizer,loss*len(raw['t'])/group_samples,should)
                if should:step+=1
                value+=float(loss.detach())*len(raw['t']);samples+=len(raw['t'])
            model.eval();validation=evaluate(model,va,scales,args.device,args.precision,records)
            key='delta_connector' if args.variant not in ('S2','S8') else 'delta_'+model.layer
            score=validation['overall'][key]['normalized_mse']
            if best is None or score<best['overall'][key]['normalized_mse']:
                best=validation;best_epoch=epoch+1;stale=0;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            scheduler.step();epoch+=1
            if args.device=='cuda':torch.cuda.synchronize()
            history.append({'epoch':epoch,'step':step,'train_loss':value/samples,'validation':validation,
                            'seconds':time.perf_counter()-tick,'shared_gradient_audit':grad_audit})
            save();status.update(epochs_completed=epoch,optimizer_steps=step,best_epoch=best_epoch)
            write_json(output/'status.json',status);print(json.dumps(history[-1],allow_nan=False),flush=True)
        save()
        loaded=make_model(args.variant,input_rms).cpu();loaded.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model'])
        verified=all(torch.equal(v,loaded.state_dict()[k]) for k,v in best_weights.items())
        result={'status':'completed','epochs_completed':epoch,'optimizer_steps':step,'best_epoch':best_epoch,'best_validation':best,
                'parameters':sum(p.numel() for p in model.parameters()),'peak_cuda_allocated_gib':max(peak_cuda,torch.cuda.max_memory_allocated()/2**30) if args.device=='cuda' else None,
                'epoch_seconds':sum(h['seconds'] for h in history),'invocation_seconds':time.monotonic()-start,
                'checkpoint_reload_verified':verified,'scope':'offline predictor; no closed-loop success claim'}
        write_json(output/'result.json',result);write_json(output/'status.json',dict(status,status='completed'));return result
    except BaseException as exc:
        write_json(output/'status.json',dict(status,status='paused_or_error',error=f'{type(exc).__name__}: {exc}',recovery='last complete epoch, unchanged config + --resume'))
        raise


def gpu_smoke(args,records,audit,guard):
    """Real data only. Same fixed architecture; optimizer and model discarded afterward."""
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);guard();deterministic()
    torch.set_num_threads(4);torch.manual_seed(42)
    rms,scales=check_stats(audit,records);model=make_model(args.variant,rms,args.gradient_checkpointing).to(args.device)
    dataset=TransitionDataset(records,'train',args.variant);loader=iter(DataLoader(dataset,batch_size=args.microbatch,num_workers=0))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    prefixes=('fusion.','heads.connector.') if args.variant=='JC' else ('fusion.','heads.','model.out_proj.')
    initial_weights={k:p.detach().clone() for k,p in model.named_parameters() if k.startswith(prefixes)}
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    results=[];step_times=[];first=None;start=time.perf_counter()
    for i in range(2+args.overfit_steps):
        if args.device=='cuda':torch.cuda.synchronize()
        tick=time.perf_counter()
        raw=next(loader) if i<2 else first
        if first is None:first=raw
        b=move(raw,args.device);optimizer.zero_grad(set_to_none=True)
        with amp(args.precision):
            pred=model(b['inputs']);selected={'delta_connector':pred['delta_connector']} if args.variant=='JC' else pred
            loss,rows=objective(selected,b['targets'],scales,scope_loss(args.variant))
        if i==0:
            before=float(loss.detach());gradient=shared_gradient_audit(rows,model) if args.variant.startswith('J') else None
        reference_update(model,optimizer,loss,True)
        results.append(float(loss.detach()))
        if args.device=='cuda':torch.cuda.synchronize()
        step_times.append(time.perf_counter()-tick)
    if args.device=='cuda':torch.cuda.synchronize()
    with torch.no_grad(),amp(args.precision):
        b=move(first,args.device);out=model(b['inputs']);selected={'delta_connector':out['delta_connector']} if args.variant=='JC' else out
        after=float(objective(selected,b['targets'],scales,scope_loss(args.variant))[0])
    if after>=before*.95:raise RuntimeError(f'overfit smoke insufficient: {before}->{after}')
    changed={k:not torch.equal(p.detach(),initial_weights[k]) for k,p in model.named_parameters() if k in initial_weights}
    if not changed or not all(changed.values()):raise RuntimeError('required trunk/head parameters did not update')
    atomic_save(output/'smoke.pt',{'model':model.state_dict()})
    other=make_model(args.variant,rms);other.load_state_dict(torch.load(output/'smoke.pt',map_location='cpu',weights_only=True)['model'])
    if not all(torch.equal(v.cpu(),other.state_dict()[k]) for k,v in model.state_dict().items()):raise RuntimeError('reload mismatch')
    report={'status':'passed','variant':args.variant,'parameters':sum(p.numel() for p in model.parameters()),
            'initial_loss':before,'final_loss':after,'losses':results,'shared_gradient_audit':gradient,
            'step_seconds':step_times,'repeated_batch_samples_per_second':args.microbatch*args.overfit_steps/sum(step_times[2:]),
            'throughput_scope':'repeated cached smoke batch, not full dataset epoch throughput',
            'two_batches':True,'overfit_steps':args.overfit_steps,'seconds':time.perf_counter()-start,
            'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else None,
            'source':source_hash(),'data':[r['provenance'] for r in records],
            'shape':{k:list(v.shape) for k,v in pred.items()},'checkpoint_reload_verified':True,'parameter_updates':changed}
    write_json(output/'smoke.json',report);return report
