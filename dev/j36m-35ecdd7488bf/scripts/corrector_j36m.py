"""Phased formal J-36M commands. No implicit GPU run, install or model download."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from corrector_data_workbench import digest,verified,write_json
from multilayer_mvp.data import load_config,validate
from multilayer_mvp.runtime import resources,ResourcePause,safe_output
from j36m.model import make_model,LAYERS
from j36m.training import source_hash

STOP=False

def stop_handler(*_):
    global STOP;STOP=True


def gpu_idle():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    if any(row.strip()!=str(os.getpid()) for row in rows if row.strip()):raise ResourcePause('GPU occupied by another process')


def make_guard(args,gpu=False):
    start=time.monotonic()
    def disk():
        report=resources(['/root','/root/autodl-tmp'])
        if report.get('effective_available_ram_gib',report.get('host_available_ram_gib',999))<2:raise ResourcePause('less than 2GiB effective RAM available')
        for root,reserve in [('/root',args.system_min_free_gib),('/root/autodl-tmp',args.data_min_free_gib)]:
            if Path(root).exists() and shutil.disk_usage(root).free<reserve*2**30:raise ResourcePause(f'{root} free below {reserve}GiB')
    def guard():
        disk()
        if STOP or (args.stop_file and args.stop_file.exists()):raise ResourcePause('operator STOP at complete boundary')
        if args.max_hours and time.monotonic()-start>args.max_hours*3600:raise ResourcePause('wall budget boundary')
    guard.resources=disk
    disk()
    if gpu:
        gpu_idle()
        if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    return guard


def receipt(args,name):
    path=args.receipts/(name+'.json')
    if not path.exists():raise FileNotFoundError('stage gate missing: '+str(path))
    data=json.loads(path.read_text())
    if data['status']!='passed' or data['source']!=source_hash():raise ValueError('failed/stale source gate: '+name)
    if not Path(data['artifact']).is_file() or digest(data['artifact'])!=data['artifact_sha256']:raise ValueError('gate artifact changed: '+name)
    return data


def publish(args,name,artifact,extra=None):
    args.receipts.mkdir(parents=True,exist_ok=True)
    p=args.receipts/(name+'.json')
    value={'status':'passed','source':source_hash(),'artifact':str(Path(artifact).resolve()),'artifact_sha256':digest(artifact),**(extra or {})}
    if p.exists() and json.loads(p.read_text())!=value:raise FileExistsError('existing immutable gate conflict')
    write_json(p,value)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['preflight','cpu-smoke','cache-audit','station-audit','gpu-smoke','extract70','verify70','train','status','stop','calibration-template','calibrate','plus','summarize'])
    base=Path(__file__).resolve().parents[1];conf=base/'studies/policy_generalization/corrector/j36m'
    p.add_argument('--config',type=Path,default=conf/'data5.json');p.add_argument('--manifest',type=Path,default=conf/'manifest70.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipts',type=Path,default=Path('/root/autodl-tmp/corrector/j36m_runs/receipts_001'))
    p.add_argument('--audit',type=Path);p.add_argument('--variant',choices=['J','J_equal','JC','C0','S2','S8'],default='J')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--microbatch',type=int,default=1);p.add_argument('--accumulate',type=int,default=32)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--overfit-steps',type=int,default=30);p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-file',type=Path);p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--runtime',type=Path,default=Path('/root/autodl-tmp/corrector/runtime'))
    p.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/corrector/runtime/vla-corrector-reference'))
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/corrector/object10-work/source'))
    p.add_argument('--allow-source-download',action='store_true');p.add_argument('--proxy-url')
    p.add_argument('--data-min-free-gib',type=float,default=20.)
    p.add_argument('--system-min-free-gib',type=float,default=8.)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--plus-source',type=Path,default=Path('/root/autodl-tmp/corrector/runtime/libero-plus'))
    p.add_argument('--plus-config',type=Path,default=Path('/root/autodl-tmp/corrector/runtime/libero-plus-config'))
    p.add_argument('--phase',choices=['calibration','online-smoke','development','heldout'],default='online-smoke')
    p.add_argument('--include-monitor',action='store_true');p.add_argument('--development-receipt',type=Path);p.add_argument('--physical-reference',type=Path)
    args=p.parse_args()
    if min(args.epochs,args.microbatch,args.accumulate,args.overfit_steps)<1 or args.patience<0:raise ValueError('invalid budget')
    if args.data_min_free_gib<5 or args.system_min_free_gib<5:raise ValueError('disk hard floor must be at least 5GiB')
    if args.mode=='summarize':
        from j36m.reporting import summarize
        print(json.dumps(summarize(args.output),indent=2));return
    if args.mode=='status':
        print(json.dumps({str(f.relative_to(args.output)):json.loads(f.read_text()) for f in args.output.rglob('*status.json')},indent=2));return
    if args.mode=='stop':
        path=args.stop_file or args.output/'STOP';path.parent.mkdir(parents=True,exist_ok=True);path.touch(exist_ok=True);print(path);return
    args.stop_file=args.stop_file or args.output/'STOP'
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop_handler)
    if args.mode=='cpu-smoke':
        if args.output.exists():raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        command=[sys.executable,'-m','unittest','discover','-s','tests/j36m','-v']
        with (args.output/'tests.log').open('x') as f:subprocess.run(command,cwd=base,stdout=f,stderr=subprocess.STDOUT,check=True)
        publish(args,'cpu',args.output/'tests.log');print('CPU gate passed');return
    if args.mode=='calibration-template':
        if args.output.exists():raise FileExistsError(args.output)
        if not args.checkpoint:raise ValueError('checkpoint required')
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        value={'status':'calibration_only','checkpoint_sha256':digest(args.checkpoint),'training_manifest_sha256':digest(args.config),
               'weights':dict(zip(LAYERS,(.25,.25,.5))),'norm_floor':{k:1e-8 for k in LAYERS},'normal_history':[.01,.01],
               'state_settings':{'base_noise_floor':10.}}
        args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,value);return
    if args.mode=='calibrate':
        from j36m.calibration import fit_calibration
        trials=[]
        for f in args.output.glob('trials/*.json'):
            r=json.loads(f.read_text())
            if r['status']!='completed':continue
            directory=args.output/'attempts'/r['case']['id']/f'attempt_{r["attempt"]:03d}'
            trials.append({'condition':r['case']['condition'],'seed':r['case']['seed'],'normal_success':r['result']['success'],
                           'checkpoint_sha256':digest(args.checkpoint),'training_manifest_sha256':digest(args.config),
                           'rows':[json.loads(s) for s in (directory/'monitor_steps.jsonl').read_text().splitlines()]})
        frozen=fit_calibration(trials,digest(args.checkpoint),digest(args.config))
        path=args.output/'frozen_calibration.json'
        if path.exists():raise FileExistsError(path)
        write_json(path,frozen);print(path);return
    cfg=load_config(args.config)
    safe_output(args.output,cfg)
    if any(part in ('dev','object10','runtime') for part in args.output.resolve().parts):raise ValueError('new outputs must stay outside frozen experiment/code/environment trees')
    if args.mode=='preflight':
        # Read-only, never initializes CUDA or creates dataset outputs.
        report={'resources':resources(['/root','/root/autodl-tmp'],gpu=True),'params':{},'data70':json.loads(args.manifest.read_text())['summary']}
        for kind in ('J','C0','S2','S8'):
            model=make_model(kind);report['params'][kind]=sum(p.numel() for p in model.parameters());del model
        all_manifest=json.loads((base/'studies/policy_generalization/corrector/manifests/libero_object10_8695891.json').read_text())
        for record in all_manifest['records']:
            if not verified(Path(cfg['features_root'])/f'episode_{record["episode"]}'):raise FileNotFoundError('deep cache missing')
        report['verified_connector_episodes']=len(all_manifest['records'])
        report['existing_raw_verified']=len(validate(cfg))
        print(json.dumps(report,indent=2));return
    if args.mode=='plus':
        receipt(args,'cpu');receipt(args,'gpu_J');receipt(args,'station')
        if not args.checkpoint or not args.calibration:raise ValueError('actual trained J and calibration required')
        from j36m.plus import run_plus
        if args.phase in ('development','heldout'):receipt(args,'online_smoke')
        if args.phase=='heldout':receipt(args,'development')
        result=run_plus(args,make_guard(args,gpu=True))
        if result['status']=='completed' and args.phase in ('online-smoke','development'):
            publish(args,'online_smoke' if args.phase=='online-smoke' else 'development',args.output/'frozen_receipt.json')
        print(json.dumps(result,indent=2))
        if result['status']!='completed':raise ResourcePause('evaluation paused at trial boundary')
        return
    if args.mode=='extract70':
        receipt(args,'gpu_J');receipt(args,'station')
        from j36m.extraction import extract70
        manifest=json.loads(args.manifest.read_text());pending=sum(r['frames'] for r in cfg['episodes'] if not Path(r['raw_path']).exists())
        missing={f['path']:f['bytes'] for r in manifest['records'] for f in r['files'] if not (args.data_root/f['path']).is_file()}
        reserve=args.data_min_free_gib*2**30+pending*2*2*1024*768*2+8*2**30+sum(missing.values())
        if shutil.disk_usage('/root/autodl-tmp').free<reserve:raise ResourcePause('insufficient capacity for complete selection + reserve/checkpoints')
        extract70(args,cfg,manifest,make_guard(args,gpu=True));return
    records=validate(cfg)
    if args.mode=='verify70':
        manifest=json.loads(args.manifest.read_text())
        expected={(r['episode'],r['frames'],r['split'],r['task_id']) for r in manifest['records']}
        actual={(r['spec']['episode'],r['spec']['frames'],r['spec']['split'],r['spec']['task_id']) for r in records}
        if len(records)!=70 or actual!=expected:raise ValueError('formal training requires exactly the frozen balanced70 episodes')
        args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'verify.json',{'status':'passed','data':[r['provenance'] for r in records]})
        publish(args,'data70',args.output/'verify.json');return
    if args.mode=='cache-audit':
        from j36m.audit import feature_scale_audit
        if feature_scale_audit(records,args.output,args.stop_file,make_guard(args)) is None:raise ResourcePause('CPU audit paused at episode boundary; partial output retained')
        return
    receipt(args,'cpu')
    if args.mode=='station-audit':
        from j36m.stations import run_station_audit
        report=run_station_audit(args,cfg,records,make_guard(args,gpu=True))
        publish(args,'station',args.output/'station_audit.json');return
    if args.audit is None:raise ValueError('train-only cache audit is required')
    audit=json.loads(args.audit.read_text())
    if args.precision=='bf16' and (args.device!='cuda' or not torch.cuda.is_bf16_supported()):raise ValueError('BF16 requires capable CUDA')
    if args.mode=='gpu-smoke':
        if args.device!='cuda':raise ValueError('real GPU smoke cannot silently run on CPU')
        receipt(args,'station')
        from j36m.training import gpu_smoke
        report=gpu_smoke(args,records,audit,make_guard(args,gpu=True))
        publish(args,'gpu_'+args.variant,args.output/'smoke.json');return
    if args.mode=='train':
        data_gate=receipt(args,'data70')
        if json.loads(Path(data_gate['artifact']).read_text())['data']!=[r['provenance'] for r in records]:raise ValueError('formal data gate/content mismatch')
        receipt(args,'gpu_'+args.variant);receipt(args,'gpu_J');receipt(args,'station')
        if len(records)!=70:raise ValueError('no old 5-episode whitelist for formal training')
        from j36m.training import train
        train(args,cfg,records,audit,make_guard(args,gpu=args.device=='cuda'))

if __name__=='__main__':main()
