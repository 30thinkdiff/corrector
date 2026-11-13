"""Read-only streaming scale audit. Exact moments, explicitly sampled quantiles."""
import csv,json
from pathlib import Path
import numpy as np
from corrector_data_workbench import write_json
from multilayer_mvp.data import LAYERS,MultiLayerTransitionDataset


class Accumulator:
    def __init__(self):
        self.n=0;self.total=0.;self.sq=0.;self.samples=[];self.camera=None;self.token=None;self.channel=None;self.max_abs=0.
    def add(self,value,salt):
        x=np.asarray(value,dtype=np.float32)
        self.max_abs=max(self.max_abs,float(np.abs(x).max()))
        self.n+=x.size;self.total+=float(x.sum(dtype=np.float64));self.sq+=float(np.square(x,dtype=np.float64).sum())
        # Seeded random offset per stratum/frame; not an exact population quantile.
        flat=x.reshape(-1);n=min(2048,len(flat));stride=max(1,len(flat)//n)
        offsets=np.random.default_rng(salt).integers(0,stride,size=n)
        index=np.minimum(np.arange(n)*stride+offsets,len(flat)-1)
        self.samples.append(np.abs(flat[index]).copy())
        cams=x.reshape(2,-1,x.shape[-1]);squared=np.square(cams,dtype=np.float64)
        c=squared.sum(axis=(1,2));t=squared.sum(axis=2);d=squared.sum(axis=(0,1))
        if self.camera is None:self.camera=np.zeros_like(c);self.token=np.zeros_like(t);self.channel=np.zeros_like(d)
        self.camera+=c;self.token+=t;self.channel+=d
    def report(self):
        mean=self.total/self.n;frames=self.n/(self.token.size*len(self.channel))
        return {'elements':self.n,'mean':mean,'std':max(0,self.sq/self.n-mean**2)**.5,'rms':(self.sq/self.n)**.5,'max_abs':self.max_abs,
                'abs_quantiles_sampled':dict(zip(['p50','p90','p99'],np.quantile(np.concatenate(self.samples),[.5,.9,.99]).tolist())),
                'quantile_method':'deterministic stratified <=2048 elements/frame, exact moments over all elements',
                'per_camera_rms':np.sqrt(self.camera/(frames*self.token.shape[1]*len(self.channel))).tolist(),
                'per_token_rms':np.sqrt(self.token/(frames*len(self.channel))).tolist(),
                'per_channel_rms':np.sqrt(self.channel/(frames*self.token.size)).tolist()}


def frame(record,key,t):
    a=record['arrays']
    return (a['z_q'][t].astype('float32')*a['z_scale'][t].astype('float32')[:,None]
            if key=='connector' else a[key][t].astype('float32'))


def feature_scale_audit(records,output,stop_file=None,resource_guard=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    splits={};episodes=[]
    for r in records:
        if resource_guard:resource_guard()
        if stop_file and Path(stop_file).exists():
            write_json(output/'status.json',{'status':'paused','boundary':'episode','completed_episodes':len(episodes)})
            return None
        split=r['spec']['split'];acc=splits.setdefault(split,{k:{'values':Accumulator(),'delta':Accumulator()} for k in LAYERS})
        e={'episode':r['spec']['episode'],'task_id':r['spec']['task_id'],'split':split,'layers':{}}
        for k in LAYERS:
            level,delta=Accumulator(),Accumulator();outliers=[];n=r['spec']['frames']
            for t in range(n):
                x=frame(r,k,t)
                if not np.isfinite(x).all():raise FloatingPointError(f'{e["episode"]}/{k}/{t}')
                level.add(x,e['episode']*1000003+t);acc[k]['values'].add(x,e['episode']*1000003+t)
                if t+10<n:
                    d=frame(r,k,t+10)-x;delta.add(d,e['episode']*1000003+t);acc[k]['delta'].add(d,e['episode']*1000003+t)
                    outliers.append((float(np.square(d,dtype=np.float64).mean()**.5),t))
            e['layers'][k]={'values':level.report(),'delta':delta.report(),
                            'largest_delta_rms_windows':[{'t':t,'rms':v} for v,t in sorted(outliers,reverse=True)[:5]]}
        episodes.append(e)
        write_json(output/'status.json',{'status':'running','completed_episodes':len(episodes),'last_episode':e['episode']})
    report={'format':'feature_scale_audit_v1','source':[r['provenance'] for r in records],
            'split':{s:{k:{kind:v.report() for kind,v in pair.items()} for k,pair in row.items()} for s,row in splits.items()},
            'episodes':episodes,'station_audit':'not executed by CPU cache audit; separate same-frame GPU station audit required'}
    report['train_only_input_rms']={k:report['split']['train'][k]['values']['rms'] for k in LAYERS}
    report['train_only_delta_rms']={'delta_'+k:report['split']['train'][k]['delta']['rms'] for k in LAYERS}
    for split,row in report['split'].items():
        for k,r in row.items():
            r['zero_raw_mse']=r['delta']['rms']**2
            r['zero_normalized_mse']=r['zero_raw_mse']/(report['train_only_delta_rms']['delta_'+k]**2+1e-12)
            r['delta_to_value_rms_ratio']=r['delta']['rms']/max(r['values']['rms'],1e-12)
    write_json(output/'feature_scale_audit.json',report)
    with (output/'scale_summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['split','layer','value_mean','value_std','value_rms','delta_rms','zero_raw_mse','zero_normalized_mse'])
        for split,row in report['split'].items():
            for k,r in row.items():w.writerow([split,k,*(r['values'][v] for v in ['mean','std','rms']),r['delta']['rms'],r['zero_raw_mse'],r['zero_normalized_mse']])
    write_json(output/'status.json',{'status':'completed','episodes':len(episodes),'gpu_used':False})
    return report
