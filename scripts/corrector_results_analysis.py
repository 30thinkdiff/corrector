"""本地CPU结果审计：训练多种子、闭环配对、成本和失败索引；不加载策略。"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

from corrector_results_bundle import validate_bundle,sha


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def mean(values):return statistics.mean(values) if values else None
def finite(value):return isinstance(value,(int,float)) and math.isfinite(value)
def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

def discordance(pairs,baseline,candidate):
    rows=[p for p in pairs.values() if baseline in p and candidate in p]
    win=sum(p[candidate] and not p[baseline] for p in rows)
    loss=sum(p[baseline] and not p[candidate] for p in rows);n=win+loss
    return {'matched':len(rows),'candidate_only_success':win,'baseline_only_success':loss,
        'both_success':sum(p[candidate] and p[baseline] for p in rows),
        'both_failure':sum(not p[candidate] and not p[baseline] for p in rows),
        'net_pp':100*(win-loss)/len(rows) if rows else None,
        'exact_mcnemar_two_sided_unadjusted':min(1.,2*sum(math.comb(n,i) for i in range(min(win,loss)+1))/2**n) if n else 1.}


def residual_mse_from_norms(pred_norm,target_norm,score,dimension):
    if dimension<1 or min(pred_norm,target_norm)<0:raise ValueError('Invalid norm/dimension')
    cosine=max(-1.,min(1.,1-score))
    return max(0.,pred_norm**2+target_norm**2-2*pred_norm*target_norm*cosine)/dimension


def coverage_stats(scored,first_due,env_steps,stride,fps):
    if first_due<0 or env_steps<0 or stride<1 or fps<=0:raise ValueError('Invalid coverage timebase')
    eligible=set(range(first_due,env_steps,stride));raw=[r['step'] for r in scored]
    observed=set(raw)&eligible;longest=current=0
    for step in sorted(eligible):
        current=0 if step in observed else current+1;longest=max(longest,current)
    return {'eligible_score_steps':len(eligible),'scores':len(observed),
        'max_gap_steps':longest*stride,'max_gap_seconds':longest*stride/fps,
        'out_of_schedule_scores':len(set(raw)-eligible),'duplicate_scores':len(raw)-len(set(raw))}


class Audit:
    def __init__(self,archive,manifest):
        self.z,self.manifest=archive,manifest;self.findings=[];self.histories={};self.cohort_pairs={}
        self.training=[];self.groups=[];self.closed=[];self.paired=[];self.failures=[];self.reviews=[]
        self.tasks=[];self.traces=[];self.baselines=[];self.model_changes=[];self.evaluation_protocols=[];self.coverage=[];self.guidance_effects=[];self.online_prediction_metrics=[];self.cross_model_reviews=[];self.trace_members={}
    def load(self,name):return json.loads(self.z.read(name))
    def finding(self,severity,code,message):self.findings.append({'severity':severity,'code':code,'message':message})
    def train(self):
        reports=[]
        for name in self.manifest['members']:
            if name.endswith('/summary.json'):
                value=self.load(name)
                if 'checkpoint_sha256' in value and 'metrics' in value:reports.append((name,value))
        for item in self.manifest['runs']:
            files=item['files'];protocol=self.load(files['protocol.json']);name=Path(item['relative']).name
            label=item['root']+'/'+name
            kind=protocol.get('model_type','mlp');h=protocol.get('h_window',1);steps=protocol.get('action_steps',1)
            variant='mlp' if kind=='mlp' else ('transformer-a'+str(steps)+'-h'+str(h) if kind=='transformer-action' else 'transformer-h'+str(h))
            result=self.load(files['result.json']) if 'result.json' in files else {}
            status=self.load(files['training_status.json']) if 'training_status.json' in files else {}
            history=self.load(files['history.json']) if 'history.json' in files else []
            marker=self.load(files['JOB_COMPLETE.json']) if 'JOB_COMPLETE.json' in files else {}
            complete=status.get('status')=='completed' and result.get('checkpoint_reload_verified') is True
            if not complete:self.finding('error','training_incomplete',label+' lacks completed/reload evidence')
            weight=item.get('weight',{});digest=weight.get('sha256') or marker.get('checkpoint_sha256')
            if weight.get('sha256') and marker.get('checkpoint_sha256')!=weight['sha256']:
                self.finding('error','weight_sha',label+' JOB marker differs from actual best SHA')
            matching=[(n,r) for n,r in reports if r['checkpoint_sha256']==digest]
            report=matching[0][1] if len(matching)==1 else {}
            if len(matching)!=1:self.finding('error','report_identity',label+' missing/ambiguous matching best report')
            split=protocol['split']
            if 'split_indices.npz' in files:
                try:
                    import numpy as np
                    with np.load(io.BytesIO(self.z.read(files['split_indices.npz'])),allow_pickle=False) as indices:
                        for part in ('train','validation'):
                            array=indices[part].astype('<i8')
                            if len(array)!=split['samples'][part] or hashlib.sha256(array.tobytes()).hexdigest()!=split['index_sha256'][part]:
                                self.finding('error','index_file_hash',label+' '+part)
                        if np.intersect1d(indices['train'],indices['validation']).size:
                            self.finding('error','sample_leak',label)
                except ImportError:self.finding('warning','indices_not_verified',label+' NumPy unavailable')
            shared=set(split.get('episodes',{}).get('train',[])) & set(split.get('episodes',{}).get('validation',[]))
            if protocol.get('split',{}).get('mode')=='episode' and shared:self.finding('error','episode_leak',label+str(sorted(shared)))
            if report and report.get('split')!=split:self.finding('error','split_mismatch',label+' report split differs')
            best=min(history,key=lambda r:r['validation']['loss']) if history else {}
            if report and best and report.get('checkpoint_epoch')!=best['epoch']:
                self.finding('error','best_epoch',label+' exported best epoch differs from minimum validation loss')
            metrics=report.get('metrics',{});val=result.get('best_validation',{})
            flat_dimension=None
            if len(matching)==1:
                pair_name=str(Path(matching[0][0]).with_name('per_pair.csv'))
                if pair_name in self.manifest['members']:
                    estimates=[]
                    with self.z.open(pair_name) as binary:
                        for i,pair in enumerate(csv.DictReader(io.TextIOWrapper(binary))):
                            if float(pair['zero_mse'])>0:
                                estimates.append(float(pair['target_norm'])**2/float(pair['zero_mse']))
                            if i>=31:break
                    if estimates:flat_dimension=round(statistics.median(estimates))
                    if estimates and max(abs(d-flat_dimension) for d in estimates)>.2:
                        self.finding('error','feature_dimension_inconsistent',label)

            if metrics and finite(val.get('mse')) and not math.isclose(metrics['mse'],val['mse'],rel_tol=1e-4,abs_tol=1e-4):
                self.finding('error','reload_report_mismatch',label+' MSE differs from reloaded best result')
            epoch_seconds=[r['seconds'] for r in history if finite(r.get('seconds'))]
            row={'run':label,'variant':variant,'model_type':kind,'history_frames':h,'action_steps':steps,
                'seed':protocol['seed'],'flat_feature_dimension':flat_dimension,'completed':complete,'parameters':result.get('parameters',report.get('parameters')),
                'epochs':result.get('epochs_completed'),'optimizer_steps':result.get('optimizer_steps'),
                'best_epoch':report.get('checkpoint_epoch',best.get('epoch')),'mse':metrics.get('mse',val.get('mse')),
                'cosine':metrics.get('cosine',1-val['cosine_loss'] if 'cosine_loss' in val else None),
                'zero_mse':metrics.get('zero_mse',val.get('zero_prediction_mse')),
                'train_mean_mse':metrics.get('train_mean_mse'),'epoch_compute_seconds':sum(epoch_seconds),
                'mean_epoch_seconds':mean(epoch_seconds),'peak_cuda_gib':result.get('peak_cuda_gib'),
                'train_pairs':split['samples']['train'],'validation_pairs':split['samples']['validation'],
                'precision':protocol.get('precision','float32'),'micro_batch':protocol['batch_size'],
                'effective_batch':protocol['batch_size']*protocol['accumulate'],
                'loss_type':protocol.get('loss_type'),'beta':protocol.get('beta'),'lr':protocol.get('lr'),
                'weight_decay':protocol.get('weight_decay'),'epochs_max':protocol.get('epochs'),'patience':protocol.get('patience'),
                'anchor_history':protocol.get('anchor_history',1),'data_manifest_sha256':protocol.get('data_manifest_sha256'),
                'training_entry_sha256':protocol.get('training_entry_sha256'),'k_step':protocol['k_step'],
                'fps':protocol['fps'],'prediction_seconds':protocol['prediction_seconds'],
                'split_hash':json.dumps(split['index_sha256'],sort_keys=True),
                'policy_revision':protocol.get('encoder_policy_revision'),'source_sha':protocol['source_sha'],
                'model_factory_sha':protocol.get('model_factory_sha256'),'checkpoint_sha256':digest,
                'remote_best_path':weight.get('path'),'source_protocol_member':files['protocol.json']}
            self.training.append(row);self.histories[label]=history
            if finite(row['epochs']) and row['optimizer_steps']!=row['epochs']*math.ceil(row['train_pairs']/row['effective_batch']):
                self.finding('error','optimizer_steps',label+' step count differs from completed epoch/effective batch')
        for field in ('data_manifest_sha256','policy_revision','loss_type','beta','lr','weight_decay','epochs_max','patience','anchor_history','effective_batch','precision','k_step','fps'):
            if len({str(r.get(field)) for r in self.training})>1:self.finding('warning','control_setting_differs',field+' differs across runs')
        hashes={r['split_hash'] for r in self.training}
        if len(hashes)>1:self.finding('error','common_anchor_split','Models do not share identical frozen indices')
        for variant in sorted({r['variant'] for r in self.training}):
            rows=[r for r in self.training if r['variant']==variant and r['completed']]
            group={'variant':variant,'seeds':','.join(str(r['seed']) for r in rows),'n':len(rows)}
            for field in ('mse','cosine','epoch_compute_seconds','mean_epoch_seconds'):
                values=[r[field] for r in rows if finite(r[field])]
                group[field+'_mean']=mean(values);group[field+'_sample_std']=statistics.stdev(values) if len(values)>1 else None
            self.groups.append(group)
            if len(rows)!=3:self.finding('warning','seed_count',variant+' has '+str(len(rows))+' completed seeds')
        if len(self.training)!=15:self.finding('warning','run_count','Expected 15 training runs; observed '+str(len(self.training)))
    def evaluate(self):
        by_sha={r['checkpoint_sha256']:r for r in self.training}
        for item in self.manifest['evaluations']:
            if item['kind']!='paired':continue
            plan=self.load(item['files']['plan']);execution=self.load(item['files']['execution'])
            if plan['protocol'].get('debug_forced_event'):
                self.finding('error','forced_formal',item['relative']+' contains forced events; excluded');continue
            cases={c['id']:c for c in plan['cases']}
            if len(cases)!=len(plan['cases']):self.finding('error','duplicate_plan',item['relative'])
            records=[self.load(n) for n in item['trials']]
            digests={c['config'].get('corrector_sha256') for c in plan['cases'] if c['config'].get('uses_corrector')}
            known=[by_sha[s] for s in digests if s in by_sha]
            run=known[0] if len(known)==1 else None
            cohort=run['run'] if run else item['root']+'/'+item['relative']
            self.evaluation_protocols.append({'cohort':cohort,'protocol':plan['protocol'],'execution':execution})
            if run and plan['protocol'].get('policy_revision')!=run['policy_revision']:
                self.finding('error','policy_revision',cohort+' policy differs from feature encoder')
            if run and plan['protocol'].get('model_api_source_sha256') is not None and plan['protocol']['model_api_source_sha256']!=run['model_factory_sha']:
                self.finding('error','model_factory_version',cohort+' evaluation factory differs from training')
            if len(digests)!=1 or len(known)!=1:self.finding('error','evaluation_weight',cohort+' no unique known Corrector SHA')
            if execution.get('status')!='completed':self.finding('error','evaluation_incomplete',cohort+' '+execution.get('status','missing'))
            success_rows=[];outcomes={};bymethod={};task_rows={};review_records={}
            for record in records:
                case=record['case'];cid=case['id'];method=case['method'];key=(case['condition'],case['seed'])
                if cid not in cases or case!=cases[cid]:self.finding('error','trial_plan',cohort+' '+cid);continue
                if record['status']!='completed':
                    self.finding('error','trial_incomplete',cohort+' '+cid+' '+record['status']);continue
                result=record['result']
                if type(result.get('success')) is not bool:self.finding('error','success_type',cohort+' '+cid);continue
                success=result['success'];outcomes.setdefault(key,{})[method]=success
                review_records.setdefault(key,{})[method]=record
                bymethod.setdefault(method,[]).append(record);success_rows.append(record)
                condition=plan['protocol']['conditions'][case['condition']]
                task=condition['task_id'];language=condition.get('language')
                task_rows.setdefault((task,language,method),[]).append(success)
                videos=item['video_inventory'].get(cid,[])
                if not success:
                    self.failures.append({'cohort':cohort,'condition':key[0],'seed':key[1],'method':method,'trial_id':cid,
                        'env_steps':result.get('env_steps'),'event_replans':result.get('event_replans'),
                        'video_paths':json.dumps([v['remote_path'] for v in videos]),
                        'videos_present':all(v['present'] for v in videos) if videos else False})
                    if case['config'].get('video_policy')!='none' and (not videos or not all(v['present'] for v in videos)):
                        self.finding('warning','failure_video_missing',cohort+' '+cid)
                actual_member=item['actual_protocols'].get(cid)
                if actual_member:
                    actual=self.load(actual_member)
                    if case['config'].get('uses_corrector'):
                        pp=actual.get('predictor_protocol') or {}
                        expected=round(pp.get('prediction_seconds',-1)*actual.get('fps',plan['protocol']['fps']))
                        if actual.get('monitor_horizon_steps')!=expected:self.finding('error','horizon',cohort+' '+cid)
                        if run and pp.get('h_window')!=run['history_frames']:self.finding('error','history',cohort+' '+cid)
                        trace_member=item['traces'].get(cid)
                        if trace_member:
                            self.trace_members[(cohort,key,method)]=trace_member
                            trace=[json.loads(line) for line in self.z.read(trace_member).decode().splitlines() if line.strip()]
                            scored=[r for r in trace if r.get('score') is not None]
                            first=actual.get('monitor_first_due_step',expected)
                            if result.get('env_steps',0)>first and (not scored or scored[0]['step']!=first):
                                self.finding('warning','first_score',cohort+' '+cid+' first score differs from recorded due step')
                            guidance=sum('guidance_loss' in r for r in trace)
                            if guidance!=result.get('guidance_calls',0):self.finding('error','guidance_count',cohort+' '+cid)
                            coverage_detail=coverage_stats(scored,first,result.get('env_steps',0),
                                plan['protocol']['corrector_settings']['stride'],plan['protocol']['fps'])
                            if coverage_detail['out_of_schedule_scores'] or coverage_detail['duplicate_scores']:
                                self.finding('error','score_schedule',cohort+' '+cid)
                            self.traces.append({'cohort':cohort,'condition':key[0],'seed':key[1],'method':method,
                                **coverage_detail,'env_steps':result.get('env_steps'),
                                'trace_skips':sum('prediction_skipped' in r for r in trace),
                                'trace_cancellations':sum(r.get('cancelled_predictions',0)+r.get('cancelled_on_new_plan',0) for r in trace),
                                'first_score':scored[0]['step'] if scored else None,
                                'mean_score':mean([r['score'] for r in scored]),
                                'online_reconstructed_mse':mean([residual_mse_from_norms(r['expected_norm'],r['observed_norm'],r['score'],run['flat_feature_dimension'])
                                    for r in scored]) if scored and run and run.get('flat_feature_dimension') else None,
                                'online_zero_mse':mean([r['observed_norm']**2/run['flat_feature_dimension'] for r in scored])
                                    if scored and run and run.get('flat_feature_dimension') else None,
                                'trigger_flags':sum(bool(r.get('trigger')) for r in trace),
                                'guidance_calls':guidance,'sequence_prediction_skips':result.get('sequence_prediction_skips'),
                                'cancelled_predictions':result.get('cancelled_condition_predictions')})
                elif case['config'].get('uses_corrector'):self.finding('warning','actual_protocol_missing',cohort+' '+cid)
            if len(success_rows)!=execution.get('completed') or len(success_rows)!=len(cases):
                self.finding('error','completion_count',cohort+' recomputed='+str(len(success_rows))+' planned='+str(len(cases)))
            self.cohort_pairs[cohort]=outcomes
            for method in ('lvm','ogg'):
                trace_rows=[r for r in self.traces if r['cohort']==cohort and r['method']==method]
                eligible=sum(r['eligible_score_steps'] for r in trace_rows);scores=sum(r['scores'] for r in trace_rows)
                coverage={'cohort':cohort,'variant':run['variant'] if run else None,'method':method,
                    'eligible_steps':eligible,'scored_steps':scores,'coverage':scores/eligible if eligible else None,
                    'skips':sum(r['trace_skips'] for r in trace_rows),'cancellations':sum(r['trace_cancellations'] for r in trace_rows),
                    'max_gap_seconds':max((r['max_gap_seconds'] for r in trace_rows),default=None),
                    'mean_max_gap_seconds':mean([r['max_gap_seconds'] for r in trace_rows])}
                self.coverage.append(coverage)
                metric_rows=[r for r in trace_rows if r['scores'] and r.get('online_reconstructed_mse') is not None]
                count=sum(r['scores'] for r in metric_rows)
                if count:
                    online_mse=sum(r['online_reconstructed_mse']*r['scores'] for r in metric_rows)/count
                    online_zero=sum(r['online_zero_mse']*r['scores'] for r in metric_rows)/count
                    self.online_prediction_metrics.append({'cohort':cohort,'method':method,'scored_steps':count,
                        'online_cosine':1-sum(r['mean_score']*r['scores'] for r in metric_rows)/count,
                        'mse_reconstructed_from_norm_cosine':online_mse,'online_zero_mse':online_zero,
                        'mse_over_zero_mse':online_mse/online_zero if online_zero>0 else None,
                        'scope':'on-policy surviving scored windows; reconstruction from FP32 norm/cosine; different states and coverage, not matched planned-vs-executed causal test'})
                if eligible and scores/eligible<.95:self.finding('warning','monitor_coverage_loss',cohort+' '+method+' '+str(round(100*scores/eligible,2))+'% after warmup')
            comparable=[v for v in review_records.values() if 'lvm' in v and 'ogg' in v]
            self.guidance_effects.append({'cohort':cohort,'matched':len(comparable),
                **{field+'_changed_cases':sum(v['lvm']['result'].get(field)!=v['ogg']['result'].get(field) for v in comparable)
                   for field in ('success','env_steps','chunk_calls','event_replans')}})
            for method,rows in bymethod.items():
                n=len(rows);k=sum(r['result']['success'] for r in rows);lo,hi=wilson(k,n)
                out={'cohort':cohort,'variant':run['variant'] if run else None,'method':method,
                    'completed':n,'planned':sum(c['method']==method for c in cases.values()),'successes':k,
                    'success_rate':k/n,'wilson_low':lo,'wilson_high':hi,'checkpoint_sha256':run['checkpoint_sha256'] if run else None}
                for field in ('env_steps','chunk_calls','policy_s','monitor_s','event_replans','guidance_calls',
                              'sequence_prediction_skips','cancelled_condition_predictions','peak_torch_allocated_gib'):
                    values=[r['result'][field] for r in rows if finite(r['result'].get(field))]
                    out['mean_'+field]=mean(values)
                out['mean_episode_wall_s']=mean([r['wall_s'] for r in rows if finite(r.get('wall_s'))])
                steps=sum(r['result'].get('env_steps',0) for r in rows)
                out['monitor_ms_per_control_step']=1000*sum(r['result'].get('monitor_s',0) for r in rows)/steps if steps else None
                self.closed.append(out)
            for (task,language,method),values in task_rows.items():
                self.tasks.append({'cohort':cohort,'task_id':task,'language':language,'method':method,
                    'completed':len(values),'successes':sum(values),'success_rate':mean(values)})
            for baseline in ('H50','H20','lvm'):
                self.paired.append({'cohort':cohort,'baseline':baseline,'candidate':'ogg',**discordance(outcomes,baseline,'ogg')})
            for category,predicate in [('ogg_rescue_H50',lambda v:v.get('ogg') and v.get('H50') is False),
                 ('ogg_rescue_lvm',lambda v:v.get('ogg') and v.get('lvm') is False),
                 ('ogg_harm_lvm',lambda v:v.get('lvm') and v.get('ogg') is False),
                 ('all_methods_failed',lambda v:all(m in v and not v[m] for m in ('H50','H20','lvm','ogg')))]:
                selected=[k for k,v in outcomes.items() if predicate(v)][:3]
                for key in selected:
                    details=review_records[key]
                    self.reviews.append({'cohort':cohort,'category':category,'condition':key[0],'seed':key[1],
                        'outcomes':outcomes[key],'failure_videos':{m:[v['remote_path'] for v in item['video_inventory'].get(r['case']['id'],[])]
                            for m,r in details.items() if not r['result']['success']}})
        cohorts=sorted(self.cohort_pairs)
        if len(cohorts)!=5:self.finding('warning','formal_cohort_count','Expected 5 formal cohorts; observed '+str(len(cohorts)))
        for i,left in enumerate(cohorts):
            for right in cohorts[i+1:]:
                a,b=self.cohort_pairs[left],self.cohort_pairs[right]
                common=set(a)&set(b)
                for method in ('H50','H20'):
                    keys=[k for k in common if method in a[k] and method in b[k]]
                    different=sum(a[k][method]!=b[k][method] for k in keys)
                    self.baselines.append({'left':left,'right':right,'method':method,'matched':len(keys),'outcome_disagreements':different})
                    if different:self.finding('warning','baseline_disagreement',left+' vs '+right+' '+method+' '+str(different)+' cases differ')
                for method in ('lvm','ogg'):
                    pairs={k:{'left':a[k][method],'right':b[k][method]} for k in common if method in a[k] and method in b[k]}
                    self.model_changes.append({'left':left,'right':right,'method':method,**discordance(pairs,'left','right')})
        for baseline,candidate in [('night/transformer-h1_s42','actions/transformer-a10-h1_s42'),
                                   ('night/transformer-h3_s42','actions/transformer-a10-h3_s42')]:
            if baseline not in self.cohort_pairs or candidate not in self.cohort_pairs:continue
            a,b=self.cohort_pairs[baseline],self.cohort_pairs[candidate]
            for key in sorted(set(a)&set(b)):
                if a[key].get('ogg') is True and b[key].get('ogg') is False:
                    failure=next((f for f in self.failures if f['cohort']==candidate and f['condition']==key[0]
                                  and f['seed']==key[1] and f['method']=='ogg'),{})
                    traces={r['cohort']:r for r in self.traces if r['cohort'] in (baseline,candidate)
                            and r['condition']==key[0] and r['seed']==key[1] and r['method']=='ogg'}
                    self.cross_model_reviews.append({'baseline':baseline,'candidate':candidate,'condition':key[0],'seed':key[1],
                        'baseline_trace':self.trace_members.get((baseline,key,'ogg')),
                        'candidate_trace':self.trace_members.get((candidate,key,'ogg')),
                        'failure_videos':json.loads(failure.get('video_paths','[]')),
                        'baseline_env_steps':traces.get(baseline,{}).get('env_steps'),
                        'candidate_env_steps':traces.get(candidate,{}).get('env_steps'),
                        'scope':'matched initial case but subsequent trajectories differ; not equal-state prediction comparison'})

    def result(self):
        return {'created_utc':datetime.now(timezone.utc).isoformat(),'training_runs':self.training,'variant_summary':self.groups,
            'closed_loop':self.closed,'paired':self.paired,'baseline_reproducibility':self.baselines,
            'cross_model_changes':self.model_changes,'per_task':self.tasks,'trace_summary':self.traces,
            'monitor_coverage':self.coverage,'online_prediction_metrics':self.online_prediction_metrics,'guidance_effects':self.guidance_effects,'evaluation_protocols':self.evaluation_protocols,
            'review_candidates':self.reviews,'cross_model_regression_reviews':self.cross_model_reviews,'findings':self.findings,
            'counts':{'training_runs':len(self.training),'completed_training_runs':sum(r['completed'] for r in self.training),
                'formal_cohorts':len(self.cohort_pairs),'formal_completed_trials':sum(r['completed'] for r in self.closed)},
            'limitations':['Development validation selected the best checkpoint; not an independent predictor test set.',
                'Action-sequence inputs contain executed demonstration future actions offline and proposed policy actions online; guidance scope also changes.',
                'Wilson intervals and exact McNemar tests are descriptive/unadjusted; repeated seeds/initializations can be dependent.',
                'Baseline repeats across cohorts are not independent extra baseline samples.',
                'monitor_ms_per_control_step is a ratio of cumulative time to control steps, not GPU kernel latency or asynchronous exposed wait.',
                'Stored 10FPS/20Hz horizon conversion does not establish original acquisition physical-clock provenance.',
                'Online residual MSE is reconstructed from logged norms and cosine; surviving scored windows are selected and trajectories differ across models, so it is not an equal-state action-distribution test.',
                'Video existence checked remotely, contents not transferred or SHA rehashed; visual failure diagnosis remains pending.']}


def fmt(value,digits=4):return f'{value:.{digits}f}' if finite(value) else '—'
def markdown(result,bundle_hash):
    counts=result['counts'];lines=['# Object10训练与闭环结果审计','',
        f"证据包SHA256：`{bundle_hash}`。已读取{counts['training_runs']}组训练，其中{counts['completed_training_runs']}组有完成与重载证据；正式评测{counts['formal_cohorts']}组、{counts['formal_completed_trials']}回合。",'',
        '## 核验状态','']
    errors=[f for f in result['findings'] if f['severity']=='error']
    warnings=[f for f in result['findings'] if f['severity']=='warning']
    lines += [f'审计错误{len(errors)}项、警告{len(warnings)}项。统计按原始逐回合记录重算，强制调用检查不计正式成功率。','',
              '## 预测器开发验证：多种子','',
              '| 配置 | 种子数 | MSE均值 ± 样本标准差 | cosine均值 ± 样本标准差 | 每次完整epoch计算累计均值/分钟 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in result['variant_summary']:
        lines.append(f"| {row['variant']} | {row['n']} | {fmt(row['mse_mean'])} ± {fmt(row['mse_sample_std'])} | {fmt(row['cosine_mean'])} ± {fmt(row['cosine_sample_std'])} | {fmt(row['epoch_compute_seconds_mean']/60 if row['epoch_compute_seconds_mean'] is not None else None,2)} |")
    lines += ['', '这里的验证集参与best选优；多动作组离线条件包含后续实际执行动作，不能将更低MSE直接归因于架构，也不能据此承诺闭环改善。','',
              '## 正式闭环与计算成本','',
              '| 模型组 | 方法 | 成功/完成 | 成功率 | 动作块调用均值 | 策略秒/回合 | 监测秒/回合 | 墙钟秒/回合 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['closed_loop']:
        lines.append(f"| {row['cohort']} | {row['method']} | {row['successes']}/{row['completed']} | {100*row['success_rate']:.2f}% | {fmt(row['mean_chunk_calls'],2)} | {fmt(row['mean_policy_s'],3)} | {fmt(row['mean_monitor_s'],3)} | {fmt(row['mean_episode_wall_s'],3)} |")
    lines += ['', '## 在线监测覆盖率','',
        '| 模型组 | 方法 | 有效打分/预热后应监测步数 | 覆盖率 | 最长无评分间隔/秒 |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in result['monitor_coverage']:
        lines.append(f"| {row['cohort']} | {row['method']} | {row['scored_steps']}/{row['eligible_steps']} | {100*row['coverage']:.2f}% | {fmt(row['max_gap_seconds'],2)} |")
    lines += ['', '覆盖率分母排除初始预热和首次预测到期前的步数，按实际monitor stride统计。skips/cancellations不是缺失评分时刻的一一分割，不能直接相加替代缺失计数。',
        '完整10动作条件与计划失效撤销形成监测空窗；这支持条件可用性是重要机制因素，但没有等覆盖率对照时不能把全部成功率差异归因于它。', '']
    lines += ['', '## 配对救回与损失','',
              '| 模型组 | 对照 → 完整OGG | 配对数 | OGG独有成功 | 对照独有成功 | 净百分点 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result['paired']:
        lines.append(f"| {row['cohort']} | {row['baseline']} → OGG | {row['matched']} | {row['candidate_only_success']} | {row['baseline_only_success']} | {fmt(row['net_pp'],2)} |")
    lines += ['', 'McNemar数值保存于JSON/CSV，为探索性未校正统计；重复seed和初始化存在相关性。跨模型重复的H50/H20不合并为更大独立分母。', '',
              '## 解释边界与下一步','',
              '1. 先核对audit_findings.json中的错误及基线逐回合不一致，再判断模型差异。',
              '2. 用相同case的LVM与OGG配对得失，区分检测/中止收益与引导增量。',
              '3. 按review_candidates.json取回少量关键失败视频，核对误触发、抓空、掉落与恢复；本包没有视频内容。',
              '4. 同步监测耗时、策略调用耗时和真实墙钟时间分别报告，不等同论文异步暴露等待。',
              '5. 若离线预测改善而闭环无改善，优先检查动作分布差异、LVM阈值校准及引导方向；不立即扩大下载/重训。','',
              '## 审计发现','']
    for finding in result['findings'][:100]:lines.append(f"- [{finding['severity']}] {finding['code']}: {finding['message']}")
    if not result['findings']:lines.append('未发现本工具覆盖范围内的不一致；未验证视频内容、物理采集时钟或全程初态/噪声Tensor。')
    return '\n'.join(lines)+'\n'


def plots(result,histories,output,bundle):
    import os
    import warnings
    cache=output/'matplotlib-cache';cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache.resolve())
    import matplotlib
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore',message='Unable to import Axes3D.*',category=UserWarning)
    from matplotlib import pyplot as plt
    groups=result['variant_summary']
    if groups:
        figure,axes=plt.subplots(1,2,figsize=(13,4))
        labels=[r['variant'] for r in groups]
        for ax,field in zip(axes,['mse','cosine']):
            values=[r[field+'_mean'] for r in groups]
            errors=[r[field+'_sample_std'] or 0 for r in groups]
            ax.bar(range(len(groups)),values,yerr=errors,capsize=4)
            ax.set_xticks(range(len(groups)),labels,rotation=25,ha='right');ax.set_ylabel(field)
            ax.set_title('Development '+field+' / seed mean and sample SD')
        figure.tight_layout();figure.savefig(output/'predictor_validation.png',dpi=150);plt.close(figure)
        figure,axes=plt.subplots(3,2,figsize=(13,10));axes=list(axes.flat)
        for ax,group in zip(axes,groups):
            rows=[r for r in result['training_runs'] if r['variant']==group['variant']]
            for row in rows:
                history=histories[row['run']]
                ax.plot([h['epoch'] for h in history],[h['validation']['loss'] for h in history],label='seed '+str(row['seed']))
            ax.set_title(group['variant']);ax.set_xlabel('Completed epoch');ax.set_ylabel('Validation loss');ax.legend()
        for ax in axes[len(groups):]:ax.set_visible(False)
        figure.tight_layout();figure.savefig(output/'validation_curves.png',dpi=150);plt.close(figure)
    closed=result['closed_loop']
    if closed:
        cohorts=list(dict.fromkeys(r['cohort'] for r in closed));methods=['H50','H20','lvm','ogg']
        short={'mlp':'MLP','transformer-h1':'T-H1','transformer-h3':'T-H3','transformer-a10-h1':'A10-H1','transformer-a10-h3':'A10-H3'}
        colors={'H50':'gray','H20':'tab:blue','lvm':'tab:green','ogg':'tab:red'}
        figure,axes=plt.subplots(1,2,figsize=(14,5));width=.19
        for j,method in enumerate(methods):
            rates=[next((100*r['success_rate'] for r in closed if r['cohort']==c and r['method']==method),float('nan')) for c in cohorts]
            axes[0].bar([i+(j-1.5)*width for i in range(len(cohorts))],rates,width=width,label=method,color=colors[method])
        axes[0].set_xticks(range(len(cohorts)),[short.get(c.split('/')[-1].rsplit('_s',1)[0],c) for c in cohorts],rotation=20,ha='right')
        axes[0].set_ylabel('Success %');axes[0].set_ylim(0,105);axes[0].legend()
        for row in closed:
            if finite(row['mean_episode_wall_s']):
                axes[1].scatter(row['mean_episode_wall_s'],100*row['success_rate'],marker='o' if row['method']=='ogg' else 'x',color=colors[row['method']])
                if row['method']=='ogg':axes[1].annotate(short.get(row['variant'],row['variant']),(row['mean_episode_wall_s'],100*row['success_rate']),xytext=(-3,6),textcoords='offset points',ha='right',fontsize=9)
        axes[1].set_xlabel('Mean episode wall seconds');axes[1].set_ylabel('Success %')
        axes[0].set_title('Matched cases; repeated baselines not pooled');axes[1].set_title('Success and end-to-end cost')
        axes[1].margins(x=.12,y=.15)
        figure.tight_layout();figure.savefig(output/'success_and_cost.png',dpi=150);plt.close(figure)


    coverage=[r for r in result['monitor_coverage'] if r['method']=='ogg' and r['coverage'] is not None]
    if coverage:
        figure,axes=plt.subplots(1,2,figsize=(12,4))
        labels=[r['variant'] for r in coverage]
        axes[0].bar(range(len(labels)),[100*r['coverage'] for r in coverage]);axes[0].set_ylabel('Scored / eligible steps %');axes[0].set_ylim(0,110)
        axes[1].bar(range(len(labels)),[r['max_gap_seconds'] for r in coverage]);axes[1].set_ylabel('Longest no-score gap seconds')
        for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right')
        axes[0].set_title('After common warmup / OGG rollouts');axes[1].set_title('20 Hz control / recorded traces')
        figure.tight_layout();figure.savefig(output/'monitor_coverage.png',dpi=150);plt.close(figure)


    candidates=[r for r in result['cross_model_regression_reviews'] if r['baseline_trace'] and r['candidate_trace']]
    if candidates:
        case=candidates[0]
        figure,axes=plt.subplots(2,1,figsize=(13,6),sharex=True)
        with zipfile.ZipFile(bundle) as archive:
            for ax,role in zip(axes,['baseline','candidate']):
                rows=[json.loads(x) for x in archive.read(case[role+'_trace']).decode().splitlines()]
                n=case[role+'_env_steps'];scores={r['step']:r for r in rows if r.get('score') is not None}
                time=[step/20 for step in range(n)]
                ax.plot(time,[scores.get(step,{}).get('score',float('nan')) for step in range(n)],label='Mismatch')
                ax.plot(time,[scores.get(step,{}).get('threshold',float('nan')) for step in range(n)],label='Threshold (bootstrap=10 off-scale)')
                visible=[r['score'] for r in scores.values()]+[r['threshold'] for r in scores.values() if finite(r.get('threshold')) and r['threshold']<=2]
                ax.set_ylim(0,max(1.05,max(visible,default=1)*1.15))
                for row in rows:
                    if row.get('interrupted'):ax.axvline(row['step']/20,color='red',alpha=.35)
                start=None
                for step in range(24,n+1):
                    if step<n and step not in scores:
                        if start is None:start=step
                    elif start is not None:
                        if step-start>=3:ax.axvspan(start/20,step/20,color='gray',alpha=.2)
                        start=None
                ax.set_title(case[role]+' / '+('success' if role=='baseline' else 'failure'))
                ax.set_ylabel('Mismatch / threshold');ax.legend(loc='upper left')
        axes[-1].set_xlabel('Control-step time seconds; shaded = no score after warmup; red = interruption')
        figure.suptitle(case['condition']+' seed '+str(case['seed'])+' / matched start, different trajectories')
        figure.tight_layout();figure.savefig(output/'coverage_case_example.png',dpi=150);plt.close(figure)


def analyze(bundle,output,plot=True):
    manifest=validate_bundle(bundle);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        audit=Audit(archive,manifest);audit.train();audit.evaluate();result=audit.result()
    result['bundle_sha256']=sha(bundle)
    write(output/'analysis.json',result);write(output/'audit_findings.json',result['findings'])
    write(output/'review_candidates.json',result['review_candidates'])
    write(output/'cross_model_regression_reviews.json',result['cross_model_regression_reviews'])
    for name,rows in [('training_runs',result['training_runs']),('variant_summary',result['variant_summary']),
        ('closed_loop',result['closed_loop']),('paired_comparisons',result['paired']),
        ('cross_model_baselines',result['baseline_reproducibility']),('cross_model_changes',result['cross_model_changes']),
        ('per_task_success',result['per_task']),('trace_summary',result['trace_summary']),
        ('monitor_coverage',result['monitor_coverage']),('online_prediction_metrics',result['online_prediction_metrics']),('guidance_effects',result['guidance_effects']),('failure_index',audit.failures)]:
        csv_write(output/(name+'.csv'),rows)
    (output/'RESULTS.md').write_text(markdown(result,result['bundle_sha256']))
    if plot:
        try:plots(result,audit.histories,output,bundle)
        except (ImportError,ValueError,TypeError) as error:write(output/'plots_unavailable.json',{'reason':str(error),'metrics_complete':True})
    print(json.dumps({'output':str(output),'counts':result['counts'],
        'errors':sum(f['severity']=='error' for f in result['findings']),
        'warnings':sum(f['severity']=='warning' for f in result['findings'])},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    analyze(args.bundle,args.output,not args.no_plots)
