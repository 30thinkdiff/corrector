"""Read-only remote metrics snapshot + local Chinese report; no GPU/model loading.

Raw ZIPs and derived reports are versioned. Running-stage summaries only use
completed trial records and never merge selected replications with discovery.
"""
import argparse,collections,datetime,hashlib,io,json,math,shlex,statistics,subprocess,sys,zipfile
from pathlib import Path
from corrector_results_analysis import csv_write,coverage_stats,discordance,mean

REPO=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import json,hashlib,datetime,io,zipfile,shutil,sys
base=Path('/root/autodl-tmp/corrector');q=Path('/root/corrector_plus_expanded_20261113_001')
buf=io.BytesIO();members={};phases={};total=0
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 def add(name,path=None,raw=None):
  global total
  if name in members:raise ValueError('duplicate export member')
  raw=path.read_bytes() if raw is None else raw
  total+=len(raw)
  if len(raw)>24*2**20 or total>180*2**20:raise ValueError('metadata budget exceeded')
  z.writestr(name,raw);members[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 for group,root in [('ordinary',Path('/root/corrector_expanded_object_20261113_001')),('plus',q)]:
  for name in ['status.json','queue_status.json','queue_plan.json','bridge_receipt.json','ordinary_receipt.json','results/status.json','results/replication_selection.json']:
   p=root/name
   if p.exists():add(group+'/'+name,p)
  for phase in ['bridge','ordinary'] if group=='ordinary' else ['plus_smoke','plus','plus_replicates']:
   rootphase=root/phase if group=='ordinary' else root/'results'/phase
   if not(rootphase/'plan.json').exists():continue
   prefix=group+'/'+phase+'/'
   for name in ['plan.json','execution.json']:
    p=rootphase/name
    if p.exists():add(prefix+name,p)
   counts={}
   for p in sorted((rootphase/'trials').glob('*.json')):
    raw=p.read_bytes();r=json.loads(raw);counts[r['status']]=counts.get(r['status'],0)+1
    add(prefix+'trials/'+p.name,raw=raw)
    if r['status']!='completed':continue
    rel=Path('attempts')/r['case']['id']/('attempt_%03d'%r['attempt'])
    for name in ['actual_protocol.json','monitor_steps.jsonl','chunk_calls.json']:
     f=rootphase/rel/name
     if f.exists():add(prefix+(rel/name).as_posix(),f)
   phases[group+'/'+phase]=counts
 for p in sorted((base/'j36m_longrun_20261113_001/training').iterdir()):
  if p.is_dir():
   for name in ['protocol.json','status.json','result.json','metrics.csv','history.json']:
    f=p/name
    if f.is_file():add('training/'+p.name+'/'+name,f)
 live={}
 for label,pid,script,out in [('queue',json.loads((q/'queue_status.json').read_text()).get('pid'),'corrector_plus_queue.py',q),
                              ('worker',json.loads((q/'results/status.json').read_text()).get('pid'),'corrector_expanded_eval.py',q/'results')]:
  try:args=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0');alive=any(Path(a).name==script for a in args) and str(out) in args
  except OSError:alive=False
  live[label]={'pid':pid,'process_matches':alive}
 snapshot={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phases':phases,'live':live,
           'free_gib':{p:shutil.disk_usage(p).free/2**30 for p in ['/root','/root/autodl-tmp']},
           'scope':'metadata and completed traces only; no weights/actions/video/Tensor download; no remote writes',
           'members':members}
 z.writestr('snapshot.json',json.dumps(snapshot,ensure_ascii=False,indent=2))
sys.stdout.buffer.write(buf.getvalue())
'''

LABELS={'H50':'H50','H20':'H20','mlp_lvm':'MLP-LVM','mlp_ogg':'MLP-OGG','j_lvm':'J-LVM','j_ogg':'J-OGG',
        'hybrid_t1_a10_ogg':'T-H1检测→A10-H1引导','planned_lvm':'A10规划动作LVM（新规则）','executed_lvm':'A10已执行动作LVM（新规则）'}
for tag,name in [('transformer-h1','T-H1'),('transformer-h3','T-H3'),('transformer-a10-h1','A10-H1'),('transformer-a10-h3','A10-H3')]:
    for mode in ['lvm','ogg']:LABELS[tag+'_'+mode]=name+'-'+mode.upper()


def finite(x):return type(x) in (int,float) and math.isfinite(x)
def fmt(x,precision=3):return '未记录/不适用' if x is None else f'{x:.{precision}f}'
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def summarize_phase(z,prefix):
    read=lambda n:json.loads(z.read(prefix+n))
    plan=read('plan.json');execution=read('execution.json');cases={c['id']:c for c in plan['cases']}
    records=[read(n[len(prefix):]) for n in z.namelist() if n.startswith(prefix+'trials/') and n.endswith('.json')]
    ids=[r['case']['id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate trial id')
    for r in records:
        if r['case']['id'] not in cases or r['case']!=cases[r['case']['id']]:raise ValueError('trial differs from frozen plan')
        if r['status']=='completed' and type(r.get('result',{}).get('success')) is not bool:raise ValueError('completed requires boolean success')
    completed=[r for r in records if r['status']=='completed'];methods=sorted({c['method'] for c in plan['cases']})
    paired=collections.defaultdict(dict);method_rows=[];case_rows=[];guide_rows=[];layer_rows=[]
    for r in completed:
        c=r['case'];result=r['result'];key=(c['condition'],c['seed']);m=c['method']
        if m in paired[key]:raise ValueError('duplicate method/condition/seed')
        paired[key][m]=result['success']
        directory=f"attempts/{c['id']}/attempt_{r['attempt']:03d}/";actual=read(directory+'actual_protocol.json')
        trace=prefix+directory+'monitor_steps.jsonl'
        rows=[json.loads(line) for line in z.read(trace).splitlines()] if trace in z.namelist() else []
        baseline=m in ['H50','H20'];first=actual.get('monitor_first_due_step',plan['protocol'].get('first_due_step'))
        cov=None
        if not baseline and type(first) is int:
            cov=coverage_stats([row for row in rows if finite(row.get('score'))],first,result['env_steps'],1,20)
        meta=plan['protocol'].get('conditions',{}).get(c['condition'],{})
        case_rows.append({'method':m,'condition':c['condition'],'seed':c['seed'],'category':meta.get('category','Object10'),
            'success':result['success'],'attempt':r['attempt'],'callback_wall_s':r.get('wall_s'),**{k:result.get(k) for k in ['env_steps','chunk_calls','policy_s','monitor_s','episode_wall_s','event_replans','guidance_calls','sequence_prediction_skips','cancelled_condition_predictions','peak_torch_allocated_gib']},
            'score_steps':cov['scores'] if cov else None,'eligible_steps':cov['eligible_score_steps'] if cov else None,
            'coverage':cov['scores']/cov['eligible_score_steps'] if cov and cov['eligible_score_steps'] else None,
            'max_gap_steps':cov['max_gap_steps'] if cov else None,'trial_member':prefix+'trials/'+c['id']+'.json'})
        for row in rows:
            loss=row.get('ogg_loss',row.get('guidance_loss',row.get('loss') if row.get('event')=='joint_ogg' else None))
            if finite(loss):guide_rows.append({'method':m,'condition':c['condition'],'seed':c['seed'],'step':row['step'],'loss':loss,
                'gradient_l2':row.get('gradient_l2',row.get('gradient_norm')),
                'clean_action_update_l2':row.get('clean_action_update_l2',row.get('candidate_action_update_norm')),
                'velocity_update_l2':row.get('velocity_update_l2',row.get('velocity_update_norm')),
                'guidance_seconds':row.get('guidance_seconds',row.get('latency_s'))})
            for layer,item in (row.get('layer_scores') or {}).items():
                p,o,s=item.get('predicted_rms'),item.get('observed_rms'),item.get('score')
                if item.get('valid') and all(finite(v) for v in [p,o,s]):
                    layer_rows.append({'method':m,'layer':layer,'success':result['success'],'mse_from_recorded_rms_cosine':max(0.,p*p+o*o-2*p*o*(1-s)),
                                       'target_energy':o*o,'cosine_error':s})
    common={k:v for k,v in paired.items() if set(v)==set(methods)}
    for m in methods:
        rows=[r for r in case_rows if r['method']==m];done=len(rows);scored=[r for r in rows if r['eligible_steps'] is not None]
        denom=sum(r['eligible_steps'] for r in scored);guides=[r for r in guide_rows if r['method']==m]
        row={'method':m,'label':LABELS.get(m,m),'planned':sum(c['method']==m for c in plan['cases']),
             'completed':done,'successes':sum(r['success'] for r in rows),'errors':sum(r['status']=='error' and r['case']['method']==m for r in records),
             'success_rate':sum(r['success'] for r in rows)/done if done else None,'common_paired_n':len(common),
             'common_paired_successes':sum(v[m] for v in common.values()),'coverage':sum(r['score_steps'] for r in scored)/denom if denom else None,
             'max_gap_steps':max((r['max_gap_steps'] for r in scored),default=None),'guide_rows_recorded':len(guides)}
        row['max_peak_torch_allocated_gib']=max((r['peak_torch_allocated_gib'] for r in rows if finite(r.get('peak_torch_allocated_gib'))),default=None)
        for k in ['env_steps','chunk_calls','policy_s','monitor_s','episode_wall_s','callback_wall_s','event_replans','guidance_calls','sequence_prediction_skips','cancelled_condition_predictions','peak_torch_allocated_gib']:
            row['mean_'+k]=mean([r[k] for r in rows if finite(r.get(k))])
        for k in ['loss','gradient_l2','clean_action_update_l2','guidance_seconds']:
            row['mean_ogg_'+k]=mean([g[k] for g in guides if finite(g.get(k))])
        method_rows.append(row)
    categories=[]
    for category in sorted({r['category'] for r in case_rows}):
        for m in methods:
            values=[r for r in case_rows if r['category']==category and r['method']==m]
            categories.append({'category':category,'method':m,'completed':len(values),'successes':sum(r['success'] for r in values),
                               'success_rate':sum(r['success'] for r in values)/len(values) if values else None})
    comparisons=[]
    for a,b in [('H50','H20'),('H50','mlp_ogg'),('H20','j_ogg'),('mlp_lvm','mlp_ogg'),('j_lvm','j_ogg'),('mlp_ogg','j_ogg'),('planned_lvm','executed_lvm')]:
        if {a,b}<=set(methods):comparisons.append({'baseline':a,'candidate':b,**discordance(paired,a,b)})
    layers=[]
    for m,l,s in sorted({(r['method'],r['layer'],r['success']) for r in layer_rows}):
        values=[r for r in layer_rows if (r['method'],r['layer'],r['success'])==(m,l,s)]
        mse=mean([r['mse_from_recorded_rms_cosine'] for r in values]);energy=mean([r['target_energy'] for r in values])
        layers.append({'method':m,'layer':l,'trial_success':s,'scored_steps':len(values),'derived_mse':mse,
                       'observed_delta_energy':energy,'mse_over_observed_energy':mse/energy if energy else None,
                       'mean_cosine_error':mean([r['cosine_error'] for r in values])})
    return {'execution':execution,'snapshot_completed':len(completed),'snapshot_errors':sum(r['status']=='error' for r in records),
            'snapshot_running':sum(r['status']=='running' for r in records),'common_paired_groups':len(common),
            'completed_callback_seconds':sum(r['wall_s'] for r in completed if finite(r.get('wall_s'))),
            'methods':method_rows,'cases':case_rows,'categories':categories,'comparisons':comparisons,'guidance':guide_rows,'prediction_layers':layers}


def report(archive,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)!=len(set(z.namelist())) or any(i.file_size>24*2**20 for i in infos) or sum(i.file_size for i in infos)>181*2**20:
            raise ValueError('duplicate ZIP member or metadata size exceeds budget')
        snapshot=json.loads(z.read('snapshot.json'))
        if set(z.namelist())!=set(snapshot['members'])|{'snapshot.json'}:raise ValueError('snapshot membership differs')
        for n,item in snapshot['members'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=item['sha256']:raise ValueError('snapshot hash differs: '+n)
        phases={}
        for n in sorted(n for n in z.namelist() if n.endswith('/plan.json') and n.startswith(('plus/plus/','plus/plus_smoke/','plus/plus_replicates/','ordinary/ordinary/','ordinary/bridge/'))):
            prefix=n[:-len('plan.json')];phases[prefix.rstrip('/')]=summarize_phase(z,prefix)
        training=[];training_layers=[]
        for n in sorted(n for n in z.namelist() if n.startswith('training/') and n.endswith('/protocol.json')):
            prefix=n[:-len('protocol.json')];proto=json.loads(z.read(n));status=json.loads(z.read(prefix+'status.json'))
            result=json.loads(z.read(prefix+'result.json')) if prefix+'result.json'in z.namelist() else {}
            history=json.loads(z.read(prefix+'history.json')) if prefix+'history.json'in z.namelist() else []
            epoch_times=[r['seconds'] for r in history if finite(r.get('seconds'))]
            training.append({'run':prefix.split('/')[1],'variant':proto.get('variant'),'seed':proto.get('seed'),
                'status':result.get('status',status.get('status')),'epochs_completed':result.get('epochs_completed',status.get('epochs_completed')),
                'best_epoch':result.get('best_epoch',status.get('best_epoch')),'parameters':result.get('parameters'),
                'optimizer_steps':result.get('optimizer_steps',status.get('optimizer_steps')),'checkpoint_reload_verified':result.get('checkpoint_reload_verified'),
                'peak_cuda_allocated_gib':result.get('peak_cuda_allocated_gib'),'mean_epoch_s':mean(epoch_times),
                'epoch_compute_seconds':sum(epoch_times) if epoch_times else None,
                'protocol_sha256':snapshot['members'][n]['sha256']})
            for layer,metric in result.get('best_validation',{}).get('overall',{}).items():training_layers.append({'run':prefix.split('/')[1],'layer':layer,**metric})
        csv_write(output/'training_runs.csv',training);csv_write(output/'training_layers.csv',training_layers)
        for name,data in phases.items():
            directory=output/name.replace('/','_');directory.mkdir()
            for key in ['methods','cases','categories','comparisons','guidance','prediction_layers']:csv_write(directory/(key+'.csv'),data[key])
        write(output/'summary.json',{'snapshot':snapshot,'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'phases':{k:{a:b for a,b in v.items() if a not in ['cases','guidance']} for k,v in phases.items()},'training':training,'training_layers':training_layers})
        lines=['# 最新实验结果快照','',f"数据截止：{snapshot['captured_utc']}（UTC；北京时间加8小时）。原始元数据ZIP SHA256：`{hashlib.sha256(Path(archive).read_bytes()).hexdigest()}`。",'',
               '各阶段独立统计：冒烟不并入正式性能，按结局选取的复测不并入首轮。未完成不是失败；覆盖率以协议首个可评分步为分母，不按首个观测评分反推。','',
               '## 新70条Episode训练对照','', '| 模型 | 状态 | 完成轮数 | 最佳轮 | 参数量 | Checkpoint重载 |','|---|---|---:|---:|---:|---|']
        for row in training:lines.append(f"| {row['run']} | {row['status']} | {row['epochs_completed']} | {row['best_epoch']} | {row['parameters'] or '未记录'} | {row['checkpoint_reload_verified']} |")
        lines+=['','每层验证指标见`training_layers.csv`：原始MSE、归一化MSE、zero-delta MSE、相对zero比值、cosine及样本数。原始MSE不可跨层直接横比。S8未完成，不将其已有best当作正式训练结果。','',
                '| 模型 | 输出层 | MSE | 归一化MSE | /zero MSE | cosine | 验证样本 |','|---|---|---:|---:|---:|---:|---:|']
        for row in training_layers:lines.append(f"| {row['run']} | {row['layer']} | {fmt(row.get('mse'),6)} | {fmt(row.get('normalized_mse'))} | {fmt(row.get('relative_mse_vs_zero'))} | {fmt(row.get('cosine'))} | {row.get('samples')} |")
        for name,d in phases.items():
            lines+=['',f'## {name}', '',f"状态：`{d['execution']['status']}`；本快照完成{d['snapshot_completed']}回合，程序错误{d['snapshot_errors']}，运行中记录{d['snapshot_running']}；全部方法共同完成条件/seed组数：{d['common_paired_groups']}。",'',
                    '| 方法 | 完成/计划 | 成功 | 成功率 | 环境步 | VLA块调用 | 重规划 | OGG调用 | VLA秒 | 监测秒 | 回合秒 | 平均显存峰值GiB | 评分覆盖 | 最长空窗步 |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for row in d['methods']:
                rate=f"{row['success_rate']:.2%}" if row['success_rate'] is not None else '—'
                cov=f"{row['coverage']:.2%}" if row['coverage'] is not None else '不适用/未知'
                cells=[row['label'],f"{row['completed']}/{row['planned']}",str(row['successes']),rate]
                cells += [fmt(row.get('mean_'+k),2) for k in ['env_steps','chunk_calls','event_replans','guidance_calls','policy_s','monitor_s','episode_wall_s','peak_torch_allocated_gib']]
                lines.append('| '+' | '.join(cells+[cov,str(row['max_gap_steps']) if row['max_gap_steps']is not None else '不适用/未知'])+' |')
            lines+=['',f"详细CSV位于`{name.replace('/','_')}/`：methods（含OGG Loss/梯度/动作更新摘要）、cases、categories、comparisons、guidance及prediction_layers（实际有记录才生成）。",'']
            if d['guidance']:
                lines+=['| OGG方法 | 真实记录调用数 | 平均Loss | 梯度L2 | 动作更新L2 | 单次记录耗时秒 |',
                        '|---|---:|---:|---:|---:|---:|']
                for row in d['methods']:
                    if row['guide_rows_recorded']:lines.append('| '+' | '.join([row['label'],str(row['guide_rows_recorded'])]+[fmt(row['mean_ogg_'+k],6) for k in ['loss','gradient_l2','clean_action_update_l2','guidance_seconds']])+' |')
                lines+=['','Loss与更新量只用于各自协议内诊断，不作为跨视觉空间的统一分数。']
            if d['snapshot_completed']!=d['execution'].get('planned') or d['execution']['status']!='completed':lines.append('**本阶段仍未完成。各方法完成条件可能不同；公平横比使用methods.csv的common_paired_n/common_paired_successes，若为0则暂不排名。**')
            if name=='plus/plus':
                lines+=['','### 扰动类别成功率（首轮，非官方全任务基准）','', '| 类别 | H50 | H20 | MLP-LVM | MLP-OGG | J-LVM | J-OGG |','|---|---:|---:|---:|---:|---:|---:|']
                for category in sorted({r['category'] for r in d['categories']}):
                    lookup={r['method']:r for r in d['categories'] if r['category']==category}
                    lines.append('| '+' | '.join([category]+[f"{lookup[m]['successes']}/{lookup[m]['completed']}" for m in ['H50','H20','mlp_lvm','mlp_ogg','j_lvm','j_ogg']])+' |')
        lines+=['','## 解读边界与历史结果入口','',
                '- [普通LIBERO六方法总对比](../../20261113/j36m_expanded_object/RESULTS.md)；这里ordinary阶段只含新J两方法，桥接不是正式成功率实验。',
                '- [历史15组训练/1200回合](../../20261111/RESULTS.md)和其Horizon/混合/已执行监测Pilot子目录；不同训练数据量、精度或阈值协议不混合。',
                '- 表中回合秒为result.episode_wall_s（eval调用计时）；methods.csv另存mean_callback_wall_s，含环境准备/回合导出，部分旧服务端overall.csv使用此口径。execution.wall_seconds仅本次调用，不能当关机恢复前后的全程时长；summary另存已完成callback时长合计。',
                '- 监测时间、VLA块生成时间与回合墙钟不同；OGG采集延迟含同步和诊断写入，VLA块计时可能包含OGG，不能简单相加；非论文异步额外等待时间。',
                '- OGG Loss及梯度不能跨模型空间直接横比；无触发就是0调用，未记录的梯度为缺失，不补0。',
                '- J逐层预测MSE由已记录RMS与cosine推导，仅统计有效评分窗口，按步汇总会偏重较长轨迹；成功/失败相关性不证明预测误差造成失败，缺独立异常标签不计算检测AUROC。',
                '- comparisons.csv中的检验值为探索性、未做多重比较校正，且两基础任务的多扰动条件存在相关性，不将其当独立场景的确证显著性。',
                '- 本地是取回时的快照，运行中最新汇总在远端对应阶段`analysis/overall.csv`；进程存活用健康检查器核验，不能只看旧running。']
        (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return {'output':str(output),'captured_utc':snapshot['captured_utc'],'phases':{k:v['snapshot_completed'] for k,v in phases.items()}}


def publish_index(output):
    """Only replace the derived navigation page; versioned raw/report stays intact."""
    output=Path(output).resolve();summary=json.loads((output/'summary.json').read_text());snapshot=summary['snapshot']
    root=REPO/'studies/policy_generalization/results';link=output.relative_to(root).as_posix()
    captured=datetime.datetime.fromisoformat(snapshot['captured_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    main=summary['phases'].get('plus/plus');repeat=summary['phases'].get('plus/plus_replicates')
    finished=all(d is not None and d['execution']['status']=='completed' and d['snapshot_completed']==d['execution'].get('planned') and d['snapshot_errors']==0 for d in [main,repeat])
    lines=['# 实验结果总入口','',f"本地快照截止：**{captured.strftime('%Y-%m-%d %H:%M:%S')} 北京时间**。本地文件不会自动追随服务器变化。",'',
        f'- [本次完整指标报告]({link}/RESULTS.md)：新70条数据训练对照、普通LIBERO、Plus首轮17方法、复测快照和逐类结果。',
        f'- [机器可读摘要]({link}/summary.json) · [新训练汇总]({link}/training_runs.csv) · [逐层验证指标]({link}/training_layers.csv)。','',
        '| 实验 | 本次状态/规模 | 阅读入口 |','|---|---|---|',
        '| 历史五模型训练与闭环 | 15组训练、1200正式回合；已完成 | [历史总报告](20261111/RESULTS.md) |',
        '| SmolVLA H50/H20/H10/H5/H2/H1 | 24回合小样本；已完成 | [Horizon Pilot](20261111/horizon_pilot/RESULTS.md) |',
        '| 单动作/A10混合检测与引导 | 完整Pilot；已完成 | [Hybrid Pilot](20261111/hybrid_pilot/RESULTS.md) |',
        '| 已执行动作被动监测 | 4回合；已完成，不干预 | [被动诊断](20261111/executed_consistency_pilot/RESULTS.md) |',
        '| 已执行监测闭环对照 | 48回合；已完成 | [闭环Pilot](20261111/consistency_closed_loop/RESULTS.md) |',
        '| 夜间扩展与预测精度诊断 | 核心600已完成，整体2486/3400按时限暂停 | [夜间报告](20261112/overnight_review/RESULTS.md) |',
        '| Replay成败组合样本库 | 119/660按研究价值暂停；诊断样本非总体成功率 | [覆盖审计](20261112/casebank_research_v2/RESULTS.md) |',
        '| J预测保留策略Pilot | 小样本、不同早期协议；已完成 | [保留预测Pilot](20261113/j36m_retention_preserve/RESULTS.md) |',
        '| Object10正式六方法 | 旧240＋新J120；已完成，桥接另4次 | [六方法报告](20261113/j36m_expanded_object/RESULTS.md) |',
        f'| 70条数据新训练 | J/C0/J_equal/S2已完成，S8在24轮暂停 | [训练指标]({link}/training_layers.csv) |']
    for label,key in [('Plus环境/资产冒烟','plus/plus_smoke'),('Plus首轮（正式选定子集）','plus/plus'),('Plus有限诊断复测','plus/plus_replicates')]:
        d=summary['phases'].get(key)
        if d:lines.append(f"| {label} | {d['execution']['status']}；{d['snapshot_completed']}/{d['execution'].get('planned','未记录')}，错误{d['snapshot_errors']} | [分方法CSV]({link}/{key.replace('/','_')}/methods.csv) |")
    if main:
        lines+=['','## Plus首轮关键结果','', '| 方法 | 成功 | 成功率 | 覆盖率 |','|---|---:|---:|---:|']
        for m in ['H50','H20','mlp_lvm','mlp_ogg','j_lvm','j_ogg']:
            r=next(row for row in main['methods'] if row['method']==m)
            cov=f"{r['coverage']:.2%}" if r['coverage']is not None else '不适用'
            rate=f"{r['success_rate']:.2%}" if r['success_rate']is not None else '未完成'
            lines.append(f"| {r['label']} | {r['successes']}/{r['completed']} | {rate} | {cov} |")
        lines+=['','当前选定Plus子集中H20领先，J尚未显示终点收益；J覆盖率完整不等于任务成功率提高。两基础任务、多类扰动不代表官方全量Plus基准。复测按首轮差异选条件，单独报告，不并入首轮。','',
                f'- [全部17方法关键指标]({link}/plus_plus/methods.csv)：步数、VLA调用、重规划、OGG调用、VLA/监测/回合计时、显存、覆盖率、空窗、梯度和更新量。',
                f'- [扰动类别]({link}/plus_plus/categories.csv) · [逐回合]({link}/plus_plus/cases.csv) · [严格配对差异]({link}/plus_plus/comparisons.csv)。',
                f'- [J逐层预测误差]({link}/plus_plus/prediction_layers.csv)：来自已记录RMS/cosine的推导，仅有效评分窗口，不是全量离线预测。']
    finals=sorted(root.glob('*/expanded_final_review_*/RESULTS.md'))
    if finals:lines+=['',f"[最终复盘：配对得失、三种子复现与Replay候选]({finals[-1].relative_to(root).as_posix()})。"]
    lines+=['','## '+('当前队列已完成：历史指标与状态核查' if finished else '正在运行的实验在哪里看'),'',
            '在**远程终端**运行只读健康检查，确认实际进程，而非只读旧running：','',
            '```bash','/root/autodl-tmp/corrector/bootstrap/bin/python /root/autodl-tmp/corrector/dev/replay-health-60bd07ea8d57.py','```','',
            '以下为已完成阶段的最终指标，未自动启动后续任务：' if finished else '复测实时指标每个完整回合返回后更新（不会每个控制步更新）：','',
            '```bash','cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/execution.json',
            'cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/analysis/overall.csv',
            'tail -n 25 /root/corrector_plus_expanded_20261113_001/worker.log','```','',
            '首轮完整表：同一路径将`plus_replicates`换成`plus`。`analysis/trials.csv`逐回合、`analysis/summary.csv`逐条件；实际原始trial与monitor日志在`trials/`和`attempts/`。','',
            '在**本地Ubuntu终端**刷新本报告及CSV（需要可用SSH连接，密码只在终端输入；只读远程、不启动实验）：','',
            '```bash','python3 scripts/corrector_progress_report.py sync','```','',
            '本次不下载视频/权重/大张量，因此不会自动给Replay补齐所有Plus视频。按需另选少量病例导入。你之前打开的`multilayer_mvp.../remote_results_001`是早期5条数据/5轮MVP，不能当作本次70条正式训练结果。']
    tmp=root/'LATEST.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(root/'LATEST.md')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['sync','report']);p.add_argument('--archive',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--ssh-host',default='autodl',help='本机SSH配置别名；真实主机名只保存在私有SSH配置中')
    p.add_argument('--socket',type=Path,default=REPO/'tools/robotlab_replay/.local/ssh/autodl-casebank.sock');a=p.parse_args()
    if a.mode=='sync':
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder=REPO/'outputs/corrector_progress_snapshots';folder.mkdir(exist_ok=True)
        archive=folder/(stamp+'.zip');partial=archive.with_suffix('.zip.partial')
        if archive.exists():raise FileExistsError(archive)
        command=['ssh','-T','-S',str(a.socket),'-o','ConnectTimeout=15','-o','ServerAliveInterval=30',a.ssh_host,shlex.join(['python3','-c',REMOTE])]
        with partial.open('xb') as f:r=subprocess.run(command,stdout=f)
        if r.returncode:raise RuntimeError('SSH snapshot failed; partial preserved; no report published')
        partial.rename(archive)
    else:
        if a.archive is None:p.error('report requires --archive')
        archive=a.archive
    with zipfile.ZipFile(archive) as z:captured=datetime.datetime.fromisoformat(json.loads(z.read('snapshot.json'))['captured_utc'])
    stamp=captured.strftime('%Y%m%dT%H%M%SZ');day=captured.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d')
    output=a.output or REPO/'studies/policy_generalization/results'/day/('expanded_snapshot_'+stamp)
    result=report(archive,output)
    if output.resolve().is_relative_to(REPO/'studies/policy_generalization/results'):publish_index(output)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
