"""Final read-only synthesis over verified completed snapshots; no model loading."""
import argparse,collections,datetime,hashlib,json,zipfile
from pathlib import Path
from corrector_results_analysis import csv_write
from corrector_progress_report import LABELS


def seed_review(discovery,replications,selected):
    """Exact condition/method/eval-seed pairing; repetitions are not new scenes."""
    conditions=[r['condition'] for r in selected]
    if len(conditions)!=len(set(conditions)):raise ValueError('duplicate selected condition')
    lookup={}
    for r in discovery+replications:
        if r['status']!='completed' or type(r.get('result',{}).get('success')) is not bool:raise ValueError('all input trials must be completed booleans')
        c=r['case'];key=c['condition'],c['method'],c['seed']
        if key in lookup:raise ValueError('duplicate trial key')
        lookup[key]=r
    methods=sorted({r['case']['method'] for r in discovery});seeds=sorted({r['case']['seed'] for r in discovery+replications})
    if seeds!=[1000,1001,1002]:raise ValueError('unexpected frozen evaluation seeds')
    for row in selected:
        if set(row['outcomes'])!=set(methods):raise ValueError('selection method set differs')
        for m,outcome in row['outcomes'].items():
            if lookup[row['condition'],m,row['seed']]['result']['success']!=outcome:raise ValueError('selection differs from discovery record')
    cases=[];summary=[];candidates=[]
    for c in sorted(conditions):
        for m in methods:
            if not all((c,m,s) in lookup for s in seeds):raise ValueError('incomplete three-seed pairing')
            values=[lookup[c,m,s]['result']['success'] for s in seeds]
            cases.append({'condition':c,'method':m,**{f'seed{s}':int(v) for s,v in zip(seeds,values)},
                          'successes':sum(values),'consistent_all_three':len(set(values))==1,'majority_success':sum(values)>=2})
        for a,b in [('H20','j_ogg'),('mlp_ogg','j_ogg'),('planned_lvm','executed_lvm')]:
            if not {a,b}<=set(methods):continue
            wins=[s for s in seeds if lookup[c,a,s]['result']['success'] and not lookup[c,b,s]['result']['success']]
            losses=[s for s in seeds if lookup[c,b,s]['result']['success'] and not lookup[c,a,s]['result']['success']]
            if len(wins)>=2 or len(losses)>=2:
                candidates.append({'condition':c,'method_a':a,'method_b':b,'a_only_success_seeds':wins,'b_only_success_seeds':losses,
                    'selection':'repeated outcome contrast; failure type not visually labeled',
                    'trial_refs':[{'seed':s,'method':m,'case_id':lookup[c,m,s]['case']['id'],'attempt':lookup[c,m,s]['attempt'],
                                   'video_artifacts':lookup[c,m,s].get('video_artifacts',[])} for s in seeds for m in [a,b]]})
    for m in methods:
        rows=[r for r in cases if r['method']==m]
        summary.append({'method':m,'label':LABELS.get(m,m),'conditions':len(conditions),
            **{f'seed{s}_successes':sum(r[f'seed{s}'] for r in rows) for s in seeds},
            'new_seed_successes':sum(r['seed1001']+r['seed1002'] for r in rows),'new_seed_trials':2*len(rows),
            'consistent_all_three_conditions':sum(r['consistent_all_three'] for r in rows),
            'majority_success_conditions':sum(r['majority_success'] for r in rows)})
    return {'summary':summary,'cases':cases,'replay_candidates':candidates}


def review(archive,snapshot,integrity,output,ordinary_summary=None):
    snapshot=Path(snapshot).resolve();output=Path(output).resolve();audit=json.loads(Path(integrity).read_text())
    if audit.get('gpu_compute_processes'):raise ValueError('final integrity audit has active GPU processes')
    for p in audit['phases'].values():
        if p['status']!='completed' or p['completed']!=p['video_artifacts_verified'] or not p['actual_protocol_sha_matches_plan']:raise ValueError('integrity not verified')
    with zipfile.ZipFile(archive) as z:
        def read(n):return json.loads(z.read(n))
        counts={};records={}
        for phase in ['ordinary/ordinary','plus/plus','plus/plus_replicates','ordinary/bridge','plus/plus_smoke']:
            execution=read(phase+'/execution.json')
            rows=[read(n) for n in z.namelist() if n.startswith(phase+'/trials/') and n.endswith('.json')]
            if execution['status']!='completed' or execution['errors'] or len(rows)!=execution['planned'] or any(r['status']!='completed' for r in rows):raise ValueError('refuse to publish final review for incomplete stage')
            if audit['phases'][phase]['plan_sha256']!=hashlib.sha256(z.read(phase+'/plan.json')).hexdigest():raise ValueError('audit does not match source plan')
            counts[phase]=len(rows);records[phase]=rows
        queue=read('plus/queue_status.json')
        if queue['status']!='completed' or queue.get('returncode')!=0 or queue.get('error'):raise ValueError('queue did not exit successfully')
        if 'plus/results/status.json' in z.namelist() and read('plus/results/status.json')['status']!='completed':raise ValueError('worker is not completed')
        selected=read('plus/results/replication_selection.json')['selected_groups']
        result=seed_review(records['plus/plus'],records['plus/plus_replicates'],selected)
    output.mkdir(parents=True,exist_ok=False)
    csv_write(output/'seed_reproducibility.csv',result['summary']);csv_write(output/'same_condition_three_seeds.csv',result['cases'])
    write=lambda n,v:(output/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write('replay_candidates.json',result['replay_candidates']);write('integrity.json',audit)
    metrics=json.loads((snapshot/'summary.json').read_text());paired=[]
    for phase,rows in records.items():
        if phase in ['ordinary/bridge','plus/plus_smoke']:continue
        lookup={(r['case']['condition'],r['case']['seed'],r['case']['method']):r['result']['success'] for r in rows}
        keys={(c,s) for c,s,m in lookup}
        paired.append({'phase':phase,'matched_j_pairs':len(keys),'different_j_lvm_ogg_outcomes':sum(lookup[c,s,'j_lvm']!=lookup[c,s,'j_ogg'] for c,s in keys)})
    ordinary_summary=Path(ordinary_summary) if ordinary_summary else Path(__file__).resolve().parents[1]/'studies/policy_generalization/results/20261113/j36m_expanded_object/summary.json'
    ordinary_data=json.loads(ordinary_summary.read_text())
    if not ordinary_data.get('historical_bridge_passed'):raise ValueError('ordinary historical bridge not verified')
    summary={'source_zip_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'ordinary_summary_sha256':hashlib.sha256(ordinary_summary.read_bytes()).hexdigest(),
             'snapshot':str(snapshot),'counts':counts,'integrity':audit,'seed_review':result['summary'],'j_ogg_pairs':paired,
             'bounds':['discovery and outcome-selected replication remain separate','evaluation seeds are not training seeds or independent physical scenes',
                       'model MLP454 vs J70 training data differ','success/failure does not label each step as abnormal']}
    write('summary.json',summary)
    rel=Path(__import__('os').path.relpath(snapshot,output)).as_posix()
    lines=['# 扩展评测最终整理 · 2026-11-14','',
        '本次活动评测队列已全部正常完成；worker于北京时间00:13:27写入completed，调度器00:13:33正常退出。最终状态及独立校验见`integrity.json`。旧训练继续暂停，未自动恢复或启动额外GPU任务。','',
        f'[完整指标快照]({rel}/RESULTS.md) · [全部17方法首轮指标]({rel}/plus_plus/methods.csv) · [复测指标]({rel}/plus_plus_replicates/methods.csv) · [同条件三种子表](seed_reproducibility.csv)。','',
        '## 完成范围与数据完整性','',
        f"新增性能/诊断完整回合{counts['ordinary/ordinary']+counts['plus/plus']+counts['plus/plus_replicates']}：普通J补测{counts['ordinary/ordinary']}、Plus首轮{counts['plus/plus']}、Plus诊断复测{counts['plus/plus_replicates']}；另有桥接{counts['ordinary/bridge']}、冒烟{counts['plus/plus_smoke']}，合计957份完成视频均通过SHA核验，实际协议SHA均匹配冻结计划。32次环境reset+一步检查不计作完整任务回合。",'',
        '普通六方法还复用了已桥接验证的历史240回合，未重跑全部基线。关机中断的旧attempt与状态收据保留，恢复后只重做未完成回合。原始视频/动作/Trace仍保留远端；此次同步只有指标与轻量Trace，不声称新视频已全部导入本地Replay。','',
        '## 主要终点结果（不同集合分开）','',
        '| 方法 | 普通Object10 | Plus首轮32条件 | Plus选例复测16条件/seed组 |','|---|---:|---:|---:|']
    ordinary={m:(ordinary_data['methods'][m]['successes'],ordinary_data['methods'][m]['completed']) for m in ['H50','H20','mlp_lvm','mlp_ogg','j_lvm','j_ogg']}
    for m in ordinary:
        cells=[LABELS[m],f'{ordinary[m][0]}/{ordinary[m][1]}']
        for phase in ['plus/plus','plus/plus_replicates']:
            r=next(r for r in metrics['phases'][phase]['methods'] if r['method']==m);cells.append(f"{r['successes']}/{r['completed']}")
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','Plus首轮H20占优；复测MLP两方法13/16、H20 12/16、J两方法4/16。复测按首轮方法差异选了8条件，用eval seed1001/1002，不是新的随机总体抽样；不能将13/16与11/32直接解释为改进，也不合成48样本总体成功率。两基础任务的扰动条件也不是32个独立任务。','',
            '## 同一批8条件：固定权重，改变评测种子','',
            '| 方法 | seed1000 /8 | seed1001 /8 | seed1002 /8 | 三次结局全一致的条件 /8 |','|---|---:|---:|---:|---:|']
    for r in result['summary']:lines.append(f"| {r['label']} | {r['seed1000_successes']} | {r['seed1001_successes']} | {r['seed1002_successes']} | {r['consistent_all_three_conditions']} |")
    lines+=['','每个seed是评测随机性（策略采样及环境随机因素），不是重新训练的seed。固定init并换seed不增加物理状态数量；同一条件三个seed也可能有不同结局，不能把单次救回视为稳定收益。详见逐条件三种子CSV。','',
            '## LVM、OGG与预测准度','',
            f"J-LVM/J-OGG在普通60组、Plus首轮32组及复测16组共{sum(r['matched_j_pairs'] for r in paired)}组配对中，终点结局翻转共{sum(r['different_j_lvm_ogg_outcomes'] for r in paired)}次。OGG真实运行并留下非零梯度/动作更新；终点一致不等于动作相同，也不能说OGG未调用。",'',
            '- J评分覆盖在上述评测中为100%、最长额外空窗0（24步预热另计）；A10未来动作条件仍约50%覆盖、最长19步空窗。planned新规则首轮覆盖39.60%，executed为100%，终点分别10/32与11/32；复测5/16与7/16，不能仅凭覆盖率推出因果提升。',
            '- J首轮事件均值LVM2.78、OGG2.41，少于MLP的4.97/4.59；这是不同闭环轨迹上的描述，不足以证明漏检。无独立逐窗异常标签，不能给出真实漏检率/误报率。',
            '- J/C0等70条数据离线MSE、归一化MSE与/zero比值在完整快照training_layers.csv。J的Connector5.211879，C0 5.149472；S2 Block2 .019093优于J对应.023722，但模型目标/容量分配/选优目标有差别，不能直接断言共享主干造成负迁移。',
            '- 在线J逐层MSE由日志RMS/cosine推导，仅有效评分窗口，并按步聚合；长失败轨迹权重更大。单动作旧模型缺足够显式维度时不伪造可比MSE。误差与成功相关可能是失败后果，不能自动解释为原因。',
            '- 表中eval耗时、callback耗时、CUDA/诊断同步、监测与OGG不同口径均保留；论文异步额外等待不等于这里同步推理墙钟。','',
            '## 优先复盘的已有案例（不新增实验）','']
    for c in result['replay_candidates']:
        if c['method_a']=='H20' and len(c['a_only_success_seeds'])==3:
            lines.append(f"- `{c['condition']}`：H20三种seed均成功、J-OGG均失败，适合检查J触发时机与纠偏后的轨迹。")
    lines+=['- `plus/soup/language_instructions/1`：J在seed1000成功、1001/1002失败，MLP-OGG三次均成功，适合检查随机性与同条件诊断。',
            '- `plus/cheese/sensor_noise/1`及`plus/soup/light_conditions/1`：H20/MLP-OGG三次均成功，J只有一次成功。',
            '', '候选清单含原case ID、attempt、视频路径与SHA，仅标记真实结局差异，不预判抓取失败类型。按需取少量视频/动作，不复制全量高维张量。','',
            '## 下一步建议（尚未执行）','',
            '1. 先用上述稳定反例复盘检测评分、动态阈值、目标与梯度，不继续以多seed/长epoch堆训练时长。',
            '2. 若做新消融，分别冻结检测器/阈值与引导器，控制单变量；保持计划与已有权重只读，先小Pilot。',
            '3. 研究多层误差的校准/融合与动作分布偏移；现有结果不支持直接把J称为更强Corrector，也不支持用更低离线MSE代替闭环收益。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--integrity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=review(a.archive,a.snapshot,a.integrity,a.output);print(json.dumps({'output':str(a.output),'counts':d['counts'],'j_pairs':d['j_ogg_pairs']}))
