"""逐回合宏平均预测误差与成功率；中途快照只用已完成回合，不把窗口当独立样本。"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def auc(values,labels):
    positive=sum(labels);negative=len(labels)-positive
    if not positive or not negative:return None
    ordered=sorted(zip(values,labels));rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum+=sum(label for _,label in ordered[i:j])*(i+1+j)/2;i=j
    return (rank_sum-positive*(positive+1)/2)/(positive*negative)


def correlation(x,y):
    if len(x)<3:return None
    a=statistics.mean(x);b=statistics.mean(y)
    xx=sum((v-a)**2 for v in x);yy=sum((v-b)**2 for v in y)
    return sum((v-a)*(w-b) for v,w in zip(x,y))/math.sqrt(xx*yy) if xx and yy else None


def write(path,rows):
    if not rows:return
    with path.with_suffix(path.suffix+'.tmp').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    path.with_suffix(path.suffix+'.tmp').replace(path)


def analyse(root):
    root=Path(root);records=[]
    for path in (root/'trials').glob('*.json'):
        r=json.loads(path.read_text())
        if r['status']=='completed':records.append(r)
    output=root/'analysis';output.mkdir(exist_ok=True,parents=True)
    rows=[];samples=[];by_case=defaultdict(dict)
    for r in records:
        case,result=r['case'],r['result'];key=(case['condition'],case['seed']);by_case[key][case['method']]=result
        for m in result['prediction_summary']:
            samples.append({'id':case['id'],'condition':case['condition'],'seed':case['seed'],'method':case['method'],
                'task_id':int(case['condition'].split('/task_')[1].split('/')[0]),'success':result['success'],
                'env_steps':result['env_steps'],**m})
    methods=sorted({r['case']['method'] for r in records})
    for method in methods:
        sub=[r for r in records if r['case']['method']==method]
        row={'method':method,'completed':len(sub),'successes':sum(r['result']['success'] for r in sub)}
        row['success_rate']=row['successes']/row['completed']
        for k in ['env_steps','chunk_calls','policy_s','monitor_s','rollout_wall_s','shared_capture_s','diagnostic_seconds',
                  'event_replans','guidance_calls','peak_torch_allocated_gib']:
            row['mean_'+k]=statistics.mean(r['result'][k] for r in sub)
        eligible=sum(r['result']['eligible_steps'] for r in sub)
        row['score_coverage']=sum(r['result']['scored_steps'] for r in sub)/eligible if eligible else None
        rows.append(row)
    groups=defaultdict(list)
    for r in samples:
        for task in ['all',str(r['task_id'])]:
            groups[(r['model'],r['condition_role'],r['scope'],r['method'],task)].append(r)
    relations=[]
    for key,sub in sorted(groups.items()):
        model,role,scope,method,task=key
        usable=[r for r in sub if scope!='early' or r['early_prefix_complete']]
        if not usable:continue
        errors=[r['mse'] for r in usable];failures=[int(not r['success']) for r in usable]
        ok=[r for r in usable if r['success']];bad=[r for r in usable if not r['success']]
        relations.append({'model':model,'role':role,'scope':scope,'method':method,'task':task,'episodes':len(usable),
            'physical_initial_states':len({r['condition'] for r in usable}),
            'successes':len(ok),'failures':len(bad),'excluded_incomplete_early':len(sub)-len(usable),
            'episode_macro_mse':statistics.mean(errors),
            'episode_macro_cosine_loss':statistics.mean(r['cosine_loss'] for r in usable),
            'success_mean_mse':statistics.mean(r['mse'] for r in ok) if ok else None,
            'failure_mean_mse':statistics.mean(r['mse'] for r in bad) if bad else None,
            'point_biserial_error_vs_failure':correlation(errors,failures),
            'episode_failure_auc_from_mse':auc(errors,failures),
            'limit':'descriptive episode-outcome association; repeated initial states/seeds correlated; not anomaly detection or causation'})
    paired=[]
    for candidate in [m for m in methods if m!='H50']:
        matched=[v for v in by_case.values() if 'H50' in v and candidate in v]
        gains=sum(v[candidate]['success'] and not v['H50']['success'] for v in matched)
        losses=sum(v['H50']['success'] and not v[candidate]['success'] for v in matched)
        paired.append({'reference':'H50','candidate':candidate,'matched':len(matched),
            'candidate_only_success':gains,'reference_only_success':losses,'net_successes':gains-losses})
    write(output/'overnight_overall.csv',rows);write(output/'episode_prediction_metrics.csv',samples)
    write(output/'prediction_success_relationship.csv',relations);write(output/'overnight_paired.csv',paired)
    summary={'completed':len(records),'methods':rows,'paired':paired,
        'prediction_scope':'same actual trajectory for all 5 frozen seed42 predictors; model does not necessarily drive policy',
        'early_prefix':'origin4..44 (due24..64), omit incomplete prefix from early associations',
        'important_limits':['complete-episode error can be consequence of failure/episode length',
            'early prefix also not causal proof; task/method stratification supplied',
            'AUC classifies episode failure, not per-step anomaly; no labels invented',
            'only matched completed conditions enter policy comparisons',
            'diagnostics batched after rollout; batch1 latency is separately measured',
            'rollout includes passive shared visual capture, not native pureVLA runtime']}
    path=output/'overnight_summary.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
    report=['# 夜间扩展实验快照','',f'已完成 {len(records)} 回合；未完成/程序错误不计为机器人失败。','',
        '| 方法 | 完成 | 成功 | 成功率 | 平均完整块调用 | rollout秒（含被动采集） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:report.append(f"| {r['method']} | {r['completed']} | {r['successes']} | {r['success_rate']:.2%} | {r['mean_chunk_calls']:.2f} | {r['mean_rollout_wall_s']:.2f} |")
    report+=['','预测误差见episode_prediction_metrics.csv；关系按模型、条件角色、完整/固定早期前缀、控制方法及任务分层，见prediction_success_relationship.csv。',
        '五个模型都对相同真实轨迹评分；这不等于五个模型分别控制机器人。规划条件兑现/改变分开，不混入已执行条件。',
        '完整回合误差可能是失败结果；窗口高度相关。早期前缀固定仍不能证明因果。此处AUC仅指回合失败关联，不是异常检测AUROC。',
        'H50/H10也额外采集视觉供相同轨迹诊断；本轮rollout耗时不直接代替此前无额外采集的原生性能。',
        '只使用已完成配对条件对比成功率；没有调阈值、没有训练、没有自动扩大冻结计划。']
    report_path=output/'OVERNIGHT_REPORT.md';temporary=report_path.with_suffix('.tmp')
    temporary.write_text('\n'.join(report)+'\n');temporary.replace(report_path)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(json.dumps(analyse(args.root),ensure_ascii=False))
