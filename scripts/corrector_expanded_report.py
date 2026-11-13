"""Audit completed J Object10 + explicitly bridged historical baselines, CPU only."""
import argparse,collections,copy,hashlib,json,zipfile
from pathlib import Path
from corrector_j36m_object_report import analyze,coverage
from replay_retention import select_groups


def report(archive,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        read=lambda n:json.loads(z.read(n))
        bridge=read('bridge_receipt.json');execution=read('ordinary/execution.json')
        if bridge['status']!='passed' or execution['status']!='completed' or execution['errors']:raise ValueError('bridge/completion not verified')
        records=[];coverage_rows=[]
        for n in z.namelist():
            if not n.endswith('.json') or not n.startswith(('ordinary/trials/','historical/trials/')):continue
            r=copy.deepcopy(read(n));r['source_member']=n;c=r['case']
            c['method']={'lvm':'mlp_lvm','ogg':'mlp_ogg'}.get(c['method'],c['method']);records.append(r)
            if n.startswith('ordinary/'):
                base=f"ordinary/attempts/{c['id']}/attempt_{r['attempt']:03d}/"
                rows=[json.loads(line) for line in z.read(base+'monitor_steps.jsonl').decode().splitlines()]
                coverage_rows.append({'case':c,**coverage(rows,r['result']['env_steps'],24)})
        totals=analyze(records);groups=collections.defaultdict(dict)
        for r in records:
            if r['status']=='completed':groups[r['case']['condition'],r['case']['seed']][r['case']['method']]=r['result']['success']
        methods=['H50','H20','mlp_lvm','mlp_ogg','j_lvm','j_ogg']
        if len(groups)!=60 or any(set(v)!=set(methods) for v in groups.values()):raise ValueError('strict six-method pairing incomplete')
        counts=collections.Counter(''.join(str(int(v[m])) for m in ['H50','H20','mlp_ogg','j_ogg']) for v in groups.values())
        flips=[{'condition':c,'seed':s,'outcomes':v} for (c,s),v in sorted(groups.items()) if v['mlp_ogg']!=v['j_ogg']]
        selection=select_groups(records,methods,max_groups=18)
        summary={'source_bundle_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'new_execution':execution,'historical_bridge_passed':True,'methods':totals['methods'],
            'paired_physical_states':len({c for c,s in groups}),'paired_condition_seeds':len(groups),
            'four_way_pattern_order':['H50','H20','mlp_ogg','j_ogg'],'observed_patterns':dict(counts),
            'mlp_ogg_vs_j_ogg_flips':flips,'coverage':coverage_rows,'selection':selection,
            'limits':['historical MLP trained454 episodes; J trained70: not architecture-only attribution',
                      'historical baseline reuse justified by four bridge rollouts, not every trajectory re-executed',
                      'diagnostic selected subset is not an unbiased success-rate sample',
                      'overall equality does not imply identical trajectories or absence of per-case gains/losses']}
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        rows=['# Object10六方法正式配对 · 2026-11-13','',
              '普通LIBERO新J补测120/120，错误0；复用旧四方法240回合。4个桥接回合已核对结局、步数、调用/事件数及保留的初始输入、噪声、状态与视频。',
              '', '| 方法 | 成功 | 成功率 | 平均步数 | 平均重规划 |', '|---|---:|---:|---:|---:|']
        for m in methods:
            g=totals['methods'][m];rows.append(f"| {m} | {g['successes']}/{g['completed']} | {g['success_rate']:.2%} | {g.get('mean_env_steps',0):.2f} | {g.get('mean_event_replans',0):.2f} |")
        rows+=['','J-LVM与J-OGG本轮结局逐案例一致。J-OGG相对历史MLP-OGG有2例独自成功、2例独自失败，总成功数相同；不能宣布整体提升。',
               'J两方法所有可评分步均有评分，额外最长空窗0；24步预热不计入覆盖率。','',
               '四位组合依次H50/H20/MLP-OGG/J-OGG：`'+json.dumps(dict(counts))+'`。优先复盘以下翻转：','']
        rows += [f"- `{r['condition']}` seed{r['seed']}：MLP-OGG={r['outcomes']['mlp_ogg']}，J-OGG={r['outcomes']['j_ogg']}。" for r in flips]
        rows += ['','完整120段J视频/动作/Trace已派生导入Replay；未伪造旧基线缺失视频。派生导入明确控制返回标志来自完整回合的逻辑推断，非独立逐步计时。',
                 'J三层目标与OGG梯度并非v1全量留存；新Plus采集器提供实际梯度/更新摘要及限量数值数组。',
                 '', '历史MLP训练454条、J训练70条，数据量不同；旧BF16 Pilot/固定正常阈值协议不混入本表。未知异常真值，不能把高分称为真实误报或漏检。']
        (output/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    s=report(a.archive,a.output);print(json.dumps({'new_completed':s['new_execution']['completed'],'paired_groups':s['paired_condition_seeds'],'patterns':s['observed_patterns']}))
