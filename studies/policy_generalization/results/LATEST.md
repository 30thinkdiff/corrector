# 实验结果总入口

本地快照截止：**2026-11-14 00:14:34 北京时间**。本地文件不会自动追随服务器变化。

- [本次完整指标报告](20261114/expanded_snapshot_20261113T161434Z/RESULTS.md)：新70条数据训练对照、普通LIBERO、Plus首轮17方法、复测快照和逐类结果。
- [机器可读摘要](20261114/expanded_snapshot_20261113T161434Z/summary.json) · [新训练汇总](20261114/expanded_snapshot_20261113T161434Z/training_runs.csv) · [逐层验证指标](20261114/expanded_snapshot_20261113T161434Z/training_layers.csv)。

| 实验 | 本次状态/规模 | 阅读入口 |
|---|---|---|
| 历史五模型训练与闭环 | 15组训练、1200正式回合；已完成 | [历史总报告](20261111/RESULTS.md) |
| SmolVLA H50/H20/H10/H5/H2/H1 | 24回合小样本；已完成 | [Horizon Pilot](20261111/horizon_pilot/RESULTS.md) |
| 单动作/A10混合检测与引导 | 完整Pilot；已完成 | [Hybrid Pilot](20261111/hybrid_pilot/RESULTS.md) |
| 已执行动作被动监测 | 4回合；已完成，不干预 | [被动诊断](20261111/executed_consistency_pilot/RESULTS.md) |
| 已执行监测闭环对照 | 48回合；已完成 | [闭环Pilot](20261111/consistency_closed_loop/RESULTS.md) |
| 夜间扩展与预测精度诊断 | 核心600已完成，整体2486/3400按时限暂停 | [夜间报告](20261112/overnight_review/RESULTS.md) |
| Replay成败组合样本库 | 119/660按研究价值暂停；诊断样本非总体成功率 | [覆盖审计](20261112/casebank_research_v2/RESULTS.md) |
| J预测保留策略Pilot | 小样本、不同早期协议；已完成 | [保留预测Pilot](20261113/j36m_retention_preserve/RESULTS.md) |
| Object10正式六方法 | 旧240＋新J120；已完成，桥接另4次 | [六方法报告](20261113/j36m_expanded_object/RESULTS.md) |
| 70条数据新训练 | J/C0/J_equal/S2已完成，S8在24轮暂停 | [训练指标](20261114/expanded_snapshot_20261113T161434Z/training_layers.csv) |
| Plus环境/资产冒烟 | completed；17/17，错误0 | [分方法CSV](20261114/expanded_snapshot_20261113T161434Z/plus_plus_smoke/methods.csv) |
| Plus首轮（正式选定子集） | completed；544/544，错误0 | [分方法CSV](20261114/expanded_snapshot_20261113T161434Z/plus_plus/methods.csv) |
| Plus有限诊断复测 | completed；272/272，错误0 | [分方法CSV](20261114/expanded_snapshot_20261113T161434Z/plus_plus_replicates/methods.csv) |

## Plus首轮关键结果

| 方法 | 成功 | 成功率 | 覆盖率 |
|---|---:|---:|---:|
| H50 | 7/32 | 21.88% | 不适用 |
| H20 | 20/32 | 62.50% | 不适用 |
| MLP-LVM | 11/32 | 34.38% | 100.00% |
| MLP-OGG | 11/32 | 34.38% | 100.00% |
| J-LVM | 9/32 | 28.12% | 100.00% |
| J-OGG | 9/32 | 28.12% | 100.00% |

当前选定Plus子集中H20领先，J尚未显示终点收益；J覆盖率完整不等于任务成功率提高。两基础任务、多类扰动不代表官方全量Plus基准。复测按首轮差异选条件，单独报告，不并入首轮。

- [全部17方法关键指标](20261114/expanded_snapshot_20261113T161434Z/plus_plus/methods.csv)：步数、VLA调用、重规划、OGG调用、VLA/监测/回合计时、显存、覆盖率、空窗、梯度和更新量。
- [扰动类别](20261114/expanded_snapshot_20261113T161434Z/plus_plus/categories.csv) · [逐回合](20261114/expanded_snapshot_20261113T161434Z/plus_plus/cases.csv) · [严格配对差异](20261114/expanded_snapshot_20261113T161434Z/plus_plus/comparisons.csv)。
- [J逐层预测误差](20261114/expanded_snapshot_20261113T161434Z/plus_plus/prediction_layers.csv)：来自已记录RMS/cosine的推导，仅有效评分窗口，不是全量离线预测。

[最终复盘：配对得失、三种子复现与Replay候选](20261114/expanded_final_review_20261114_002/RESULTS.md)。

## 当前队列已完成：历史指标与状态核查

在**远程终端**运行只读健康检查，确认实际进程，而非只读旧running：

```bash
/root/autodl-tmp/corrector/bootstrap/bin/python /root/autodl-tmp/corrector/dev/replay-health-60bd07ea8d57.py
```

以下为已完成阶段的最终指标，未自动启动后续任务：

```bash
cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/execution.json
cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/analysis/overall.csv
tail -n 25 /root/corrector_plus_expanded_20261113_001/worker.log
```

首轮完整表：同一路径将`plus_replicates`换成`plus`。`analysis/trials.csv`逐回合、`analysis/summary.csv`逐条件；实际原始trial与monitor日志在`trials/`和`attempts/`。

在**本地Ubuntu终端**刷新本报告及CSV（需要可用SSH连接，密码只在终端输入；只读远程、不启动实验）：

```bash
python3 scripts/corrector_progress_report.py sync
```

本次不下载视频/权重/大张量，因此不会自动给Replay补齐所有Plus视频。按需另选少量病例导入。你之前打开的`multilayer_mvp.../remote_results_001`是早期5条数据/5轮MVP，不能当作本次70条正式训练结果。
