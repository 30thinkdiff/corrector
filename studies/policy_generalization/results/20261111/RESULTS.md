# Object10训练与闭环结果审计

证据包SHA256：`425a22e6d2ebbdf9c611eef8d78b9e58ae70da7de7c2957b9858b19662293c8e`。已读取15组训练，其中15组有完成与重载证据；正式评测5组、1200回合。

## 核验状态

审计错误0项、警告4项。统计按原始逐回合记录重算，强制调用检查不计正式成功率。

## 预测器开发验证：多种子

| 配置 | 种子数 | MSE均值 ± 样本标准差 | cosine均值 ± 样本标准差 | 每次完整epoch计算累计均值/分钟 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 3 | 4.0316 ± 0.0149 | 0.8020 ± 0.0012 | 44.18 |
| transformer-a10-h1 | 3 | 3.6315 ± 0.0134 | 0.8224 ± 0.0013 | 47.02 |
| transformer-a10-h3 | 3 | 3.6627 ± 0.0197 | 0.8214 ± 0.0015 | 97.26 |
| transformer-h1 | 3 | 3.8817 ± 0.0186 | 0.8112 ± 0.0016 | 251.07 |
| transformer-h3 | 3 | 3.9051 ± 0.0263 | 0.8100 ± 0.0010 | 266.34 |

这里的验证集参与best选优；多动作组离线条件包含后续实际执行动作，不能将更低MSE直接归因于架构，也不能据此承诺闭环改善。

## 正式闭环与计算成本

| 模型组 | 方法 | 成功/完成 | 成功率 | 动作块调用均值 | 策略秒/回合 | 监测秒/回合 | 墙钟秒/回合 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| night/mlp_s42 | H50 | 44/60 | 73.33% | 3.83 | 0.843 | 0.000 | 8.669 |
| night/mlp_s42 | lvm | 55/60 | 91.67% | 4.63 | 1.120 | 3.330 | 11.620 |
| night/mlp_s42 | H20 | 51/60 | 85.00% | 8.27 | 1.808 | 0.000 | 8.946 |
| night/mlp_s42 | ogg | 55/60 | 91.67% | 4.60 | 1.172 | 3.334 | 11.729 |
| night/transformer-h1_s42 | H50 | 44/60 | 73.33% | 3.83 | 0.848 | 0.000 | 8.634 |
| night/transformer-h1_s42 | lvm | 55/60 | 91.67% | 4.82 | 1.166 | 3.591 | 11.823 |
| night/transformer-h1_s42 | H20 | 51/60 | 85.00% | 8.27 | 1.802 | 0.000 | 8.850 |
| night/transformer-h1_s42 | ogg | 55/60 | 91.67% | 4.87 | 1.332 | 3.608 | 12.075 |
| night/transformer-h3_s42 | H50 | 44/60 | 73.33% | 3.83 | 0.829 | 0.000 | 8.706 |
| night/transformer-h3_s42 | ogg | 56/60 | 93.33% | 5.30 | 1.459 | 3.561 | 12.257 |
| night/transformer-h3_s42 | H20 | 51/60 | 85.00% | 8.27 | 1.773 | 0.000 | 8.943 |
| night/transformer-h3_s42 | lvm | 56/60 | 93.33% | 5.18 | 1.237 | 3.546 | 11.918 |
| actions/transformer-a10-h1_s42 | H50 | 44/60 | 73.33% | 3.83 | 0.907 | 0.000 | 8.714 |
| actions/transformer-a10-h1_s42 | ogg | 50/60 | 83.33% | 4.08 | 1.170 | 3.773 | 12.328 |
| actions/transformer-a10-h1_s42 | lvm | 50/60 | 83.33% | 4.08 | 1.065 | 3.772 | 12.168 |
| actions/transformer-a10-h1_s42 | H20 | 51/60 | 85.00% | 8.27 | 1.927 | 0.000 | 8.973 |
| actions/transformer-a10-h3_s42 | H50 | 44/60 | 73.33% | 3.83 | 0.810 | 0.000 | 8.243 |
| actions/transformer-a10-h3_s42 | lvm | 51/60 | 85.00% | 4.07 | 0.950 | 3.603 | 11.445 |
| actions/transformer-a10-h3_s42 | H20 | 51/60 | 85.00% | 8.27 | 1.727 | 0.000 | 8.458 |
| actions/transformer-a10-h3_s42 | ogg | 51/60 | 85.00% | 4.03 | 1.033 | 3.603 | 11.556 |

## 在线监测覆盖率

| 模型组 | 方法 | 有效打分/预热后应监测步数 | 覆盖率 | 最长无评分间隔/秒 |
| --- | --- | ---: | ---: | ---: |
| night/mlp_s42 | lvm | 7585/7585 | 100.00% | 0.00 |
| night/mlp_s42 | ogg | 7578/7578 | 100.00% | 0.00 |
| night/transformer-h1_s42 | lvm | 7674/7674 | 100.00% | 0.00 |
| night/transformer-h1_s42 | ogg | 7713/7713 | 100.00% | 0.00 |
| night/transformer-h3_s42 | lvm | 7554/7554 | 100.00% | 0.00 |
| night/transformer-h3_s42 | ogg | 7582/7582 | 100.00% | 0.00 |
| actions/transformer-a10-h1_s42 | lvm | 4862/8260 | 58.86% | 0.95 |
| actions/transformer-a10-h1_s42 | ogg | 4873/8256 | 59.02% | 0.95 |
| actions/transformer-a10-h3_s42 | lvm | 4766/8131 | 58.62% | 0.95 |
| actions/transformer-a10-h3_s42 | ogg | 4787/8131 | 58.87% | 0.95 |

覆盖率分母排除初始预热和首次预测到期前的步数，按实际monitor stride统计。skips/cancellations不是缺失评分时刻的一一分割，不能直接相加替代缺失计数。
完整10动作条件与计划失效撤销形成监测空窗；这支持条件可用性是重要机制因素，但没有等覆盖率对照时不能把全部成功率差异归因于它。


## 配对救回与损失

| 模型组 | 对照 → 完整OGG | 配对数 | OGG独有成功 | 对照独有成功 | 净百分点 |
| --- | --- | ---: | ---: | ---: | ---: |
| night/mlp_s42 | H50 → OGG | 60 | 11 | 0 | 18.33 |
| night/mlp_s42 | H20 → OGG | 60 | 5 | 1 | 6.67 |
| night/mlp_s42 | lvm → OGG | 60 | 0 | 0 | 0.00 |
| night/transformer-h1_s42 | H50 → OGG | 60 | 11 | 0 | 18.33 |
| night/transformer-h1_s42 | H20 → OGG | 60 | 5 | 1 | 6.67 |
| night/transformer-h1_s42 | lvm → OGG | 60 | 0 | 0 | 0.00 |
| night/transformer-h3_s42 | H50 → OGG | 60 | 12 | 0 | 20.00 |
| night/transformer-h3_s42 | H20 → OGG | 60 | 6 | 1 | 8.33 |
| night/transformer-h3_s42 | lvm → OGG | 60 | 0 | 0 | 0.00 |
| actions/transformer-a10-h1_s42 | H50 → OGG | 60 | 6 | 0 | 10.00 |
| actions/transformer-a10-h1_s42 | H20 → OGG | 60 | 3 | 4 | -1.67 |
| actions/transformer-a10-h1_s42 | lvm → OGG | 60 | 0 | 0 | 0.00 |
| actions/transformer-a10-h3_s42 | H50 → OGG | 60 | 7 | 0 | 11.67 |
| actions/transformer-a10-h3_s42 | H20 → OGG | 60 | 3 | 3 | 0.00 |
| actions/transformer-a10-h3_s42 | lvm → OGG | 60 | 0 | 0 | 0.00 |

McNemar数值保存于JSON/CSV，为探索性未校正统计；重复seed和初始化存在相关性。跨模型重复的H50/H20不合并为更大独立分母。

## 解释边界与下一步

1. 先核对audit_findings.json中的错误及基线逐回合不一致，再判断模型差异。
2. 用相同case的LVM与OGG配对得失，区分检测/中止收益与引导增量。
3. 按review_candidates.json取回少量关键失败视频，核对误触发、抓空、掉落与恢复；本包没有视频内容。
4. 同步监测耗时、策略调用耗时和真实墙钟时间分别报告，不等同论文异步暴露等待。
5. 若离线预测改善而闭环无改善，优先检查动作分布差异、LVM阈值校准及引导方向；不立即扩大下载/重训。

## 审计发现

- [warning] monitor_coverage_loss: actions/transformer-a10-h1_s42 lvm 58.86% after warmup
- [warning] monitor_coverage_loss: actions/transformer-a10-h1_s42 ogg 59.02% after warmup
- [warning] monitor_coverage_loss: actions/transformer-a10-h3_s42 lvm 58.62% after warmup
- [warning] monitor_coverage_loss: actions/transformer-a10-h3_s42 ogg 58.87% after warmup
