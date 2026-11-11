# Corrector：SmolVLA视觉偏离监测与恢复实验

本仓库对应4090服务器上的 **`/root/autodl-tmp/corrector`现有项目根目录**，不是另一个替代目录。原始本地学习总仓库仍为[robot-lab](https://github.com/30thinkdiff/robot-lab)。两者拥有独立Git历史与origin；本仓库保留远程实际源码布局，以便GitHub/ChatGPT阅读、修改与后续同步。

## 当前研究状态（2026-11-11）

已完成Object10的454条示范、66984帧特征准备；MLP、单帧/三帧Transformer、单帧/三帧＋10动作Transformer各3个训练seed，共15组。五组seed42正式闭环各240回合，共1200回合；forced_check另计。H50=44/60、H20=51/60；完整OGG依次55/60、55/60、56/60、50/60、51/60。A10监测评分覆盖约59%，单动作100%，最长空窗0.95秒；当前LVM与OGG逐场景成败相同。尚未验证新的覆盖修复或训练—部署分布对齐方法，不宣称完整论文等价复现或新算法提升。

- [研究报告](studies/policy_generalization/research_report.md)
- [唯一行动/历史工作日志](studies/policy_generalization/work_log.md)
- [本轮结果审计](studies/policy_generalization/results/20261111/RESULTS.md)
- [冻结版本与数据来源](studies/policy_generalization/results/20261111/provenance.json)

## 目录与版本规则

| 路径 | 用途 |
| --- | --- |
| `night-code-1e80242789/` | 九组单动作训练/评测的真实冻结代码；旧checkpoint必须使用对应工厂版本 |
| `action-code-9274415dd6/` | 六组动作序列训练/评测的真实冻结代码 |
| `guardian-code-2263c95a33/` | 已部署36h监督器原版 |
| `runtime/*.py`及早期`*-code*` | 保留真实历史部署源码，不当作最新统一入口 |
| 根目录`corrector_priority_*.py` | 从远程取回的实际排序/自动驾驶脚本 |
| `scripts/` | 本地同步的开发、审计、学习与恢复工具；不是原冻结运行目录 |
| `studies/policy_generalization/` | 研究报告、学习练习、可发布的结果CSV/JSON/图 |
| `provenance/` | 132份远程源码/元信息的原始SHA清单与发布路径映射 |
| `third_party/` | 依赖许可副本；依赖Git源码/环境本体不入库 |

冻结目录不能原地修改后冒充续训。训练协议记录了源文件SHA；新模型/监测改动请在`dev/`或新部署目录开发，使用新输出目录，保留旧版本复算能力。不要用`scripts/`的新父队列直接续跑旧冻结`sweep_plan.json`。

## 现有服务器上的Git接入

初始化脚本在**现有根目录**创建`.git`，连接`https://github.com/30thinkdiff/corrector.git`。它通过离线Git bundle接入已发布提交，已有工作文件一律保留，只补齐原来没有的说明/工具/报告。不会搬走或覆盖环境、数据、模型。若源码在导出后有新改动，会保留为工作区修改供审阅。

```bash
cd /root/autodl-tmp/corrector
git status
git log -5 --oneline
git remote -v
```

后续从ChatGPT/GitHub修改的内容，需要在服务器检查工作区后`git pull --ff-only`；本次初始化可以离线完成，后续pull/push仍需服务器网络与自己的GitHub认证。本仓库不保存令牌、SSH私钥或个人邮箱，提交使用GitHub noreply地址。

## 数据、环境和权重

它们仍位于原目录，例如`runtime/venv`、`runtime/sim-venv`、`runtime/hf`、`runtime/checkpoints`、`object10/features`以及实验输出。Git默认忽略这些内容；只有清单/协议及精选报告进入版本控制。原始视频/模型不通过GitHub分发。系统盘`/root/corrector_supervision_36h/checkpoint_archive`中的无损latest转存也必须随完整备份保留，不能仅复制数据盘链接。

固定基础策略为`lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`；Corrector参考`927a06da8f00d7df483459e9039219e076c846de`，LeRobot源码`7e241bd630a3719a56157a497ce5d08f244784f1`，LIBERO源码`8f1084e3132a39270c3a13ebe37270a43ece2a01`。参考源按固定commit获取，不把`runtime`中的完整依赖克隆目录嵌入Git。

现有Python环境：`runtime/venv/bin/python`用于预测器，`runtime/sim-venv/bin/python`用于LIBERO闭环，`bootstrap/bin/python`用于标准库调度。当前只进行版本接入，不自动安装环境、训练、下载或清理。

## 本地结果复算

先取回轻量结果包（不含权重或视频），再用CPU审计：

```bash
python3 scripts/corrector_results_bundle.py fetch --fetch-only --output outputs/corrector_audit_20261111
python3 scripts/corrector_results_analysis.py --bundle outputs/corrector_audit_20261111/results_bundle.zip --output outputs/corrector_audit_20261111/analysis
```

绘图需NumPy/Matplotlib；读取/导出本身主要使用标准库，不加载VLA。核心学习练习继续保留在`studies/policy_generalization/corrector/`，不要自动填完用户尚未完成的方法。
