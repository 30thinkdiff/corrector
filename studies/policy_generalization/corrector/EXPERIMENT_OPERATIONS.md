# 实验结果与操作入口

更新：2026-11-14。本次普通J补测、Plus首轮和有限复测均已正常完成；旧训练保持暂停。下面的查看命令只读；恢复命令仅针对以后已确认的中断，当前不需要恢复已完成队列。

## 本地阅读与刷新

当前结论：[研究报告](../research_report.md)；状态与后续：[工作日志](../work_log.md)；所有阶段：[结果入口](../results/LATEST.md)；完整收尾：[最终复盘](../results/20261114/expanded_final_review_20261114_002/RESULTS.md)。

在仓库根目录执行：

```bash
python3 scripts/corrector_progress_report.py sync --ssh-host autodl
```

`autodl`应是本机私有SSH配置中的别名，主机、端口和认证只保存在本机，不写入Git。也可传入另一个已配置别名。已有共享连接可继续使用；失效时只在终端完成交互认证，不把密码发到聊天或写入脚本。

脚本只读取元数据和已完成回合的轻量Trace，不运行仿真、不下载模型/训练数据/视频，也不改远程结果。每次保存新的原始ZIP和版本化报告，更新LATEST导航；失败保留partial，不发布半成品报告。已有原始包可在本地重算：

```bash
python3 scripts/corrector_progress_report.py report --archive <本机已保存的元数据ZIP>
```

## 远程真实状态与指标

在远程终端运行：

```bash
/root/autodl-tmp/corrector/bootstrap/bin/python /root/autodl-tmp/corrector/dev/replay-health-60bd07ea8d57.py
cat /root/corrector_plus_expanded_20261113_001/results/plus/execution.json
cat /root/corrector_plus_expanded_20261113_001/results/plus/analysis/overall.csv
cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/execution.json
cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/analysis/overall.csv
```

健康检查匹配实际PID、脚本和输出目录，避免关机遗留running或PID复用。正常完成看completed=planned、errors=0和调度器退出码；进程退出本身不代表失败。逐回合在`analysis/trials.csv`，逐方法/条件在`analysis/summary.csv`，原trial和monitor分别在`trials/`和`attempts/`。运行时汇总在完整回合返回后更新。

## 暂停与恢复边界

若以后确需安全暂停该队列：

```bash
touch /root/corrector_plus_expanded_20261113_001/STOP
touch /root/corrector_plus_expanded_20261113_001/results/STOP
```

等待当前完整回合返回并核验paused及GPU进程；不强杀。恢复前先确认原因、源包/计划/权重SHA、已完成视频、目录及资源，保存旧状态，不修改冻结参数。原队列以同版本继续并跳过完整结果，中断回合开新attempt；当前队列已经完成，不应为继续占用GPU再次启动。

## 保存与发布

所有回合保留指标、协议、轻量Trace、真实控制动作、队列与视频；完整OGG数组只保留每回合前2次、J目标首事件，其余有摘要。活动队列已发布的视频SHA参与恢复验证，不能静默删视频。原始资产、训练缓存或权重的清理另行审计，不由同步脚本执行。

公开仓库只放源码、配置、测试和派生报告；原始ZIP、视频、权重、环境、SSH连接和本地Replay状态均不提交。实际运行版本的逐文件SHA可从corrector的`archive/autodl-deployments-20261114`分支核对；当前算法主线在`feature/j36m-lvm-ogg-20261112`，工作台在robot-lab的`feature/robotlab-replay`。
