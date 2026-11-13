# J36M普通LIBERO闭环接入（2026-11-13）

用户要求先正常LIBERO、后Plus。本入口是冻结训练核心之外的独立适配，不改旧计划和Checkpoint；当前仅启动工程检查、正常校准及24回合自然Pilot，不自动启动360回合。

## 实际进度

16:40远程检查：70条完整原始特征提取、校验及train-only尺度统计完成；J_s42、C0_s42、J_equal_s42、S2_s42各60轮完成。S8_s42尚在训练。训练无重试/错误，系统盘剩余15.06GiB、数据盘26.76GiB。随后通过主队列STOP请求完整epoch暂停，后续队列暂停、看护器退出；S8的latest与原计划保留。确切暂停轮数见新结果目录的remote_snapshot.json。

单种子离线验证（同20条验证Episode/2774窗口）如下。这些不是机器人成功率。

| 模型 | best epoch | Connector MSE | Block2 MSE | Block8 MSE |
|---|---:|---:|---:|---:|
| J |32|5.211879|0.0237223|0.1355102|
| C0 |12|5.149472|—|—|
| J_equal |30|5.247951|0.0235451|0.1346579|
| S2 |40|—|0.0190927|—|

不能将各层原始MSE相加比较，也不能据此宣称J有闭环收益。S2当前单层指标优于J的对应头，跨层共享是否值得需结合检测/恢复与开销验证。

## 方法与阶段

- H50/H20：冻结SmolVLA每次生成原50动作块，执行50/20前缀；reset后检查实际队列容量与清空。
- c0_lvm/c0_ogg：同70条数据训练的原版38.05M Connector-only；当前动作条件，不是历史454条训练权重。
- j_lvm/j_lvm_ogg：同一个正式36.79M J best，多层观测/分数/阈值/触发相同，差别仅事件后OGG开关。
- 全部20Hz控制，10Hz训练k10对应20控制步；使用套件原语言，通过文字匹配记录数据集Task20–29到Object套件Task0–9的真实映射，禁止默认二者编号一致。

顺序：

1. 被动检查3回合：task0/init6/seed9100、最多60步；H50/C0 monitor/J monitor。逐条提交动作必须完全一致。截短工程回合不计性能。
2. 独立正常校准40回合：10任务×预定(init7,seed9000)/(init8,seed9001)×2个monitor。失败回合保留，不用其拟合正常阈值；不追加搜种子。至少2条正常成功轨迹、足够分层有效分数，否则停止。
3. 强制接口4回合：task0/init6/seed9101、最多60步，第20步强制事件。比较各模型LVM/OGG分叉前分数和阈值；LVM引导调用为0，OGG必须至少有一次有效调用。结果不计成功率。
4. 自然Pilot24回合：task1、9 × init0、1 × seed1000 ×六方法，最多280步。完成后停止并分析，不自动扩大。

校准采用显式ordinary_libero_object域，复用既有正常MAD估计公式及robust状态机，不将普通任务伪装成Plus/default；J/C0分别校准，同模型LVM/OGG共享同一份冻结校准。假阳性、覆盖与低范数跳过均保留，不把监测分数自动当作失败真值。

## 部署和操作

独立包：robot-lab的`outputs/autodl_setup/j36m_object_20261113/object_addon_002.zip`。

SHA256：`9c029654a32eb0eab3a20526f3a9d3c5566220c95208d17ec41817ffb9ab092f`。

实际核心：`/root/autodl-tmp/corrector/dev/j36m-35ecdd7488bf`，manifest SHA `a742e42b77fa2d05592c6b7b639881c80ed28515aa57524bf4eab7b6d499d0cc`。新入口逐文件校验它，并检查J/C0保存的source/timebase/config；不重写训练来源协议。

```bash
# 状态、分阶段完成数、日志
cat /root/corrector_j36m_object_20261113_002/status.json
cat /root/corrector_j36m_object_20261113_002/pilot/execution.json
tail -n 30 /root/corrector_j36m_object_20261113_002.launch.log

# 完整回合边界安全暂停
touch /root/corrector_j36m_object_20261113_002/STOP

# 恢复前确认旧PID已退出，处理错误并仅移除自己的STOP；同代码/同输出
nohup /root/autodl-tmp/corrector/runtime/sim-venv/bin/python -u \
 /root/autodl-tmp/corrector/dev/object-j36m-9c029654a32e/scripts/corrector_j36m_object_eval.py run \
 --output /root/corrector_j36m_object_20261113_002 --max-hours 4 \
 >> /root/corrector_j36m_object_20261113_002.launch.log 2>&1 < /dev/null &
```

每次显式恢复授予最多4小时，逐回合恢复，程序错误停止。此规则与旧训练32小时固定截止不同。系统盘下限8GiB、数据盘5GiB，每新回合检查；不删除旧数据、不自动安装/下载。输出放系统盘独立目录，不复制大特征缓存。

保存真实视频、归一化及提交控制器动作、首输入/状态/噪声、实际动作块调用/耗时、LVM trace、J分层OGG损失/梯度/动作更新及目标。录像帧数与提交步数检查，尚不声称单凭FPS证明录制相位。

## 测试与边界

本地原40项与新增5项通过（首版全套44项，修复后新增专项5项）；新增5项在实际远程现有Python环境通过。包含完整J形状/动作梯度/真实状态机旧回归和新校准隔离、计划配对、C0接口测试。CPU构造数据不是真实机器人结果。远程实际在线进度、错误及后续修复记录以结果目录为准。

Plus环境和正式大规模Object10仍是后续阶段，不冒充已接入当前队列。所有额外训练种子/JC保持原计划可恢复，但当前优先验证已训练权重的闭环价值。

首版001在真实H50回合后的结果解析遇到KeyError(per_task)。只读核对现有LeRobot返回per_episode，新增严格单回合bool解析测试；旧001全部保留，修复在新源码SHA及002目录部署。该程序错误不计机器人失败。

17:00前后实测：V2被动3/3及校准40/40、强制检查4/4完成，错误0。各模型预定20条被动正常候选中14条成功，对应1555个有效正常分数；失败原记录保留。J融合权重Block2/Block8/Connector=0.417925/0.288372/0.293703，C0权重Connector=1；同模型两干预组共享校准。强制检查每组恰好1次事件，LVM组0次引导、OGG组各4次引导，检测分叉前一致。自然Pilot随后启动，结论以最终结果报告为准。
