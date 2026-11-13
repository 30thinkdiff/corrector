# Codex执行任务书：查明三层特征尺度，正式J-36M训练，新增J-LVM-only与完整J-LVM+OGG配对评测

**2026-11-12晚；状态：已明确研究协议并提交文档，**不代表这份任务书对应的代码、训练与闭环已经执行。由本地Ubuntu22.04上的Codex实施，本机只修改新分支，不要在AutoDL安装/启动Codex；远程RTX4090使用独立ZIP+SHA部署。基础模型设计见[FORMAL_MULTILAYER_CORRECTOR_J36M_DESIGN_20261112.md](FORMAL_MULTILAYER_CORRECTOR_J36M_DESIGN_20261112.md)，容量、历史条件和实验边界见[OVERNIGHT_RAW_SINGLE_JOINT_LIBEROPLUS_PLAN_20261112.md](OVERNIGHT_RAW_SINGLE_JOINT_LIBEROPLUS_PLAN_20261112.md)。

## 0. 先确认哪些是真实存在的东西

- 2026-11-12真实4090 MVP源版本`721f2bae4c3cf2c9ef7abaf548caf40ee9d9f92d`、ZIP SHA256 `a98b5be6319054a532af7d35e0939d8684fea48c21485d8a215b808996515565`，远程源路径`/root/autodl-tmp/corrector/dev/multilayer-training-mvp-a98b5be63190`。GitHub可见分支未发现这个源码提交，**先定位用户本地对应工作树/ZIP**，不可用旧pilot代码覆写真实MVP。已有A小Connector-only`273120`参，B小多层`767328`参，都是5轨迹5轮工程闭环，**不是J-36M正式研究模型**。
- 用户重启扩容后真实`df`：`/` 30G总/16G空闲；`/root/autodl-tmp` 130G总/89G空闲；4090 24GB空闲，`episode_813/COMPLETE.json`与Object10 manifest存在。这仅证实标记存在，不代表每个原始FP16的SHA在本次重启后已全部验算。
- 已有32×32双层原始FP16，Episode813–817（749帧，约4.39GiB，分别在两个系统盘根目录），旧Connector缓存含`[T,128,960]` int8+scale，旧缓存不可改；旧38M Corrector、Transformer权重与历史结果亦不可改。
- 正式目标：具有明确跨层交互的J-36M（设计参数约36.79M，真实实现必须按`sum(p.numel())`重新审计），输入`F2/F8 [B,2,1024,768]`、Connector`[B,128,960]`、已执行/候选当前动作`[B,7]`，预测所有三层`t→t+10`的变化。训练10Hz/k10=1s；仿真控制20Hz对应20环境步。原数据不得被隐式池化、INT8或替换同形状生成数据。

## 1. 必做：解释Connector zero-delta MSE为什么很大（真实测量，而不是推断）

来自用户的固定旧验证数据`episode815`、142个重叠窗口：

| 层 | zero raw MSE | delta RMS | zero normalized MSE |
|---|---:|---:|---:|
| Block2 | 0.038176628329 | 0.19539 | 0.98705127798 |
| Block8 | 0.242567362549 | 0.49251 | 0.99390108373 |
| Connector | 9.926980713506 | 3.15071 | 0.99652203032 |

Connector的逐元素MSE约为Block2的260×、Block8的40.9×；**不能解释为预测难度或Token总数差异**：MSE是逐元素平均，且原Connector和中层是不同数值空间，Connector经过后续Transformer、post-layernorm、pixel shuffle与投影；pixel shuffle仅重排值。旧Connector为INT8+scale反量化，新层为FP16，量化误差另行统计；Episode813已验证155帧重新量化完全一致。

实现`feature_scale_audit`，只读同一训练/验证Episode（至少已有813–817）：
1. 每层原值`F_t`的mean/std/RMS、绝对值分位数、按相机和Token/通道分组统计；变化`F_{t+10}-F_t`的RMS、p50/p90/p99和异常样本。
2. 在同一真实帧/模型权重下，对Block8→后续层→post-layernorm→pixel shuffle前后→Connector原浮点输出逐站点量测`F_t`、差量RMS，说明观察的幅度变化在哪些转换出现；不可对非对应的Token直接逐元素作差；不允许改冻结模型forward。
3. 原浮点Connector vs 旧反量化Connector的`Δ`比较，计算相对L2/MSE/余弦、按场景统计，验证帧/相机顺序及源权重SHA。
4. train-only RMS得到`s2,s8,sC`，分别在train和val产生original/normalized MSE与zero基线；记录每头对共享参数的梯度范数与夹角，避免MSE幅值高的一头统治训练，且不要在val上拟合scale。
5. 输出`feature_scale_audit.json`、小CSV与结论：区分**已测事实**和**可能机制**。保持现有特征/量化缓存不变。

## 2. 实现/审查J-36M训练（与OGG衔接）

先打印**真实MVP的现有模型拓扑**，再独立建立新J类，不要凭设计文档猜旧代码。将Block2/8原32×32局部4×4重组到与Connector的8×8**区域位置**一致，独立adapter、cross-layer fusion、action FiLM、共享Transformer、完整32×32中层输出头及高分辨率skip；见正式设计文档的精确维数和36.79M参数预算。严格核对两相机和pixel shuffle排列。Patch grouping不允许改变监督目标分辨率。

每层`ΔF_l=F_l(t+10)-F_l(t)`；候选`L_l=MSE(ΔFhat,ΔF)/(s_l²+eps)+β[1-cos(flat(ΔFhat),flat(ΔF))]`，初始β=1候选，`L_total=0.25L2+0.25L8+0.50LC`；另固定等权消融。loss配置**必须可序列化并冻结**，逐层报告原始/归一MSE、cosine、zero相对改善、梯度统计，保存best/latest/optimizer/RNG且能精确恢复。原C0单Connector-only 38.05M与原版S2/S8约37M是正式对照；小MVP A/B只保留为历史工程对照。不能直接跨层比较原始MSE。

GPU进阶门槛：CPU shape/梯度测试 → 已有5轨迹单batch forward/backward、FP32或显式AMP、峰值显存、吞吐、训练单batch过拟合、Checkpoint重载→仅在通过后按冻结manifest扩展70条多任务Episode。正式J真实参数量必须报告，不得为防OOM默默改成width96；先降microbatch、梯度累积、缓存加载方式，调整结构需要独立实验ID和用户许可。

**数据方案（当前130GB）**：Object10十个TaskID20–29各固定5 Train+2 Validation=70条，10265帧，train50Episode/6791配对窗口，val20Episode/2774配对窗口，两层完整FP16约60.15GiB，其中已提取的813–817共4.39GiB必须复用，新增约55.76GiB；清单所需23个Parquet分片总约2.06GiB。按manifest确定具体ID，选择规则/版本冻结，严禁通过验证集成绩选Episode。数据盘保留≥15–20GiB，系统盘≥8GiB；缓存逐Episode完成标志、SHA、STOP、resume、lazy memmap并保留所有旧特征和训练资产。要特别检查原脚本对Episode白名单的限制，不能只训练原5条。空间不足则按完整Episode边界暂停并汇报，不自动扩容、不删除无主张数据。

## 3. 新增LVM-only必须对照完整LVM+OGG（两组共用一个J best）

**当前冻结代码已有`monitor/lvm/ogg`三种语义：**见`scripts/corrector_online_runtime.py`。multiJ适配必须**复用状态机和动作队列语义**，新增多层视觉Hook/预测结果/分层打分/OGG梯度，不要再造两份容易不一致的代码。

### 3.1 训练、偏离检测、OGG三个不同量

- **训练**：监督预测未来真实`ΔF`，更新J网络权重。
- **LVM**：起点t时存`F_l(t),ΔFhat_old_l`，到`t+20`控制步时观察`ΔFobs_l`，计算分层`d_l=1-cos(ΔFhat_old_l,ΔFobs_l)`、可靠性掩码；权重与阈值仅在**独立正常运行校准集**拟合并冻结，融合`score=Σ w_l d_l`，接入原robust-threshold状态机。零范数/极低变化或非有限值必须退化/跳过并记录，不能把cos=0的返回当成确凿异常。
- **OGG**：触发时构造分层`e_l=ΔFhat_old_l-ΔFobs_l`，冻结J和当前视觉上下文，生成新候选动作`a(v_tau)`，用`L_OGG=Σ w_l[1-cos(ΔFhat_new_l(a),e_l)]`做输入梯度引导；在有效层重新归一权重，测试`dLoss/dv_tau`非零有限。**不能使用尚不存在的未来观测当作OGG目标**；训练`λ_l`和OGG`w_l`可不同，但在测试前冻结。
- 当前`[7]`单动作预测1秒变化仍有条件信息不足，未来多动作条件只作为独立消融，不得混在本轮多层收益归因中。

### 3.2 方法矩阵与软件验收

| 归档key | 分层监测 | LVM触发中止＋重新规划 | OGG引导 |
|---|---|---|---|
| H50 | 否 | 否，正常50前缀 | 否 |
| H20 | 否 | 否，正常20前缀 | 否 |
| j_monitor | 是 | **否** | 否 |
| **j_lvm** | 是 | **是** | **否** |
| **j_lvm_ogg** | 是 | **是** | **是** |

`j_lvm`和`j_lvm_ogg`**必须使用同一J best、同样多层LVM融合分数和阈值、相同策略权重/相机/语言/初始化/seed/monitor stride/cooldown/动作队列**；唯一预定处理差异是OGG开关。`j_monitor`不是`j_lvm`，只能记录偏离、不清队列。

CPU回归及真实GPU在线smoke：
1. 在冻结相同视觉/计划/到期预测的模拟事件下，两个j_*在轨迹分叉前的`score`/阈值/触发结果一致；
2. `j_monitor`不能改动作队列，`j_lvm`须触发时清空待执行计划、下次调用重新采样，但`guidance_calls==0`且无去噪梯度路径；
3. `j_lvm_ogg`相同中止条件下触发正确的引导次数，至少一次非零有限`∂L/∂v_tau`，冻结网络参数没有变化；
4. 触发后新计划条件到期预测必须更新，旧计划的待到期预测如果无效则清空或显式取消，避免下一次假报警；
5. 两路相机Block Hook、完整高分辨率输出、多层score与correction target的shape、bfloat16/FP16/FP32转换、时间对齐(10Hz k10=20Hz 20步)与分层可靠性单测全部通过；
6. **轨迹因OGG分叉后不可要求两方法后续score完全相同**，但仍须同一状态机逻辑；不能人为追求数值完全配对而覆盖真实结果。

### 3.3 历史LIBERO-Plus复测

必须读取`30thinkdiff/robot-lab/studies/policy_generalization/replanning_results.md`和`libero_plus_protocol.md`的真实条件，而不是混同普通Object10：

- 冻结`lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`、Plus`4976dc30028e805ff8094b55501d532c48fec182`、正式指令`pick the alphabet soup and place it in the basket`，20Hz、50动作块/10次去噪，同物理初始状态、seed1000/1001/1002；
- 相机default、view15（开发）、view5/view10（保留）。历史H50/H20成功数：default3/3→3/3、view15 1/3→3/3、view5 0/3→1/3、view10 1/3→2/3。历史基线仍需核对实际代码版本，不能将不同版本运行视为一模一样。
- 先default中测试LVM假阳性与安全性，view15开发、**冻结模型/阈值/OGG权重后**才做view5/10保留评价；禁止根据保留结果调参与反复挑模型。
- 正式最低配对：`H50,H20,j_lvm,j_lvm_ogg`×4相机×3种子=48回合，`j_monitor`加入则60回合。应优先有单独的监控安全smoke，不用它冒充完成60个回合。
- 各条件/seed独立逐回合CSV+JSON：success、env_steps、可观察失败阶段、重规划事件数、监测score/阈值/无效层、每层OGG损失/梯度、policy调用/monitor/引导时间、动作连续性、视频路径、预处理输入与首噪声审计；形成J-LVM与J-LVM+OGG的**配对救回/退化**矩阵，保留H20作为更多重规划的参照。禁止从3seed统计显著性或跨任务泛化。
- 模型闭环效果未验证，所有`LVM-only`与`OGG`结果均需实际GPU执行，不得从训练MSE外推。

## 4. 实施与交付（分阶段、STOP与安全）

推荐具体里程碑：
1. 找到真实MVP源码/来源、更新单元测试和代码版本审计；提交报告。
2. 只读尺度审计，解释9.93 / 0.038 / 0.243差异的实测来源；发布小报告。
3. J-36M模型类/梯度/形状CPU测试 + 5轨迹GPU smoke；保留原C0/S2/S8/旧MVP，不动权重。
4. 冻结70Episode清单并按需提取完整32×32 FP16 + 校验；训练/验证/Checkpoint/归档。
5. 联合多层monitor/LVM/OGG适配，CPU mock+GPU在线smoke，检查同一触发只开/关OGG。
6. 按固定协议评测LIBERO-Plus开发视角，再保留视角；生成对照报告、失败分析与可复现命令。

Codex必须输出：新分支提交SHA、修改文件清单、数据/版本/超参冻结协议、CPU单测真实通过数、GPU执行命令（**不可冒充已在4090运行**）、ZIP SHA、`preflight`/`status`/`stop`/`resume`脚本、磁盘余量/模型参数/显存与耗时记录；文件夹按实验ID独立，任何旧模型/checkpoint/cache/结果、原第三方环境保持原样。若不能在同一晚完成，保留完整已验证阶段供次日分析，不许跳过训练/网络验证关卡。
