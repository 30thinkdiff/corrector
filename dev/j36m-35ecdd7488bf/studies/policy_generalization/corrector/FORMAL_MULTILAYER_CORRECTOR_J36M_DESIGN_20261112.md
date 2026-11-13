# 纠偏设计：正式多层Corrector候选J-36M，不再把0.767M工程MVP当成正式模型

日期：2026-11-12。本文**仅是可检验的架构规范与实验计划，不是已经实现/训练的模型**。用户最初希望获得一个适合有限4090资源、对Block 2/8/Connector进行有意义融合、可用于LVM和OGG的正式Corrector架构。此前交给Codex的轻量任务书只要求MVP，结果使用width96/depth2，产生767,328参数。这个任务交付不足来自架构要求未明确，不应归咎于Codex；已跑通的MVP只证明工程闭环，不能代表正式多层Corrector算法效果。

## 1. 固定输入输出与时间

- 冻结同一`lerobot/smolvla_libero`权重，视觉层从零索引选Block 2和8，双相机顺序固定；
- `F2,F8: [B,2,1024,768]`，来自每路32×32特征；
- `Zc: [B,128,960]`，每路8×8，原pixel shuffle factor4 + connector投影；
- 当前动作`a_t:[B,7]`，冻结processor归一化；不输入`F_{t+10}`或未部署可得的未来实际动作；
- 预测`delta_F2,delta_F8:[B,2,1024,768]`与`delta_Zc:[B,128,960]`，数据10FPS、k=10，1秒跨度；在线20Hz对齐20控制步，不能误作10步；
- 原始中层FP16全分辨率文件**保持不变**，模型内部可进行有记录的下采样/patch grouping，输出和监督保持全32×32。

## 2. 候选J-36M架构（一个模型、真正跨层融合）

1. 两路相机各自将Block2/8 32×32 reshape成8×8的4×4**局部Patch组**。每组16×768=12,288维，共128个组（两相机各64）。Patch顺序必须与Connector pixel shuffle factor4空间布局核对；空间网格对齐不等于特征数值相等。
2. 各层独立Adapter，`Linear(12288,512)`×2，对`Zc`使用`Linear(960,512)`。三个路径输出都为`[B,128,512]`。对各层输入独立归一化，使用**只基于训练集**估计的固定RMS或LayerNorm（LayerNorm如计入参数需重算总量），禁止看validation统计。
3. 逐Token拼接3个Adapters得到`[B,128,1536]`，`Linear(1536,768)`得到`[B,128,768]`；加入8×8位置编码和camera embedding，**两个相机空间位置不能简单假定为同一坐标**。该融合层确实接受三个层的同一区域信息。
4. 动作`Linear(7,256)→SiLU→Linear(256,1536)`生成对主干的FiLM gamma/beta，保证OGG对7维动作可求梯度；动作的256维embedding同时供local skip使用。
5. 共享主干：3×`TransformerEncoderLayer(d_model=768,nhead=8,dim_feedforward=1536,dropout=0,batch_first=True,norm_first=True)`，上下文Token数128，避免把4096个中间层Token直接做全局attention。交互可跨空间区域、跨相机，但需记录注意力的成本与可能的伪对齐风险。
6. Connector Head：`Linear(768,960)`，输出`[B,128,960]`。
7. Block2/8 Head：各自`Linear(768,256)→SiLU→Linear(256,12288)`，按原4×4局部Patch布局unpatchify回`[B,2,1024,768]`。为了保留高频信息，各层额外有本地跳跃预测路径`Concat(F_l token[768],action_embed[256])→Linear(1024,128)→SiLU→Linear(128,768)`，局部输出与共享主干的对应局部Patch预测**相加**作为最终`ΔF_l`。共享主干输出均依赖于2/8/Connector，这才构成跨层融合。
8. 禁止在整个模型/损失中对`F2`/`F8`偷偷从32×32降为8×8目标；如果为了对照显式采用8×8监督，另写独立实验ID。

### 参数量：按上述torch.nn模块**仅初始化并数参**的本地计算（未做GPU训练）

- Block2 Adapter: 6,291,968；Block8 Adapter: 6,291,968；
- Connector Adapter: 492,032；
- Fusion linear: 1,180,416；
- Action FiLM MLP: 396,800；
- Shared 3个TransformerEncoderLayer: 14,181,120；
- Block2 decoder: 3,354,880；Block8 decoder: 3,354,880；
- Connector head: 738,240；
- Block2 local skip: 230,272；Block8 local skip: 230,272；
- Camera embed(2×768)+pos embed(64×768): 50,688；
- **合计36,793,536参数（约36.79M），对比原版单Connector38,052,800，属于同一量级；不是要追求恰好等参。**

上面的数值对应一个具体torch.Module配置，但尚未执行完整forward、GPU显存/性能、LVM与OGG验证；精确实现若增加norm/gate需要重新数参。对于两种中层Block原上游MLP只是`token_dim=768`、`scale=M20`，该版结构参数量应从模型初始化精确输出，不能将小模型替代并称为论文版。

## 3. 训练损失与公平对照

- 主评估任务Connector ΔZ，三层辅助目标分开报告。以训练集RMS归一MSE和可选cosine辅助，`L=λ2 L2 +λ8 L8+λc Lc`；初始候选`λc=0.5,λ2=0.25,λ8=0.25`与等权对照（不能事后只挑好看的）；每种层的真值目标形状完全匹配预测。
- C0: 原38.05M Connector-only MLP；S2: 上游原架构Block2-only MLP；S8: Block8-only；J-36M: 本架构全输入三层监督；
- JC: 同样J架构/输入但仅Connector监督，区分额外输入与辅助预测竞争；
- J-shuffle/J-no-fusion（可选）：打乱其他层输入或移除跨层融合，同时匹配容量/预算，检测是否真正使用额外信息；不得在validation上调参。
- 以固定Episode级train/validation、同seed、固定更新预算和同评估协议对照；不仅记录参数量，还要记录实际GPU时间和吞吐。S2/S8与Connector输出空间不同，原始MSE不可直接比较；比各自zero baseline及跨组在线表现。
- 保留旧0.767M模型作为`J-small-MVP`工程/轻量化对照，不作为正式架构对手。

## 3A. Connector原始MSE显著较大：现有证据、禁止误读及必做诊断

**这是数值尺度问题的待验证机制，不可直接解释为Connector较难预测、权重更重要、信息量更高或层间训练不公平已经被消除。** 在2026-11-12的5条Episode MVP（验证Episode815，142个高度重叠窗口），原始zero-delta MSE分别为：

| 预测目标 | zero-delta逐元素原始MSE | 对应变化RMS（sqrt MSE） | zero-delta归一MSE（已有报告） |
|---|---:|---:|---:|
| Block2 | 0.0381766283 | ≈0.19539 | 0.9870513 |
| Block8 | 0.2425673625 | ≈0.49251 | 0.9939011 |
| Connector | 9.9269807135 | ≈3.15071 | 0.9965220 |

Connector的zero-MSE为Block2约260×、Block8约40.9×，但对应**每元素RMS变化**比率只有约16.1×、6.4×；MSE平方放大尺度差异。**不能归因于2048 vs 128个Token的总元素数量**：报告使用逐元素平均MSE，而不是全Tensor误差平方和。Block2/8是视觉Transformer内部Block输出，Connector还经过后续Block、post-layernorm、pixel shuffle和学习投影；三者处于不同数值空间。**pixel shuffle只是重排，本身不放大数值**。老Connector数据`z_q`+per-token`z_scale`反量化，Block2/8为原始FP16；量化也需核对，但已验证Episode813原Connector重新量化155/155完全相同，不得把9.93直接归因于量化。

**新增强制只读统计诊断（可与小规模GPU smoke复用，不阻塞模型类CPU实现）：**

1. 对同一train/val Episode、相同锚点和两相机，分别统计`F_l(t)`的逐元素mean/std/RMS、每Token/per-camera分位数与最大绝对值；`F_l(t+10)-F_l(t)`的RMS、p50/p90/p99、变化与原值RMS比；记录train/val分布，不在val上拟合归一化scale。
2. 沿**同一真实帧**额外采样视觉Block8后续输出、post-layernorm、pixel shuffle前/后的张量、Connector float输出；对应每个站点的增量RMS。查清是后续层/归一化/投影哪个阶段导致主要数值变化；避免把Tensor形状更改解释成幅值放大。不得破坏原模型前向或缓存。
3. 同帧原生Connector浮点与已有INT8+scale反量化、以及它们的`t→t+10`差量做MSE/RMS/cosine误差分析；区分量化噪声与表示本身变化。
4. 按任务/episode层级报告并核查shape、camera顺序、k10、帧/动作时标、实际模型revision，检查极端值驱动均值的可能性；写`feature_scale_audit.json`和小型CSV，不修改已冻结文件。
5. 对三层**各自**以训练split的`s_l²=mean_train[(ΔF_l)^2]`计算Normalized MSE。验证时使用同一训练`s_l`；报告真实原始MSE、normalized MSE、cosine、zero-delta基线。normalized MSE有助于损失尺度公平但不保证梯度范数公平；另记录每层对共享主干的梯度范数和cosine冲突。

## 3B. 训练目标、LVM偏离评分、OGG引导损失必须分开

三者都是作用于J模型的三路变化输出，但**目标、可用信息、是否反向传播均不同**：

- **训练**：同一示范的真实未来差量`ΔF_l=F_l(t+10)-F_l(t)`作为监督；`L_train=Σ λ_l[ MSE(ΔFhat_l,ΔF_l)/(s_l²+eps) + β(1-cos(ΔFhat_l,ΔF_l)) ]`。候选`λ2=0.25, λ8=0.25, λc=0.5`，另有等权消融；β默认候选1须固定并记录，不能当成已最佳。预训练与验证不能使用真实未来观测作为**输入**。
- **LVM（预测到期后）**：t时在实际执行动作条件下存`ΔFhat_l^old`和`F_l(t)`；t+k时**真实观测已经发生**，计算`ΔFobs_l=F_l(t+k)-F_l(t)`，对每层`d_l=1-cos(flat(ΔFhat_old_l),flat(ΔFobs_l))`。先从独立正常执行轨迹校准有效向量范数门槛、分层可靠性和冻结权重`w_l`，例如`score=Σ w_l d_l`并建立冻结阈值/滞回/冷却；任何小范数向量方向不可靠，必须记录、跳过或退化，而不是强制解释为异常。分层误差、融合分数、触发阈值与实际触发步骤需分别存日志。**训练集/保留视角不可用于在线调阈值；view5/view10不得泄漏用于调参**。
- **OGG（LVM触发后，生成替代动作期间）**：将到期误差`e_l=ΔFhat_old_l-ΔFobs_l`固定为候选恢复方向。冻结J权重与当前真实视觉上下文，对新的候选动作`a(v_tau)`得到`ΔFhat_new_l(a)`，使用`L_OGG=Σ w_l[1-cos(flat(ΔFhat_new_l(a)),flat(e_l))]`（仅对有效且可靠的层计算并重新归一有效权重）。此时真实未来尚未发生，**不得用未来真实帧指导OGG**。梯度经`v_tau→a→J→L_OGG`反传给Flow Matching速度变量，不更新Corrector和SmolVLA参数；先记录每层梯度方向/范数、冲突及数值稳定性，才允许最终联合引导。**训练λ不必等于OGG的w**。
- 当前候选损失不是在线成功率的证明。首先对比单层、Connector-only、分层加权与可选梯度冲突控制；遇不可靠层零向量、非有限梯度或超预算则按协议跳过/回退并记录，不暗中改变分层目标或重标度。
- 单动作当前设计用`a_t∈R^7`预测1秒变化，有条件信息不足；未来动作序列扩展须另列架构/消融，训练已执行示范动作与部署候选规划动作不能视为同分布；本轮不得与多层融合因素混合归因。

## 3C. **必做**在线消融：J-LVM-only 对照 J-LVM+OGG

这是用户明确要求的正式实验，不是可选项。**J表示Joint多层Corrector，不是新算法名。**

| method key（推荐） | 分层视觉监测与阈值 | LVM触发时中止旧队列并重新规划 | 本次新动作去噪时OGG | 解释 |
|---|---|---|---|---|
| `H50` | 否 | 否（按50动作前缀） | 否 | 长窗口参照 |
| `H20` | 否 | 否（按20动作前缀） | 否 | 短窗口重规划参照 |
| `j_monitor` | 是 | 否 | 否 | 监测-only安全/误报审计，**不是LVM-only** |
| **`j_lvm`** | **是** | **是** | **否** | **新增强制对照：仅LVM检测＋事件触发重规划** |
| **`j_lvm_ogg`** | **是** | **是** | **是** | **完整LVM+OGG** |

**代码事实**：既有`scripts/corrector_online_runtime.py`已经有`monitor/lvm/ogg`三种控制语义；`lvm`触发时清空动作队列但`self.active=False`，`ogg`会让后续`denoise_step`进入guidance。**不要平行再造状态机**；给原三种语义增加J多层预测器与多层观测桥接，method标签只在结果归档和CLI中明确`j_*`，或映射到原语义。第一步先查真实冻结部署版本；不要直接修改旧实验源码。

**核心公平条件：**
1. `j_lvm`与`j_lvm_ogg`必须加载**同一份J best checkpoint、同一冻结分层LVM权重、阈值、初始化与控制协议**，相同FPS/k步换算、monitor stride、滞回/冷却、动作队列实现、VLA revision、随机种子和相机配置。**两组唯一有意改变的控制模块应为OGG引导开关**，严禁在`j_lvm`里保留隐性梯度更新，也不能一组用三层分数而另一组只用Connector。
2. 强制单步重放测试：给定相同的观测/计划及已到期预测，检查`j_monitor`记录分数但不清队列，`j_lvm`清队列并`guidance_calls==0`，`j_lvm_ogg`具有相同触发判定并在触发后记录可验证的非零动作梯度/至少一次guidance调用；在**策略轨迹开始分叉之前**比较日志严格一致，分叉后不能强求触发时间完全相同。
3. 用旧`robot-lab` LIBERO-Plus的已审查条件：default、view15（开发视角）、view5/view10（保留视角），各seed1000/1001/1002，固定同一政策checkpoint/语言/初始物理状态，20Hz、chunk50/10次去噪。先校准和smoke，再冻结阈值与OGG超参，**最后**评估view5/10，禁止根据保留结果调参。基础`H50,H20,j_lvm,j_lvm_ogg`共4相机×3seed×4方法=48配对回合；加`j_monitor`则共60，按预算明确标注而非悄悄替换。前期监测只读安全验证不应和正式成功率混算。
4. 每回合记录成功/失败、总步数、完整动作块调用、监测时长、Flow Matching/OGG时长、事件次数、有效预测覆盖、报警误报线索、动作连续性、视频路径及分层`d_l`与恢复梯度；按相机/seed展示**成对救回与变差**，不是仅平均成功率。小样本不推断统计显著，也不把多层+OGG收益混同于单层架构收益。
5. 当成套J接口/参数/尺度/动作梯度验证未通过时，**不运行**`j_lvm_ogg`正式闭环；失败记录为未完成，不得修改Frozen结果，也不得把原来单层Corrector的旧`lvm/ogg`日志冒充J的成功率。

---

## 4. 在线部署：三个硬性门槛

1. 动作梯度：模型`eval`且所有参数冻结，输入当前7维动作`requires_grad`，验证`∂loss/∂a`有限非零；不能`detach`掉动作用于融合/skip路径。OGG原方案通过动作相关变量梯度引导候选动作；先使用独立强制触发smoke。
2. LVM：不同层的误差量级差异显著；各层归一化与阈值必须在独立正常轨迹上定标，再于view15开发与view5/view10保留条件测试，不能沿用Connector阈值。
3. 资源：先在4090实测microbatch1/2前向反向峰值、GPU time和在线`1step monitor`+OGG引导时延；不能仅按36.8M参数量判断可部署性。必要时混合精度、梯度累积与局部重计算，显式记录。

## 5. 面向Codex的约束与验收

1. **先查现有真实MVP源码**，重新核对77万模型的每个真实模块及形状；原Source SHA为`721f2bae4c3cf2c9ef7abaf548caf40ee9d9f92d`，目前不在GitHub已知分支，不可凭任务书推测已经存在的实现。
2. 先只实现本J-36M模型类与单元测试，不启动70或454条数据提取；使用合成Batch检查形状、3层交叉梯度（扰动Block2应改变Connector预测，及逆向梯度）、动作梯度、Patch顺序、双相机位置编码区分、full-res目标loss与checkpoint重载一致。
3. 再在已有5条Episode做真实小batch和过拟合测试，报告实测显存和吞吐；若CUDA OOM，先调整microbatch和梯度累积，不能偷偷改成width96/depth2再当作J-36M。
4. GPU smoke全部通过后，才启动多任务数据提取和正式训练。具体Episode集合/分片/磁盘上限由冻结manifest生成，避免训练过程中修改数据选择。
5. 不承诺本设计最优或在线收益；对照、失败和局限同等保存。
