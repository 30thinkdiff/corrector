# 2026-11-12 深夜实验方案：扩容后原始Block 2 / Block 8单层、联合Corrector与LIBERO-Plus

状态：**计划，未经实际运行；覆盖此前29-episode/16×16方案的优先级，但不删除历史文档或任何实验数据。**
明天主要做论文/实验沉淀与VLA微调知识准备；今晚允许长时运行，仍需阶段性验证、STOP、续训与预算保护。

## 研究目标

1. 用冻结SmolVLA的**未经池化**的Block 2、Block 8分别训练独立的**原始VLA-Corrector MLP架构**单层对照；保留与原Connector基线的可比性。
2. 训练一个多层联合Corrector（单个共享主干，分别预测Block 2、Block 8、Connector变化），与前述单层/Connector-only比较。可加一个多层输入但只监督Connector的消融C，分离特征输入与辅助任务。
3. 在此前`robot-lab`确实使用过的LIBERO-Plus alphabet soup相机条件下，先验证监控、打分、重新规划与OGG接口，再按相同种子和初始化做配对闭环。**离线MSE降低不等于成功率提升。**

## 核验后的技术事实

- 原版 `ZJU-OmniAI/vla-corrector/src/siglip_dynamics/MLP.py` 的`SiglipResidualMLP.forward`接受`[B,L,D]`、`[B,7]`，Token数L未硬编码128。单层Block2/8特征是两相机拼接`[B,2048,768]`，可令`token_dim=768`、保持原残差MLP等结构；参数量及计算/显存必须由GPU预检实测，不得假定旧`[128,960]`微批适用。不要把目前76.7万参数联合MVP冒充原论文约38M参数模型。
- 旧Connector `[128,960]`、量化缓存及动作时标原样复用。中层单独训练时预测自身`F[t+10]-F[t]`；原Connector预测自身差量。不同层的原始MSE不可横向直接比较，只能比较同目标下的zero基线改善、归一尺度指标或在线结果。
- `object10/manifest.json`固定Task24（alphabet soup）共**44 Episode、6867帧、35 train+9 validation**，train`5084`对、val`1343`对，预测跨度10帧/10Hz即1秒。该任务涵盖实验所需同一物体操作，恰好对应先前LIBERO-Plus alphabet soup条件；仍是单任务研究，不能推广Object10全部任务。
- 原始32×32双相机一个中间层每帧3 MiB，两个层6 MiB。Task24全部6867帧两个层约**40.236 GiB**，单层20.118 GiB（`.npy`元数据额外极小）。当前已经有813–817共749帧、两层4.39GiB在系统盘，但其中只有813属于Task24；如果两根缓存并存，任务加载器必须显式记录文件位置，不能凭新数据盘有总空间就重复提取813。
- 整个Object10共454 Episode、66984帧；两层原始FP16约**392.48 GiB**，另需原深层缓存、原始分片、训练输出和安全余量。**今晚不默认跑全454。**
- 新训练MVP于2026-11-12真实4090跑通，原源码`721f2bae4c3cf2c9ef7abaf548caf40ee9d9f92d`目前用户本地有、远端`dev/multilayer-training-mvp-a98b5be63190`有；**不能假定GitHub已包含该提交**，本地Codex先定位它并沿用。MVP的5轮A/B只是工程验收，不能代表扩容后的单层原始架构。

## 扩容与数据位置

AutoDL普通容器实例有系统盘和`/root/autodl-tmp`数据盘。官方文档表明：数据盘可付费扩容；单价按所在主机显示，关机后付费数据盘通常仍计日费；实例被释放则数据可能丢失，系统重置可能清空系统盘。因此扩容由用户在控制台确认容量与账单后操作；操作后`df -h / /root/autodl-tmp`并查磁盘实际挂载。

- **Task24整任务、未经池化两层、训练检查点与一些Plus评测：建议总数据盘150–200 GiB**（当前50GiB已使用约42GiB；40.24GiB raw、约3.30GiB源分片、训练/评测产物、运行余量均计入），选200GiB更宽松。
- 全Object10双层未经池化若计划长时间保留：总数据盘**约550–600GiB**更从容，具体是否可扩到该容量需看机器扩容上限和价格；不要求今晚购买如此大空间。若只能扩到500GB，先计算实际`df`可用空间和额外产物成本，不能盲目承诺可容纳全部。
- 系统盘`/root/multilayer_export_813_full_001/`和`/root/multilayer_export_file213_rest_001/`保留，若后续迁移请SHA校验后再删除源，而且不是本轮默认操作。旧Connector、历史Checkpoint、模型环境保留。
- 每个缓存目录显式标注episode、层索引、camera顺序、shape/dtype、模型revision/weights SHA、旧Connector及action来源、时间戳与分片SHA。发布`COMPLETE.json`最后一步，未完成绝不能进入训练；每次启动校验完整性。
- 扩容和容量阈值用于防止写满：系统盘保留至少8GiB、数据盘保留至少15–20GiB（扩容后的推荐值）；不足时按完整Episode边界停止，而非清理旧权重或跳过SHA。原始Parquet仅下载指定task所需37个不同物理分片（约3.30GiB，依据manifest大小）且只有缺失才下载、固定数据修订，下载成功与已用分片需统一校验。每条Episode缓存确认COMPLETE后才允许逐分片清理**可复取且无未完成episode引用的原始分片**；默认不清理，确需清理先产出候选清单并供人审，不碰原始中层FP16/Connector/Checkpoint。

## 工作流与阶段依赖

### 阶段0：本地Codex准备与远程preflight

本地Ubuntu22.04使用Codex，检查上次真实MVP的原始代码/ZIP/冻结SHA，从中**增量**实现新方案；提交新Git分支并生成独立ZIP+SHA与明确上传/解压/启动/STOP/查看/恢复命令。远端不要求Codex登录，不覆盖已冻结`dev/`代码。预检扩容与实际`df`、源Parquet清单、旧深层454条cache SHA、GPU空闲、依赖环境，以及已有5条raw特征来源。

### 阶段1：Task24特征提取（首批完整实验）

先从已存在的813 raw重用Block2/8，其余Task24 Episode按需下载、冻结模型一次前向同时提取两个层；**不池化、不INT8**，连续`[T,2,1024,768]` FP16独立存储。逐Episode写入/恢复，旧Connector由原`object10/features`复用。先至少2 Train+1 Validation做提取和DataLoader smoke，然后按Task24全量44条完成。未来delta标签自动构造，禁止跨episode和未来观测泄漏。额外验证测试数据是否覆盖同任务多个初始化，而不是只依据TaskID判断多样性。

### 阶段2：三类训练，分别独立输出

- S2（原版MLP）：Block2输入`[B,2048,768]`+单动作7维，预测自身变化（原上游MLP骨架，仅配置token_dim=768）。原训练参考协议可选择MLP 38M规格，必须先做真实microbatch1/2前向反向/内存与梯度验证；若过重，显式建立减小模型的独立变体，不得冒充论文同规模。报告梯度、参数数、每epoch时长、训练RMS、损失。
- S8：对Block8做同样实验，固定相同训练/验证Episode、时距、action规范、模型超参/预算/seed等。
- J（联合）：冻结视觉编码器+Block2/Block8/原Connector融合成**一个共享主干和三个输出头**，归一化层内损失；至少有Connector-only小模型A与多层输入单监督C对照（如预算允许），不能只将3层raw MSE相加而当成性能。模型/梯度与三种输入不同维数及token数，不能把两个层直接拼成一个128-token表示而不说明变换。
- 训练可长时运行但不能无限运行：先对每种配置记录GPU吞吐、峰值内存并估计完成时间；每模型预设epoch上限和早停，按**完整Epoch**原子保存可恢复best/latest及优化器+RNG，遇STOP在边界退出；没有用户批准不改变冻结计划、自动增购或自动放大seed数。
- 统一记录train/val每层原始MSE、归一MSE、zero baseline、cosine，每TaskID/每Episode指标、跨seed变动及训练时长；S2与S8原始MSE不直接横向比较。

### 阶段3：离线与在线接口闭环

先拿单层和联合best在训练任务的旧验证Episode进行offline、fake observation/shape/timebase与动作梯度校验，再进入Object10仿真与LIBERO-Plus。
- 不能直接拿旧Connector的LVM阈值去判Block2/Block8/联合模型的误差：必须在未参与训练的正常执行轨迹上校准、冻结，再去扰动视角最终测试；需要区分“正常成功动作”与“相机移位导致的表示域偏移”。
- 训练10Hz/k10=1秒对应在线20Hz/20控制步；相机Hook必须确实取到正确层，保持双相机顺序、bf16模式、预测输入形状、与策略相同权重。Monitor-only不应改变动作；OGG必须通过动作变量求出有效梯度，各头按尺度归一，不得将不同维的输出直接粗暴相加。
- 在线两层`[2048,768]`可能产生高额每步计算开销；记录分组件耗时、整回合总耗时、监测打分覆盖率、触发/撤销次数及相对重规划基线的收益；若超过预算，记录为可部署性限制，不默默换16×16架构后继续称“原始层”。

### 阶段4：沿用已有LIBERO-Plus相机实验配对条件

从`30thinkdiff/robot-lab`历史正式结果`studies/policy_generalization/replanning_results.md`复用：
- 冻结`lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`；
- LIBERO-plus源码固定`4976dc30028e805ff8094b55501d532c48fec182`，基础alphabet soup，指令`pick the alphabet soup and place it in the basket`；同初始化，20Hz仿真、chunk_size50、10次去噪，种子1000/1001/1002；
- 四个相机条件：基础默认、`view_15_15_100_0_0_initstate_0`（选定条件）、`view_5_15_100_0_0_initstate_0`、`view_10_15_100_0_0_initstate_0`（保留视角）。历史配对成功数：
  | view | H50 | H20 |
  |---|---:|---:|
  | default | 3/3 | 3/3 |
  | 15 | 1/3 | 3/3 |
  | 5 | 0/3 | 1/3 |
  | 10 | 1/3 | 2/3 |
- **优先在默认条件核验异常检测与假阳性**，再在view15做开发试验；view5/10保留为晚期测试，不用它们校准阈值/选模型/调OGG超参。
- 先做H50/H20复现和monitor-only安全验证，再以固定H50基线加入S2/S8/J各自LVM、条件重规划、完整OGG，逐级比较；如果任一接口/阈值/成本未通过验证，不继续自动运行更高风险组合。只在调参与接口冻结后做view5/10，逐seed配对成功、失败类型、执行步数、调用数和耗时。
- 不把历史3/3等小样本成功率当作真实稳定成功率；新运行须重新检查Plus任务语言、相机外参、机器人初始化状态、首噪声/策略输入的实际一致性。避免将普通LIBERO Object10仿真成绩称作Plus相机鲁棒性结果。

## 明天的交付物和学习重点

- 用自动摘要把数据准备→原上游单层→联合→离线预测→在线监测/OGG→Plus结果串成一个**证据链**，每步标示版本、任务、分布、指标、局限。产生机器可读`experiment_index.json`、逐轮CSV、逐任务/条件表、异常日志索引、资源/账单摘要、可复现命令；权重和原始FP16依然不入Git。
- 明天不安排新VLA微调任务。集中学习：SmolVLA视觉Encoder/Connector→动作专家的输入输出与冻结/可训练模块；动作chunk与Flow Matching；微调数据格式（episode、相机顺序、动作归一、时标、语言）；LoRA/全参数训练/混合精度/显存；相机域变化增强、训练/验证拆分、在线评测；再决定下一批微调数据与目标。
- 不保证今晚按墙钟必然完成全部阶段；若卡于磁盘/模型训练/仿真版本，应在上个已验证阶段保存，明天有可审计结果可总结，而非冒险跳过门槛。

## 本地Codex实现要求

先读本文件、`studies/policy_generalization/replanning_results.md`（robot-lab）、之前MVP真实源码和提取脚本；基于实际可用结构提交独立新分支和可执行ZIP。整个流水线按`preflight→extract→verify→single_block2→single_block8→joint→offline_eval→online_smoke→Plus_heldout→summary`，支持STOP、resume、预算和合理资源安全边界。运行中高频状态写JSON；权限和网络问题时离线部署不依赖远端Codex。只能在经过前一阶段真实成功验收后进下一阶段，不能把计划写成已完成实验。
