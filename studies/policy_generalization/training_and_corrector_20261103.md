# 2026-11-03：从已验证的推理实验进入训练闭环

> 历史准备单：下文“尚未训练”等状态对应11月3/4日。11月5日真实训练、重载与离线评分已完成，见[训练结果](training_results_20261105.md)。当前任务以[Corrector机制实践](corrector_mechanism_practice.md)为准，不重复追赶旧日期清单。

2026-11-04补充：用户已实现参数统计，my_cpu_001/002为两组成功的CPU LoRA机制实验，
003/004在adapter创建时失败，尚未完成全参数对照。最新下一步与未实现项见
[实践记录顶部](lora_training_practice.md#2026-11-04进展核对与下一步)；下文保留11月3日审计时的历史状态。

当前实际状态：已有30个执行窗口评测回合与失败诊断；用户已初读Corrector论文，尚未运行真实PEFT训练。
14:51后补充：[LoRA机制与CPU/GPU实践入口](lora_training_practice.md)已补齐。CPU官方PEFT单模块机制实验已实测，
不计作完整SmolVLA训练；单单元格远程准备入口及GPU真实batch/5步审计代码已准备，真实GPU执行仍待进行。
本轮完成的是官方源码/数据元信息核对，不把阅读、配置准备或CPU检查记作GPU训练完成。
按下列验收点推进，原先11月2/3/4日的日期安排不作为必须追赶的进度。

## 1．两阶段的关系

第一阶段：以官方SmolVLA PEFT入口补齐真实batch、前向、反向、参数更新、保存及重载能力。
这是训练能力验证，不以20步内成功率提高作为验收，也不以单步loss必须下降作为验收。

第二阶段：使用冻结的、已能完成任务的VLA研究潜在动力学监控和事件重规划。
Corrector不要求先用LoRA重新训练已有有效策略；训练练习与原相机基线分开保存，避免改变既有对照。
完整论文还包含OGG（Online Gradient Guidance，在线梯度引导）。只做预测/监控/截断，应称子机制复现，不称完整方法复现。

## 2．已核对的环境与接口

本地训练源码位于已安装的LeRobot 0.6.1；检查得到PyTorch 2.11.0+cpu，
transformers、peft、accelerate尚未安装。因此目前不能声称该环境已能加载完整策略或运行训练。
没有为此升级/安装依赖；下一次GPU会话应重新确认实际导入路径与源码，不能只凭0.6.1版本号判断代码相同。

| 要查什么     | 本地入口                                                                     | 关键事实                                                                   |
| ------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 官方训练入口 | `lerobot/scripts/lerobot_train.py` 的 train / update_policy                | forward → accelerator.backward → 裁剪梯度 → optimizer.step → zero_grad |
| 注入LoRA     | `lerobot/policies/pretrained.py` 的 wrap_with_peft                         | 先冻结原参数，再调用get_peft_model                                         |
| 默认目标层   | 本项目 `policies/smolvla/modeling_smolvla.py` 的 _get_default_peft_targets | expert的q_proj/v_proj及五个state/action/time投影                           |
| 训练配置     | `lerobot/configs/train.py`、configs/default.py                             | 支持peft、dataset.episodes/revision；本地仿真评测频率字段为env_eval_freq   |
| 保存训练状态 | `lerobot/common/train_utils.py` 的 save_checkpoint                         | adapter、策略配置、processor及training_state应分别核查                     |
| 加载adapter  | `lerobot/policies/factory.py` 的make_policy                                | use_peft路径先读adapter配置，再加载对应base和adapter                       |

五个投影：state_proj、action_in_proj、action_out_proj、action_time_mlp_in、action_time_mlp_out。
默认是对这些层插入adapter；不要把它误解成默认全量更新这些层。full_training_modules另行控制全量训练。
新建LoRA使用 `--peft.method_type=LORA`；不要提前把base配置的use_peft手工设为true，
本地factory会将该标志用于“加载已有adapter”的分支。

训练参数变少不保证激活显存等比例下降。即使某层权重被冻结，梯度仍可能需要穿过该层传播到可训练输入。

## 3．真实训练数据已经定位，尚未下载主体或读取真实batch

官方示例数据集：`HuggingFaceVLA/libero`。
本轮固定revision：`86958911c0f959db2bbbdb107eb3e17c5f9c798e`。
元信息为v3.0、1693个episode、273465帧、40条任务，标注10 FPS。

| 数据字段                          | 元信息                                           |
| --------------------------------- | ------------------------------------------------ |
| observation.images.image / image2 | 两路256×256×3图像，dtype标为image              |
| observation.state                 | 8维float32                                       |
| action                            | 7维float32                                       |
| task_index / episode_index        | 数据集任务索引与示范轨迹编号，不能替代仿真任务ID |

与原操作对应的指令为 `pick up the alphabet soup and place it in the basket`，
数据集task_index=24，共44个episode。可用813、818作为训练冒烟样本，825先保留不训练。
其长度分别为155、164、153帧。11月3日按元信息data/file_index判断为file-037的记录有误：
11月5日已读取真实文件核对，实际分别位于data/chunk-000/file-213.parquet、file-214.parquet、file-215.parquet。
file-037实际只有90/91/92号轨迹，不能用于上述训练episode。
数据集前两个episode是别的任务，所以不能盲选0、1后声称在训练原任务。

还需要实际batch核查：

- image与image2分别对应哪台相机、顺序是否正确；仅看字段名和shape不能确认。
- 图像加载后的布局、数值范围与预处理；state/action的坐标、单位、夹爪及相对控制约定。
- 10 FPS与旧仿真20Hz的采样/执行关系。相同维度不足以证明可以直接沿用全部旧评测设置。
- action_chunk及action_is_pad是否正确；数据集选择少量episode不保证只下载几条轨迹的字节数，文件可能包含多个episode。

首个数据文件约100.7MB，仅用于说明下载粒度；本轮没有查询/下载目标file-037主体，不能把首文件大小当成其精确大小。
旧的诊断PT只保存首六秒部分物理量/执行动作，Trace还只有少量策略输入；它们不是完整专家示范数据集。

## 4．最小训练命令草案：先完成batch核查再执行

资源补充：用户目前可连接Colab T4，但额度不确定。本地可用内存与磁盘余量有限，且没有完整权重/训练依赖，
当前分工为本地准备和CPU检查，T4承担完整模型的真实batch与反向更新；完整CPU训练速度尚未实测。
首轮缩为5步并在第5步保存，核验和持久化checkpoint后，再按官方resume路径续到总计20步。

用途是训练闭环冒烟。起点按官方示例采用smolvla_base，原成功评测的smolvla_libero保持为另一个实验。
base固定revision：`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`。
以下字段已与本地源码核对，但尚未通过GPU训练器实际解析/运行，也未验证T4或4090上的峰值显存。
GPU会话需先核对自己的训练字段，尤其env_eval_freq；不盲目复制到不同版本。

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.pretrained_revision=5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=0 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.revision=86958911c0f959db2bbbdb107eb3e17c5f9c798e \
  --dataset.episodes='[813,818]' \
  --peft.method_type=LORA \
  --peft.r=8 \
  --peft.lora_alpha=8 \
  --batch_size=1 \
  --num_workers=0 \
  --steps=5 \
  --log_freq=1 \
  --save_freq=5 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir=outputs/train/smolvla_peft_smoke_20261103_001
```

说明：默认目标层保留，不同时改模型结构；先禁用额外仿真评测和外部上传。
此命令仅保存Colab本地checkpoint，尚未自动备份到Drive。5步结束后应将整个已完成step目录
（包括pretrained_model和training_state）归档并验证完整，再继续。只保留adapter不足以承诺断连后可续训；
只有最后一次完整持久化的checkpoint可作为恢复入口。不要对已有输出目录按非resume方式重复启动。
warmup设0以避免刚开始学习率为0时，把“没有参数变化”误判成反向传播断开。
use_amp=false仅是首轮精度策略，不保证权重全部是float32，也不保证低显存；需打印真实参数/计算dtype，T4与4090分别验证。
遇到OOM先定位峰值来源，不为强行跑通而同时修改分辨率、模型结构和动作长度。

## 5．验收顺序：先20步闭环，再讨论效果

| 门槛               | 需要保存的证据                                                                         | 当前状态                  |
| ------------------ | -------------------------------------------------------------------------------------- | ------------------------- |
| G0：数据与接口     | 一个真实处理后batch、字段语义、padding和匹配的checkpoint/源码版本                      | 元信息已查，真实batch待查 |
| G1：真实更新       | 可训练参数名/数量；有限loss；部分adapter有效梯度；step后至少一个参数确实变化；峰值显存 | 未进行                    |
| G2：保存和重载     | base的不可变版本、adapter/config、processor统计；固定输入和初始噪声下重载前后输出误差  | 未进行                    |
| G3：继续训练与评测 | optimizer/scheduler/RNG/step状态；续训一步；训练前后采用一致的数据和环境协议           | 未进行                    |

梯度必须在backward之后、zero_grad之前观察。官方update_policy返回时已执行zero_grad，不能只在返回后检查p.grad。
LoRA默认A随机、B为0，初步反向时部分A梯度为0并不异常；不要要求所有adapter第一步都非零。
loss随随机噪声和生成时间变化，不要求20步逐步单调下降；先验证数值有限、更新发生及保存加载正确。
重载PEFT不应把adapter目录直接当完整SmolVLA权重；走官方make_policy/PEFT加载路径，并带上processor。
恢复推理不等于恢复训练，只有adapter还不足以保证optimizer和随机状态可续接。
训练前后的小规模评测用于验证评测流程，不预设20步必有成功率增益；原相机实验不混入此新base策略的结果。

## 6．Corrector代码核对：现在只建立地图，下一阶段再实现

官方参考源码固定于 `927a06da8f00d7df483459e9039219e076c846de`，
本地只读副本在outputs/source_audit/vla-corrector，pyproject声明LeRobot 0.5.1，不能直接覆盖现有0.6.1。
官方没有附带训练数据或已训练Corrector权重，完整复现需要另备数据、提特征、训练与验证。

| 核心环节        | 官方源码位置                                                     | 输入/输出或职责                                                         |
| --------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| SmolVLA视觉特征 | src/siglip_dynamics/extract.py，SmolVLAExtractor                 | 图像预处理后调用vlm_with_expert.embed_image，不是预测RGB或机器人8维状态 |
| 时间配对        | src/siglip_dynamics/dataset.py，__getitem__                | z_t、a_t，标签为z_{t+k}−z_t；MLP使用z_t而非完整历史                    |
| 外部MLP         | src/siglip_dynamics/MLP.py，SiglipResidualMLP.forward            | z_t[B,L,D] + a_t[B,A] → delta_z[B,L,D]                                 |
| 训练损失        | src/siglip_dynamics/train.py，_combined_loss                     | 支持mse/cosine/both；论文一般式组合二者，README部署示例使用cosine       |
| 在线监控/截断   | lerobot_eval_modified_detection.py与inference/circuit_breaker.py | 预测/观测特征比较，持续偏离时丢弃过时队列并重规划                       |
| SmolVLA梯度引导 | policies/smolvla_modified/modeling_smolvla_modified.py           | 修改后的采样路径对速度副本求导，不是在此更新VLA权重                     |

理论最小关系：z_t=E(o_t)，监督目标delta_z=E(o_{t+k})−E(o_t)，MLP用(z_t,a_t)预测该变化。
它不要求人工为每个样本标“抓取失败”；物理接触真值可作离线诊断，不应悄悄加入视觉监控的部署输入。
MLP输出不是Flow Matching速度，也不是直接输出纠正动作。
训练Corrector的梯度更新MLP权重；OGG在冻结权重时求采样变量的梯度，这两件事需分开。

注意：10 FPS下k=10对应1s，20Hz下10步为0.5s。训练缓存和在线比较必须明确物理时间、动作归一化、相机顺序与特征来源。
先按episode划分训练/验证，再组成时间对；避免相邻帧泄漏。以前看过并用于分析的view_5/view_10，不能在调新方法后仍宣称是全新未见测试集。
下一阶段可先做“预测+监控+普通事件重规划”，保留固定50/20步对照；OGG单列为后续部分，名称与实验结论保持一致。

## 7．用户今天只需先掌握的两段代码

先看本项目_get_default_peft_targets，再看本地lerobot_train.py的中forward至zero_grad这段。
各自用几句自己的话说明：哪些模块插入LoRA、原权重是否更新；loss如何产生梯度、梯度在什么时候可以观察。
有真实策略对象后，由用户亲手补一个短循环统计总参数/可训练参数、列出可训练名字；助手协助检查，而非先生成整套训练框架。
不需要重背整个Transformer、所有shape或整篇Corrector实现，GPU就绪后直接用真实batch验证上述理解。

参考：[官方PEFT文档](https://huggingface.co/docs/lerobot/peft_training)、
[Corrector论文](https://arxiv.org/abs/2607.01804)、
[固定源码](https://github.com/ZJU-OmniAI/vla-corrector/tree/927a06da8f00d7df483459e9039219e076c846de)、
[LoRA初始化](https://huggingface.co/docs/peft/developer_guides/lora)。
本轮元数据与源码字段/SHA审计保存于outputs/training_audit_20261103。只新增本准备单，不执行训练命令或改变原基线。
