# LoRA如何进入SmolVLA，以及这次实际练什么

## 当前状态：训练练习收束，转入Corrector

GPU导出和用户指标评分已完成；闭环入口保留作可选验证，当前主线见[Corrector机制实践](corrector_mechanism_practice.md)。下方阶段说明保留供复用。

用户已完成masked_action_mae，核心算法保留；助手补齐形状/空输入/非有限数检查并完成CPU评分。
真实base MAE 1.066619→LoRA MAE 1.061330，三个验证时刻的小样本结果不能代表任务成功率。
详见[离线结果](offline_eval_results_20261105.md)。下一步入口是Notebook新增的LIBERO准备/成对执行/录像，
核心调用链、频率限制与留给用户的短练习见[闭环评测实践](libero_evaluation_practice.md)。

2026-11-05：真实五步训练、非零梯度、参数更新、保存与重载已经完成。**不需要等指标写完再训练，也不用重新跑inspect/practice/train。**

你亲写的函数现在位于[evaluation_metrics.py](evaluation_metrics.py)，不是本文的Markdown代码块：

```python
def masked_action_mae(pred, target, is_pad):
    ...
```

接口约定：pred/target为[B,H,D]，本次D=7；is_pad为bool[B,H]，True表示补齐位置，需要排除。
返回一个Python float，表示所有有效动作标量的平均绝对误差。分母为有效时间位置数乘D；不是先对每个样本求平均再等权相加。
全padding、形状不匹配、非bool mask应明确报错。函数主体已由用户实现，并通过CPU检查。

调用链已经接上：workbench的check-metric/score-eval → smolvla_eval_tools → load_metric → masked_action_mae。
score-eval既计算整体指标，也计算每个样本的指标。**export-eval不调用它**，因此指标未完成也不占用GPU等待。

### 已完成的T4导出步骤（无需重跑）

Notebook准备单元格现在根据工作目录自动判断：已有会话只同步代码，新运行时重新准备。
执行该单元格后，跳过原来的STAGE训练单元格，直接执行新增的“验证集预测导出”单元格。
结果保存在已挂载Drive的evaluations/heldout_825_001中；已有结果不会覆盖，新实验需换EVAL_RUN。

导出从官方五步checkpoint加载策略与保存的预处理器，用未参加训练的episode 825的0、76、152帧，
固定状态、语言、图像、初始噪声和推理配置，对比禁用/启用LoRA的动作预测。这里只有LoRA参数经过更新，故禁用adapter可作为冻结base对照。
首个样本重复预测，检查数值重复性；每完成一对就保存CPU预测和进度，最终complete=true才算完整导出。
同时保存各帧输入图片；不执行optimizer更新，不调用LIBERO，不产生MP4。

拿到完整导出后，可以释放GPU。本地下载paired_predictions.pt和evaluation_manifest.json到**同一目录**，原始文件保留。
先在本地实现函数，再运行：

```bash
python scripts/smolvla_training_workbench.py check-metric
python scripts/smolvla_training_workbench.py score-eval \
    --predictions outputs/你的评测目录/paired_predictions.pt
```

check-metric使用小张量检查padding和返回值，不加载模型；用户实现已通过检查。
score-eval检查导出完整性和SHA256，保存时间戳命名的score JSON和指标源码快照，不覆盖原预测。
trained_minus_base_mae小于0表示本次验证动作误差降低；这里是训练时标准化空间，不是米、弧度或任务成功率。
三个时刻只用于第一次评测链路检查，且来自同一episode，不能声称已验证泛化或训练收益。

GPU导出已在T4实测通过；LIBERO接入已补代码和独立仿真环境准备，完整模型闭环尚待T4实测。
没有向已跑通的训练环境混装仿真依赖；先通过真实场景/录像检查，再加载保存策略。

---

2026-11-05 10:09：已核对真实GPU的inspect、用户单步、官方五步和归档重载，详见[结果记录](training_results_20261105.md)。
G0/G1/G2已实测；G3续训与独立任务评测仍未完成。以下“待T4重跑”等文字为排错历史，不是当前待做清单。

2026-11-05第二次inspect失败已定位：配置读取已通过，数据集episode过滤为空。预下载file-037实际包含90/91/92，
发布元信息里的data/file_index=37不对应物理分片。之前仅凭元信息选文件的检查不充分。
固定revision下已通过Parquet页脚和实际数据行核对：813→file-213（155帧），818→file-214（164帧），825→file-215（153帧）。
已本地下载这三份实际文件，使用LeRobotDataset和datasets4.8.5读取，训练集319帧、图像[3,256,256]、状态[8]、动作[50,7]、末帧49个padding全部通过。
证据：outputs/training_audit_20261105/real_dataset_probe/dataset_check.json；825保留不参与训练。
准备入口现在先核对数据内容，再用官方读取器在CPU上读真实样本，不只检查文件存在；旧manifest会因data_files变更自动补齐分片。
补齐数据时保留已记录VLM SHA，旧错误分片和日志不删除；Notebook已同步。完整模型inspect仍待T4重跑，不能把CPU数据检查当成GPU训练。
本次run.log（私人链接不公开，原始记录保存在私有归档）。

2026-11-05 09:53更新：inspect在CLI配置校验时失败，尚未读取训练batch或执行更新。
准备已缓存base的固定SHA，但CLI配置加载没有转发pretrained_revision，默认请求main；离线模式下找不到main引用。
已在本项目运行入口限定作用域补齐配置revision，保留Hub base标识及adapter的可迁移性，不修改安装包或缓存refs/main。
准备阶段同时新增固定SHA配置的离线解析检查。真实Hub缓存与真实LeRobot CLI解析均通过本地复现；T4需重跑确认后续加载。
Notebook已同步新脚本，保留失败输出；先执行更新后的准备单元格，再运行STAGE=inspect，不需删除缓存或修改用户训练函数。
本次日志见inspect run.log（私人链接不公开，原始记录保存在私有归档）。

更新：2026-11-04 17:44进展核对。CPU官方PEFT机制实验已经实际运行；T4环境安装、导入检查与资产下载已成功；完整策略加载、真实batch和GPU训练仍待验证。

## 2026-11-04进展核对与下一步

### 视频要求：训练检查后另做仿真执行与录像

用户要求能观看真实策略执行视频，纳入后续验收。当前训练命令env_eval_freq=0，没有调用LIBERO仿真，
所以inspect/practice/train均不会产生机器人执行MP4。train末尾的predict_action_chunk比较是离线重载检查。
policy.eval()只是切换模块模式，也不等于执行环境评测。

已有LeRobot评测入口支持保存videos/<任务目录>/eval_episode_0.mp4与eval_info.json，旧相机Notebook也有视频显示逻辑。
但新训练环境尚未安装/验证完整LIBERO仿真依赖，新adapter也尚未接入仿真评测；不能把旧策略视频当作本次微调结果。

下一步顺序：真实batch检查→用户单步→官方五步/保存重载→单独准备仿真评测→运行一个完整episode并观看视频。
视频应来自加载保存checkpoint的闭环执行，成功/失败均保留；先核对任务、相机字段、状态/动作归一化、
控制模式与频率，并记录真实渲染/播放FPS。当前数据10 FPS与既有仿真20Hz的关系不能只凭维度一致略过。

计划将评测结果保存到独立eval_after目录，包含videos、eval_info.json和完整日志，并复制到Drive。
之后做训练前后效果比较时，两边必须使用同一评测协议，初始策略也需采用匹配的数据接口与processor。
单回合视频用于检查行为，五步训练不预设已有任务成功率增益。视频生成与自动归档尚待接入及GPU实测。

### 用户单步实现已完成，准备进入T4训练

用户已经亲写train_one_batch，并将返回值修正为单个float。CPU检查现在直接调用该函数，验证训练模式、
旧梯度清理、一次参数更新和运行入口的返回协议；没有改写用户函数体。完整SmolVLA GPU执行仍待进行。

Notebook已同步最新版PRACTICE_SOURCE，保留原准备成功输出和用户GPU检查单元格。
已有的--inspect-only单元格扩展为阶段入口：只改STAGE，按inspect→practice→train逐步执行，不一次全部运行。

| STAGE | 实际执行 | 结果 |
| --- | --- | --- |
| inspect（默认） | 官方加载模型/处理器/真实batch；不更新 | batch_inspection、图片、摘要；挂载Drive时复制检查材料 |
| practice | 同步并显式调用用户函数一次；旧远程文件先备份 | 独立practice_step，记录梯度/参数变化、adapter及optimizer |
| train | 重新从初始base创建模型，运行官方5步及重载推理比较 | train_run及完整checkpoint归档 |

三个阶段不是自动续训。准备流程仍保留一个单元格；执行入口是后续另一个单元格。
每次启动会将environment.json复制到本阶段日志目录，Drive可用时一同持久化，保留VLM解析SHA等运行信息。
核心代码和运行框架已具备；接下来先验证真实batch，不要求用户继续补齐整套训练器。

### 17:44核对：准备单元格已通过

本地Notebook第一个单元格与Drive最新run.log一致：SmolVLA/官方训练入口导入成功，pip check无损坏依赖，
Tesla T4显存约14.56GiB，选定数据文件、smolvla_base及SmolVLM2权重下载完成。
Drive status.json确认ok=true、mode=prepare；这不代表真实模型前向/反向已执行。
实际核心版本为LeRobot0.6.1、Torch2.11.0+cu130、PEFT0.21.0、Transformers5.5.4、Accelerate1.14.0。
训练解释器是/content/smolvla-training/venv/bin/python；原Notebook内核不是这个新venv。
第二个单元格中的CPU/CUDA错误来自本地ws_lerobot路径，不能拿它否定此前T4会话的成功。

资产下载到/content/smolvla-training（数据与HF缓存），当前归档到Drive的是准备日志；未备份整套运行环境/权重。
重新分配Colab运行时可能需要重新prepare；不把断开连接等同于运行时必然已经销毁。
旧段落中的“待重跑”为首轮失败时的历史状态。下一步先用--inspect-only读取真实batch，此入口不依赖用户的单步函数。

日志链接：最新run.log（私人链接不公开，原始记录保存在私有归档）。

### T4首轮准备报错与修复

本轮已连接Tesla T4；Notebook和Drive setup_logs/20261104T082754.959765Z_prepare/run.log均显示：
Python3.13.16、约112.54GiB磁盘空余，失败于venv内部的ensurepip，还未安装LeRobot或加载模型。
日志没有ensurepip内部的完整异常，因此不能断言缺少某个具体apt组件。
旧代码只检查bin/python是否存在，会把失败留下的半成品误当成已就绪环境。

已改为venv --without-pip创建Python，再用宿主pip>=22.3的--python选项向目标环境安装pip。
宿主没有可用pip时才尝试环境内ensurepip，并记录其直接输出；不升级系统pip或重装宿主Torch。
重跑会检查Python所属环境与pip可用性，补齐半成品，不需要删除venv或断开T4。
Notebook准备单元格的错误提示已指向实际LOG_ROOT，用户添加的其他单元格保留。
修复前完整Notebook在本地outputs/training_audit_20261104/prepare_before_bootstrap_fix.ipynb保留。

本地已实测“无pip环境的修复+再次复用”；本次Colab Python3.13的依赖安装/入口导入已通过，策略加载及训练仍待执行。

### train_one_batch现在在哪里、由谁调用

实现文件：[train_one_batch.py](train_one_batch.py)，六步已由用户亲写完成；Markdown仅描述接口。
单单元格准备会在远程WORKDIR/train_one_batch.py生成同一模板；已有远程文件会保留，避免覆盖你的实现。
仅运行准备单元格、--inspect-only或默认五步训练，都不调用它。
完成函数并同步远程文件后，显式运行以下入口才会调用一次：

```python
subprocess.run([
    str(WORKDIR / 'venv/bin/python'), str(SCRIPT), 'train',
    '--workdir', str(WORKDIR),
    '--practice-step-file', str(WORKDIR / 'train_one_batch.py'),
    '--archive-dir', str(ARCHIVE),
], check=True)
```

这里要求ARCHIVE已经是实际挂载的云盘目录；若未挂载，应去掉--archive-dir及其值，并另外保存结果。
函数接收官方入口已创建的策略、预处理后的batch和optimizer；不用在函数里再次下载、归一化或创建optimizer。
调用关系为train→audited_update→你的train_one_batch(policy,batch,optimizer)。
审计确认一次有效更新后保存adapter、optimizer和证据，并主动结束；有archive-dir时直接保存到其practice_step目录。
该目录是单步练习材料，不是包含processor/scheduler/RNG的官方可续训checkpoint。
默认五步入口仍调用官方update_policy，另写train_run；不与练习更新混在一起。

本地改完函数后，重新export-notebook会更新嵌入的PRACTICE_SOURCE；准备阶段保留远程已有文件。
显式选择STAGE=practice后，执行入口会备份旧远程实现，再同步并调用Notebook携带的用户函数。
若自行运行上面的subprocess命令，则需自行同步这个单独的.py文件。运行前应核对所用代码版本。

用户已亲手实现可训练参数统计，并执行以下实验。记录依据各目录的adapter_config.json、evidence.json与status.json；
主脚本经过修改，不能用当前源码倒推每次历史实验配置。

| 运行目录（outputs/training_workbench下） | 实际方法                        | 可训练参数 | 第1/第5次前向loss       | 状态                           |
| ---------------------------------------- | ------------------------------- | ---------: | ----------------------- | ------------------------------ |
| my_cpu_001                               | q_proj的LoRA，r=8               |      13440 | 0.01004257 / 0.00968884 | 训练、冻结检查与重载通过       |
| my_cpu_002                               | q_proj和o_proj的LoRA，r=8       |      26880 | 0.01004257 / 0.00935371 | 训练、冻结检查与重载通过       |
| my_cpu_003                               | LoraConfig未指定target_modules  | 未进入训练 | 无                      | 无法自动推断自定义模型的目标层 |
| my_cpu_004                               | LoraConfig的target_modules=None | 未进入训练 | 无                      | 同上                           |

001/003各有一次重复启动，被已有目录保护拒绝；它不推翻先前运行的成功或失败。
002仍是LoRA，不是全参数训练。target_modules=None的语义是尝试使用模型类型对应的默认目标层，
不是“不冻结参数”或“训练全部参数”。即便给所有线性层插入LoRA，也不等于更新原权重。
如果练习全参数训练，需从新建的普通模型开始，不调用get_peft_model，并重新核对requires_grad与optimizer。
现有cpu_lab的冻结断言、base_model路径及adapter保存均面向LoRA，不能仅删一个配置字段后原样用于全参模式。

这两组成功运行支持的结论是：在这个合成输入、固定batch和优化设置下，增加一个LoRA目标层后，第5次前向训练loss更低。
两组没有独立验证集，不支持“更多参数一定更好”或“机器人成功率提高”。本次不扩展更多小模型对照。

### 下一次T4会话：先完成真实训练与重载

1. 运行单单元格准备，日志持久化；本轮安装与下载已实测通过，重新分配运行时后按需恢复。
2. 用--inspect-only检查真实batch，保存两路图像、任务文本/token、state/action与action_is_pad。
   确认归一化、相机顺序和数据10 FPS的动作语义；仅维度一致不算完成。
3. 运行官方5步训练，检查有限loss、部分非零梯度、参数变化、显存和adapter/processor重载。
   数据来自真实专家示范，这一步本身已经是微调；5步只是验证工程流程，不是充分训练。
4. 完整checkpoint归档后再实现/验证官方resume。现有入口没有一键续训到20步，也没有完整训练前后评测。
   先按实际步耗时和显存决定短程训练预算，不预先承诺T4额度内训练出有效策略。
5. 质量评估单列：训练episode为813/818，825暂不参与训练；按episode划分，避免相邻帧混入验证。
   对固定验证样本、噪声/Flow Matching时间比较速度损失，并对固定初始噪声比较动作块预测。
   动作误差只统计有效时间/真实动作维度，并区分标准化空间、原尺度和各动作分量。
   这些评估代码尚未实现；离线误差与仿真成功率分别报告，正式仿真须先对齐相机、控制和频率。

准备阶段已实现：固定源码/base/数据版本、选定Parquet下载、官方5步入口、梯度审计、checkpoint归档与推理重载。
已实测：T4依赖安装、入口导入、pip check与资产下载；待实测：真实batch、完整策略的显存/更新与重载。
待实现并实测：GPU续训、独立验证指标、匹配协议的训练前后任务评测。
当前训练从smolvla_base起步；既有相机失败诊断来自smolvla_libero，不混合两者的结果。

### 留给用户亲写的核心

环境准备/归档可以复用；不要求先读懂600行工具代码。
用户已完成[单batch训练函数](train_one_batch.py)，本轮固定为真实SmolVLA的接口；不要求再完成一套CPU对照。
以下仅保留最初的接口骨架，实际执行以.py文件为准：

```python
def train_one_batch(policy, batch, optimizer):
    # 由你补全：训练模式、清理旧梯度、前向取loss、反向、更新、返回数值loss。
    # 真实SmolVLA的policy(batch)返回(loss, loss_dict)。
    raise NotImplementedError
```

先选择一种明确的模型接口完成该函数，不同时写一套兼容所有模型的框架。
验收时在清除梯度前检查梯度，比较step前后一个可训练参数，并解释为什么图像不能直接传给loss。
该练习不替换已准备的官方训练器，练习产生的更新/权重需另存，不混入官方5步对照。
助手先审查你的短实现，再提供提示；不提前补全核心训练步骤。

PEFT实现暂时只追踪三个环节即可：get_peft_model按配置选择LoRA实现；inject_adapter查找并替换目标模块、设置可训练参数；
lora.Linear.forward执行原分支与BA分支相加。无需先掌握其他适配方法、分布式或量化分支。
完成真实训练闭环后进入Corrector：先核对特征/动作时间配对和预测目标，保留固定窗口基线，
按“特征预测→偏离监控→事件触发重规划”逐段实现。OGG单独安排，不把子机制称作完整方法复现。

## 1．先把“插入”理解为一次模块替换

PEFT指参数高效微调这类方法，也是Hugging Face同名库；LoRA是其中一种具体方法。
本次不改变SmolVLA的Flow Matching监督目标，而改变允许优化的参数。

取当前Action Expert中常见的q_proj：每个token的720维特征变成960维Q特征，
原权重W0形状为[960,720]。公式将单个token写成列向量：

`原模块：y = W0 x`

`LoRA模块：y = W0 x + (alpha/r) B(Ax)`

选r=8、alpha=8时：

| 对象 | Shape     | 是否更新       |
| ---- | --------- | -------------- |
| W0   | [960,720] | 原权重冻结     |
| A    | [8,720]   | 新增可训练参数 |
| B    | [960,8]   | 新增可训练参数 |

对整个输入[1,50,720]，原分支输出[1,50,960]，新增分支先变成[1,50,8]再变成[1,50,960]，最后逐元素相加。
输入输出维度不变，没有插入新的token，也没有把动作多执行一遍；这里的数还是内部特征，不是关节命令。

原W0有691200个参数；新增A/B只有8×720+960×8=13440个，约为原矩阵参数量的1.94%。
这不是整个SmolVLA的可训练参数比例，完整模型还在其他目标层插入adapter，需实际统计。

`get_peft_model`按target_modules匹配模块名，在原q_proj的位置放入包装后的LoRA模块。
包装模块保存原来的base_layer，并新增lora_A、lora_B等子模块；不会把原训练器替换成一种新训练算法。
原权重不动，有效映射W0+(alpha/r)BA却能改变。

## 2．如何做到区别对待

LeRobot的wrap_with_peft先把原参数requires_grad设为False，再由PEFT创建可训练A/B。
CPU实践的optimizer只接收requires_grad=True的参数。完整训练还需核对实际参数名，不能仅凭CLI写了LORA就认定正确。

- requires_grad控制是否为该参数求梯度。
- backward计算梯度，不直接更新参数。
- optimizer.step更新优化器管理且具有有效梯度的参数。
- zero_grad清理旧梯度，避免下一个batch意外累积。
- train()/eval()影响Dropout等行为，不是冻结/解冻参数的开关。

冻结W0不等于整段forward使用no_grad。可训练输入经过冻结层时，仍可能需要对输入求导。
CPU实践中o_proj被冻结，但loss必须经过o_proj把梯度传回q_proj的A/B；把整个前向放入no_grad会切断这条路径。

默认A随机、B为0，所以初始BA=0，模块最初与原映射一致。第一步B可能获得非零梯度，A梯度则可能为0；之后B非零，A也开始获得梯度。
本次CPU机制实验显式使用weight_decay=0，隔离梯度更新。不要把零梯度与冻结混为一谈：实际AdamW若开启权重衰减，可训练参数即使当前梯度为零也可能变化。

## 3．真实执行的CPU机制证据

使用官方PEFT、与q_proj对应的720→960维投影，以及一个冻结的960→720投影。
输入是合成隐藏特征[1,50,720]，不是机器人示范；因此这里验证LoRA机制，不报告机器人能力。
证据在本地 `outputs/training_workbench/cpu_clean_001/cpu_lora/evidence.json`。

| 记录                       | 实测           |
| -------------------------- | -------------- |
| 可训练参数                 | 13440          |
| 第一步A梯度范数            | 0              |
| 第一步B梯度范数            | 0.001051785    |
| 第二步A梯度范数            | 0.000181350    |
| 原q_proj/o_proj权重        | 逐元素保持不变 |
| 训练5步后的adapter重载误差 | 0              |
| optimizer恢复后再更新一步  | 已完成         |

训练后，第一个token的第一维Q特征实际为：

`原分支 0.139649 + LoRA分支 0.010235 ≈ 模块输出 0.149884`

三组数由真实CPU计算记录；它们不是SmolVLA处理真实图像后的Q，也不是最终动作。
本机制实验的loss约从0.010043变为0.009689，不能把它当作正式SmolVLA的Flow Matching训练loss或任务效果。
完整GPU训练保持原来的速度监督loss，只将其梯度用于adapter更新。

## 4．统一入口与单单元格准备

主脚本：[smolvla_training_workbench.py](../../scripts/smolvla_training_workbench.py)。
单单元格Notebook：[smolvla_training_prepare.ipynb](../../scripts/smolvla_training_prepare.ipynb)。
Notebook自带脚本内容，在远程写出工作文件，因此不依赖Colab已同步本地新文件。
它由主脚本export-notebook生成，代码修改后应重新导出；测试会核对单元格内的源码与主脚本一致。

准备单元格顶部只需确认WORKDIR、PROFILE和MOUNT_DRIVE：

- Colab：默认/content/smolvla-training、gpu；可挂载Drive并持久化准备日志，首次需要Google授权。
- AutoDL：工作目录改为自己的持久化目录，已有匹配CUDA的PyTorch；非Colab不调用Drive。
- 固定依赖组合需要Python3.12/3.13及PyTorch2.7–2.11；不会安装驱动或替换宿主的大体积Torch。

若VS Code连接不支持Drive授权，可先在网页完成挂载，或设MOUNT_DRIVE=False；后者日志仅在当前服务器，不能宣称已云盘备份。
单元格执行准备，不自动训练。训练依赖在独立venv中安装，只链接现有Torch/native wheel，不继承宿主可选diffusers/torchao。
训练数据使用图像列/PyAV，没有准备完整LIBERO仿真环境；以后做环境成功率评测再准备仿真依赖。
本轮安装日志实际包含间接依赖TorchCodec0.11.1；过滤显式dataset列表不能保证其不被extras间接引入，实际安装以freeze为准。
独立环境仍依赖提供Torch的原环境，不能随意移动或删除该原环境。

CPU准备（本机已完成；以下是复现入口）：

```bash
python scripts/smolvla_training_workbench.py prepare --profile cpu
outputs/training_workbench/venv/bin/python scripts/smolvla_training_workbench.py cpu \
  --workdir outputs/training_workbench/my_cpu_001
```

第一条需从已有Python3.12/3.13且能导入Torch的环境启动。第二条只做CPU机制实验，新workdir防止覆盖旧结果。
原ws_lerobot环境未装入新PEFT依赖，本轮已另行核对。
本地调试可选择outputs/training_workbench/venv/bin/python作为VS Code解释器。
远程Notebook内核仍是原解释器，训练由新venv的Python子进程运行；不要用原内核的import结果判断新环境是否装好。

## 5．GPU显式分两步：看batch，再更新

准备单元格成功后，远程文件和解释器分别为WORKDIR/smolvla_training_workbench.py、WORKDIR/venv/bin/python。
终端使用下列入口；Notebook中可用subprocess参数列表运行同样命令。

```bash
/content/smolvla-training/venv/bin/python /content/smolvla-training/smolvla_training_workbench.py train \
  --workdir /content/smolvla-training --inspect-only
```

它走官方数据/处理器/策略创建流程，保存第一个处理后batch、两张图像、参数名/数量、dtype和数值摘要，
在backward前主动结束。用户先核对相机、语言、state/action/padding，随后再启动真正更新：

```bash
/content/smolvla-training/venv/bin/python /content/smolvla-training/smolvla_training_workbench.py train \
  --workdir /content/smolvla-training \
  --archive-dir /content/drive/MyDrive/robot-lab/experiments/smolvla_peft_20261103_001 \
  --log-root /content/drive/MyDrive/robot-lab/experiments/smolvla_peft_20261103_001/train_logs
```

Drive路径使用前须确认已经挂载；普通服务器可改为自己的持久化目录。
在Notebook中应以subprocess列表调用，不把这段终端命令当普通Python直接粘贴。
脚本复用官方训练器5步，第一步记录处理后batch；在backward之后、zero_grad之前审计梯度，并检查参数变化和峰值显存。
保存时先调用官方save_checkpoint，再复制完整step目录，逐文件SHA256校验，完整后写COMPLETE.json。
已有不同内容拒绝覆盖；部分拷贝保留.partial，不能当成功checkpoint。

训练结束还实现了固定batch/初始噪声的adapter重载比较和保存的postprocessor加载比较；
为限制GPU占用，比较前释放第一份模型再加载第二份。这部分当前只完成代码及CPU合同检查，尚无真实GPU通过记录。
当前入口不一键GPU续训到20步，且拒绝复用已有train_run；后续G3应走官方resume并单独核验optimizer/RNG/步数。
CPU实践已验证optimizer重载并再更新一步，不把它冒充GPU策略的续训结果。

所有模型、数据和依赖实际版本记录在environment.json与pip-freeze.txt；VLM首次从main解析的实际SHA也被记录，重复准备复用已有快照。
base和数据集使用固定SHA；后续跨机器复现还需按记录准备VLM快照，不能把未来main变化视为相同实验。

## 6．出错时与个人实践重点

每次运行有独立日志目录：run.log、console.log、status.json；安装另外有pip报告和完整freeze。
命令失败会保留输出与traceback，不只留下CalledProcessError；可用--log-root将日志直接放持久化目录。
CLI子进程会清除继承的PYTHONPATH再启动，避免本机ROS路径或Notebook临时源码路径混入；不修改父终端或Notebook的环境。
突然进程终止仍可能来不及写最终状态；只有完成校验的归档才能用于恢复。

用户先只看cpu_lab中的get_peft_model、optimizer构造、loss.backward和optimizer.step四处。
亲手把可训练参数的sum表达式改成for循环，列出参数名/shape/ requires_grad，并验证得到13440；不需要先理解全部安装代码。
这一短任务服务于检查真实库的模块和训练状态。完成一次机制理解后就转向真实batch，不无限扩展小模型实验。

脚本运行于子进程时，policy对象不会自动出现在主Notebook内核中；需在函数内检查，或读取保存的证据文件。
不要在一个尚未创建policy的Notebook单元直接print(policy)。

当前验证：CPU独立环境安装、依赖检查、重复准备、官方PEFT机制训练及重载已实测；
16项CPU检查覆盖梯度采样时机、NaN loss阻断、归档校验/防覆盖、假Drive路径拒绝、日志、无pip环境修复、
Notebook用户单元格保留、显式单步函数调用、固定SHA离线配置读取与物理数据分片验证。真实样本另用官方读取器实测。
宿主pip向无pip环境安装的分支已单独本地实测；T4准备已经重跑通过，完整策略训练仍未验证。

参考：[LoRA论文](https://arxiv.org/abs/2106.09685)、[PEFT文档](https://huggingface.co/docs/peft/package_reference/lora)、
[LeRobot PEFT](https://huggingface.co/docs/lerobot/peft_training)。
