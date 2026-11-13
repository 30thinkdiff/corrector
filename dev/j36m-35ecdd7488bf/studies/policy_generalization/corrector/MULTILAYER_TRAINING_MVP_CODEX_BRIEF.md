# 统一多层Corrector：本地Codex开发 / 远程4090小规模训练任务书

> **历史MVP任务书（2026-11-12）**：本文件用于解释已完成的5条Episode、0.767M联合工程模型及其部署验收，不再作为正式J-36M模型训练或在线实验的当前需求。新任务须以[正式J-36M架构与LVM/OGG消融规范](FORMAL_MULTILAYER_CORRECTOR_J36M_DESIGN_20261112.md)及[Codex新执行任务书](J36M_LVM_OGG_CODEX_TASK_20261112.md)为准。旧结果不可覆盖。


日期：2026-11-12。研究目标：在已有深层Connector Corrector基线基础上，以**一个联合模型**利用冻结SigLIP中间层Block 2及8，并以未来视觉变化为监督目标。此任务是MVP、不是大规模实验。

## 0. 已完成、不得重做

真实4090已验证：`--save-fp16 --layers 2 8`，Episode 813、814、815、816、817，均已导出未池化双相机视觉Token `[T,2,1024,768]`、dtype float16；Episode 813两文件及其SHA已由用户检验，814~817的文件形状、dtype与生成标记存在已检查，后续Loader须重新核验文件SHA。原Connector缓存 `object10/features/episode_N` 具有 `z_q.npy` (int8, `[T,128,960]`)、`z_scale.npy` (float16,`[T,128]`)、`actions.npy` (`[T,7]`，冻结checkpoint processor归一化)、`frame_index.npy`、`timestamp.npy`、`metadata.json`和SHA完成标记；均复用，不重复视觉提取。

| Episode | Task ID | Split | Frames | k=10 pairs | 原始FP16目录 |
|---|---:|---|---:|---:|---|
| 813 | 24 | train | 155 | 145 | `/root/multilayer_export_813_full_001/episode_813` |
| 814 | 25 | train | 175 | 165 | `/root/multilayer_export_file213_rest_001/episode_814` |
| 815 | 21 | validation | 152 | 142 | `/root/multilayer_export_file213_rest_001/episode_815` |
| 816 | 26 | train | 124 | 114 | `/root/multilayer_export_file213_rest_001/episode_816` |
| 817 | 25 | train | 143 | 133 | `/root/multilayer_export_file213_rest_001/episode_817` |

**训练557对；验证142对；5条示范共749帧。** 验证仅一条且Task21不在当前训练组出现，只能用于端到端能力测试，**不可声称统计显著的泛化提升**。相邻k10窗口高度重叠。

资源：4090 24GB，数据盘`/root/autodl-tmp`剩余约8.1GiB，系统盘`/`最新剩余约16GiB。大特征留系统盘，不复制到数据盘。冻结Checkpoint、历史结果和源码目录不得修改。系统盘文件持久性仍需核对并做备份。

## 1. 开发与部署工作流

Codex运行于**用户本地Ubuntu22.04**，不在远程4090上使用Codex。使用本地`corrector`仓库（若无则clone `30thinkdiff/corrector`），`git fetch`并检出 `feature/multilayer-feature-screening-20261112` 的最新版本，在其上创建**新的MVP开发分支**。本地实现/测试后输出源码压缩包、SHA256、部署命令；由用户上传到远程并在独立目录运行。若GitHub网络正常，也可通过GitHub分支+远程git fetch部署。**不要依赖上传包含数据/权重/venv的完整仓库压缩包。**

本地不能假装运行了真实CUDA训练。任何GPU数量、loss或验证结果必须由远程日志证实。

## 2. 交付1：数据集

实现 `MultiLayerTransitionDataset`，输入明确给出的`{episode_id: raw_intermediate_path}`配置和原特征根目录。禁止把目前`scripts/corrector_data_workbench.py`仅有3条EPISODES白名单用于新训练。至少：

- 验证`COMPLETE.json`、各层SHA256、冻结权重版本、层索引、dtype和shape；核对原缓存`metadata.json`、帧数、episode id、fps=10、双相机顺序、frame_index连续、timestamp连续；拒绝混用训练和验证及不完整/部分提取的目录。
- `np.load(...,mmap_mode='r',allow_pickle=False)`，训练按需读帧，不把4.39GiB全部载入RAM，也不复制到数据盘；初始`num_workers=0`。
- 对任一合法t（0…T-11），输入仅`F_2(t),F_8(t),Z_deep(t),a(t)`；监督`F_2(t+10)-F_2(t)`、`F_8(t+10)-F_8(t)`、`Z_deep(t+10)-Z_deep(t)`。所有`t+10`特征只用于监督，绝不能泄漏给模型输入。不跨Episode。默认条件只有当前已执行/计划动作一帧，与原单动作Corrector基线对齐；**不得把未来实测动作序列作为本次模型输入**。
- 原Connector反量化=`q.astype(float32)*scale.astype(float32)[...,None]`，动作保持旧processor归一化形式；保存所有输入/目标形状测试。
- 多根目录索引路径写到config中，绝不可假定新增两层位于同一个根目录。

## 3. 交付2：小模型与实验对照

真正**一个联合模型**，包含Block2+Block8+Connector+7维动作，带共享主干和按层变化预测头，不训练三个互不相干的模型。

优先轻量、能够在4090运行的Token-wise projection + 局部/潜在交互结构，不得直接对两路×1024×2层全部Token做不受控的全局`O(N^2)`注意力。尽量保留真实32x32空间结构；**不能默默对输入和监督改为4×4/8×8均值池化**。如硬件限制确需训练时降采样，必须显式标记为消融，并报告实际监督空间；保留原raw FP16数据不变。

输出`delta_block_02`、`delta_block_08`、`delta_connector`，校验对应目标形状。多层损失必须按层先求平均、用训练集估计的尺度归一（禁止从validation估计），避免特征维度/幅值直接决定加权。提供`zero-delta`基线与每层验证MSE/归一MSE/余弦（零向量谨慎处理），特别报告Connector预测误差。

起码两种训练配置：
- A: Connector-only输入，仅预测Connector未来变化（参考/对照），使用同样训练Episode、k10、动作定义、训练步数和seed。
- B: Block2+Block8+Connector联合输入，一个共享主干、三输出头、多层损失。
可选C：多层输入但只训练Connector损失，以区分额外输入特征与辅助监督收益。严格记录模型参数量/训练时间；参数量不同不能将全部改进归因于特征本身。

## 4. 交付3：启动、检查、结果

提供：
- 本地CPU合成fixture测试（shape、时间泄漏、split不交叉、故意篡改SHA必须拒绝、mmap、梯度流、single-batch overfit、保存/重载一致）；
- 远程 **preflight-only** 命令（只读5条Episode，按完成标记SHA复核，磁盘/CPU RAM/GPU资源）；
- 远程GPU smoke（2个batch的前向/反向、loss finite、梯度存在、单batch过拟合）；
- 首轮短训练（小批量、可调整epoch/早停，分别A/B），独立输出根目录，原子写checkpoint并记源码git SHA/固定参数、各Episode列表、帧与时间协议、损失与种子；
- 简洁JSON/CSV结果，用于回传ChatGPT分析，**不上传权重/数据到GitHub**；
- 遇CUDA OOM、磁盘不足、SHA不符、GPU被占用须停止/安全暂停，不得自动删除缓存或继续下载数据。
- 有必要时训练中的feature-crop/gradient checkpointing是显式选择，不能偷偷改变监督目标或A/B比较设置。

先实现并审查代码、创建可部署包与具体命令，再由用户选择执行4090。**本任务禁止全量454 Episode特征提取，禁止大规模训练、自动安装包及任何修改冻结旧模型。**

## 5. 验收

1. CPU回归可复现且无数据泄漏。
2. 5条真实Episode的校验全部通过，包括新raw FP16及旧Connector。
3. GPU smoke前后显存与loss正常，并能真正反向更新联合模型共享部分和各预测头。
4. 短训练输出A/B各自每层可解释指标、zero-baseline、完整checkpoint和恢复记录。
5. 形成下一阶段是否值得扩大数据的客观建议；**暂不要求多层必胜**。
