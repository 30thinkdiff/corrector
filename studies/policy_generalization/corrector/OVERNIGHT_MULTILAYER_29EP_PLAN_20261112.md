# 2026-11-12 当晚执行：29条Episode / 联合多层Corrector A-B-C

## 目标与边界

用户希望在当晚尽量完成数据准备、训练、离线测试。**绝不直接扩大454条**。该批次使用固定`object10/manifest.json`（数据修订`86958911c0f959db2bbbdb107eb3e17c5f9c798e`），选`file-212.parquet`到`file-217.parquet`共6个原始分片（其中file-213已验证在远程）。真实`721f2bae4c3cf2c9ef7abaf548caf40ee9d9f92d`训练MVP源代码目前只确认部署在远程`dev/multilayer-training-mvp-a98b5be63190`和用户的本地Codex工作区；**不要假定这个源提交已经存在于GitHub分支**，先核对真实本地版本。

29条Episode、4195帧、22条训练3150帧、7条验证1045帧；固定k=10、10Hz，训练配对2930，验证975，总计3905。验证来自多个task_id，但同任务各个episode之间仍有相关性；按Episode聚合指标并报告各TaskID，不做假独立性结论。

## 为什么先用16×16 FP16

- 原始每层、每帧：[2 cameras,1024 tokens,768],FP16；两个层合计6MiB/帧；4195帧约24.58GiB，无法在安全余量下全量落盘。
- 空间均值池化成[2,256,768]，**16×16 FP16双层共1.5MiB/帧**；4195帧约6.145GiB，留原Block2/8未池化数据不删。这是为了今晚训练的**明示有损工程折中**，不是宣称已证明最优；155帧试验的16×16相对L2空间重建P90为Block2约0.771、Block8约0.697。
- 已提取的5条813–817拥有未池化原始FP16（749帧、4.39GiB）。用CPU只读对这些原特征做2×2均值池化，发布独立16×16缓存，无需再跑视觉模型；其余24条可从原始Parquet经冻结SmolVLA一次提取后**在内存中池化为16×16，再落盘**。两种缓存均用统一FP32均值→FP16写入、相同camera顺序与Patch排列，并抽样验证差值。
- 新增16×16缓存共6.145GiB，系统盘用户报告有16GiB可用，预留**不小于8GiB**；原始Parquet仅约0.5GiB新增（若需下载），数据盘目前8.1GiB，预留不小于5GiB。多轮训练复用该缓存，**不要每epoch重新编码，不要训练中删除已存特征**。
- 如系统盘预检后预测余额不足8GiB，应按完整Episode边界缩小范围，例如从6个分片缩为212–215并保留原完整raw；**绝不能挤到满盘**。系统盘overlay的数据持久化机制尚未确认，模型Checkpoint/结果同步到持久数据盘前要做SHA、剩余空间审查。

## 固定29条Episode清单

| Parquet | Episodes (按固定manifest，具体split以manifest为准) |
|---|---|
| file-212 | 808,809,810,811,812 |
| file-213 | 813,814,815,816,817 |
| file-214 | 818,819,820,821,822 |
| file-215 | 823,824,825,826 |
| file-216 | 827,828,829,830,831 |
| file-217 | 832,833,834,835,836 |

7条验证Episode：**811、815、819、824、825、829、834**；其余22条训练。至少Task21、23、24、27、20、29存在train/validation的任务重合，但Task22、25、26、28在这29条中无相应验证任务。不可将此拆分称为任务全集全面评测。

## 本地Ubuntu Codex任务（代码、测试、打包）

1. **先定位此前完成4090真MVP的原始本地工作树/zip/源码SHA 721f2bae...；勿用仅有旧pilot的GitHub分支覆盖真MVP。** 新增night实验目录，保留旧冻结代码、三方环境和旧数据不改。制备ZIP+SHA256和远程部署命令。
2. 写`export_pooled_multilayer.py`：支持清单驱动逐Episode、`--layers 2 8 --grid 16 --dtype float16`、`--existing-raw-roots`、`--data-root`、`--old-deep-root`、`--output-root`、`--min-system-free-gib 8 --min-data-free-gib 5`、STOP与按Episode安全恢复。缓存文件临时名写入/flush/校验SHA后最后原子发布COMPLETE；已完成Episode重新校验并跳过；半成品绝不当成完成。零下载默认；如源Parquet缺失只允许按固定revision下载**必要分片**并核对大小/哈希后继续，不得拉完整Object10。
3. 混合路径：五条原有raw FP16从已存在两个export根目录按需池化；其它24条从Parquet前向一次提取、池化后直接保存16×16；保留原raw FP16不删。Hook/冻结VLM权重严格复用已验证版本；核对原Connector的逐Token量化一致性、相机顺序、Episode frame/timestamp、原缓存SHA。单位episode训练/验证标志沿用manifest。
4. Dataset lazy mmap，避免拷贝进RAM。对16×16层的监督目标用`F[t+10]-F[t]`，旧Connector目标依然按旧Q/scale反量化差分。确保没有未来观测输入或未来真实动作条件泄漏。Frozen 原Connector及动作、索引文件只读引用数据盘。
5. 训练配置：A Connector-only；B 16×16两层输入+三个输出头（共享主干、联合损失）；C与B相同输入/主干但只计算Connector监督，区别于三层辅助目标。保留参数量、层内loss权重、训练集估计的RMS标准化、seed42，输出每epoch Connector原始MSE/归一MSE、zero baseline和余弦，B的中层指标；至少15轮预算上限、早停（如patience5，具体经训练曲线确认）并存最佳。因为原5轮两模型Connector val仍下降，不能以5轮为充分收敛。**对照同样的29条Episode、split、k10、动作条件，训练epoch及optimizer updates明确记录**；若无法严格控制参数量，明确为非同参对照。
6. 强制GPU smoke：独立部署后preflight SHA全部通过；单batch forward+backward各头有梯度；可逆备份、保存/重载一致；检查显存和系统盘余量；再训练A/B/C，不启动机器人闭环。执行完输出 per-epoch CSV/JSON、每TaskID和每Episode验证指标，参数量、训练时间、最佳epoch、GPU mem；把小型结果压缩包（不含原始features和权重）回传本地。
7. 自检：层特征shape、缓存SHA损坏拒绝、不同Episode不跨界、时基一致、train-val分离、模型不读取未来帧、源权重不变、GPU空闲、输出不覆盖旧任务；所有明显异常暂停。CPU+GPU测试分别报告真实通过/失败，不能编造。
8. 监督脚本阶段按`preflight → pooled-export → verify29 → A → B → C → summarize`；每步独立status、明确错误和退出码，SIGINT/STOP在Episode或Epoch边界安全退出，tmux可观察；**不清空缓存、不删除raw Parquet、不动best/latest，不自动扩容购买**。

## 可只读检查磁盘现状（远程）

```bash
df -h / /root/autodl-tmp
du -xhd1 /root 2>/dev/null | sort -h | tail -n 20
du -xhd1 /root/autodl-tmp/corrector 2>/dev/null | sort -h | tail -n 25
du -sh /root/multilayer_export* /root/multilayer_training_runs 2>/dev/null
for id in 212 213 214 215 216 217; do
  p=/root/autodl-tmp/corrector/object10-work/source/data/chunk-000/file-${id}.parquet
  if test -f "$p"; then ls -lh "$p"; else echo "MISSING $p"; fi
done
nvidia-smi
```

## 清理策略

默认零删除；所有路径先列清单、人审和SHA核验。可优先关注可再生的重复实验小型中间输出、已验证的重复缓存/过期下载分片，但**不能**仅凭目录叫`tmp`、`cache`或`checkpoint`就删除。不要删`/root/corrector_supervision_36h/checkpoint_archive`：部分latest可能有指向系统盘归档的符号链接。不要删除五条原始全分辨率FP16（研究上独一份，系统盘持久性未知）、冻结model weights、Object10 Connector缓存、评测视频或旧结果。为数据盘保留≥5GiB、系统盘保留≥8GiB的运行边界。**滚动缓存是逐Episode产出/复用，而不是多轮训练期间删除已经用于训练的数据。**

## 晚间顺序与止损

- 首选29条Episode、16×16 FP16。没有足够分片、网络不稳或预计低于余量时，缩为20条左右并报告具体选择；最低保留多个任务+train/validation。
- 不重复抽取已有深层Connector，不能将原raw全量32×32重新提取。
- 如果preflight或smoke失败，停止正式训练；不要热修改冻结模型后冒充同协议续训。
- 只有离线loss的提高/降低，没有任何OGG/LVM在线收益或闭环成功率证据；下一阶段需再验证相同物理episode分布之外的稳定性。
