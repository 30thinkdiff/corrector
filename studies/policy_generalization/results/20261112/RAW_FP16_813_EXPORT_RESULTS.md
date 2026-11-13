# 2026-11-12 Episode813 Block2+Block8 无池化FP16特征导出：4090结果

> 数据来自用户粘贴的4090终端记录；原始FP16张量和report.json仅在远程系统盘，未上传到GitHub。本报告不代替对原始字节的独立复核。

## 已执行并通过

- 本分支fast-forward更新至`54585de`后，`test_corrector_multilayer_pilot.py` 9个CPU测试全部通过，`Ran 9 tests in 0.056s; OK`。
- `--episodes 813 --layers 2 8 --grids 8 --max-frames 21 --save-fp16 --output /root/multilayer_export_813_21f_001`：小样本成功，21帧、11个1秒配对，生成report。
- `--episodes 813 --layers 2 8 --grids 8 --save-fp16 --output /root/multilayer_export_813_full_001`：完整Episode成功，155帧、145个1秒配对，生成report。
- 完整导出目录：`/root/multilayer_export_813_full_001/episode_813/`，分别有`block_02.fp16.npy`、`block_08.fp16.npy`，都为shape `[155,2,1024,768]`、`float16`、各约465MiB。
- 用户实际执行SHA256验证，两文件均打印`SHA OK: True`。
- 终端摘要`deep_median_relative_l2 = 0.028644966638388634`；此前同一Episode155帧筛选中，重新量化的旧Connector Q/scale 155/155完全一致。
- 文件存储为**未池化的原始网格Token**；`--grids 8`仅作用于报告，不改变保存内容。

## 精确边界

1. 目前导出的一条Episode属于任务24（train），尚未形成多样本训练/验证数据组合。
2. 旧深层Connector缓存`object10/features/episode_813`由既有训练使用；新文件只含中间层，动作、时间戳与训练标签按episode/frame索引关联，未来监督`F[t+10]-F[t]`。
3. 现有压缩/完整性报告不能证明多层神经网络有预测收益；模型仍未训练。
4. 系统盘和数据盘是不同文件系统；系统盘上产物的关机/释放后保留规则尚未确认。对重要产物保持独立备份。
5. 当前实测约930MiB/155帧（两层）；原raw未全量导出，不要再引用之前误算的465MiB/两层。

## 下一步优先采用原始分片file-213（可复用已存在的98MB Parquet）

根据固定`object10/manifest.json`，以下Episode均来自`data/chunk-000/file-213.parquet`：

| Episode | task_id | split | frames |
|---|---:|---|---:|
| 813 | 24 | train | 155（已导出） |
| 814 | 25 | train | 175 |
| 815 | 21 | validation | 152 |
| 816 | 26 | train | 124 |
| 817 | 25 | train | 143 |

在远程确认其旧Connector缓存存在与系统盘余量后，可只导出814/815/816/817，预计新增3564 MiB（两层、共594帧），无需重算813，也不需重新下载这些Episode的原始Parquet。

这些Episode可用于**训练管线端到端烟雾测试和非常初步的学习曲线**，不能凭一个validation Episode（并且其Task 21未在train集合出现）就推断跨任务泛化效果。后续再增加同任务不同Episode的独立验证组，避免只靠单一拆分做论文式统计结论。
