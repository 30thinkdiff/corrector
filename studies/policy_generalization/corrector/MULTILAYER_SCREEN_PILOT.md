# 多层SigLIP真实小样本筛选（不训练、不下载、不重算深层缓存）

开发分支：feature/multilayer-feature-screening-20261112

## 已知事实与设计边界

- 原Object10共454条示范、66,984帧，10FPS；旧的Connector深层缓存在 `object10/features/episode_NNN/`，保存 `z_q.npy`、`z_scale.npy`、`frame_index.npy`、`timestamp.npy`、`actions.npy` 和SHA完成标记。
- `scripts/corrector_multilayer.py` 移植自 robot-lab 已经完成三帧实机GPU探针的只读Hook；之前验证Block 2/5/11每路[1024,768]、Connector每路[64,960]，18次Hook前后Connector输出完全相等。
- **真实4090首轮已完成**：Episode813的21帧，11个1秒变化对，Block2/5/8的8×8池化变化重建误差P90依次为0.89849/0.88689/0.83771；对应FP16误差为0.001007/0.000630/0.000581，INT8误差为0.11404/0.09928/0.07803。报告位于远程`object10/multilayer_screen_813_21f_001/report.json`；不上传大文件。
- 我们的最终目标是**一个联合多层动作条件视觉变化预测器**；本阶段只是选择层级/压缩协议，完全不训练独立中层模型。

## 实际操作（在4090容器里执行）

先确认工作区干净，代理/网络可用时获取实验分支：

```bash
cd /root/autodl-tmp/corrector
git status -sb
git fetch origin
git switch feature/multilayer-feature-screening-20261112
git pull --ff-only
runtime/venv/bin/python -m unittest discover -s tests -p 'test_corrector_multilayer_pilot.py' -v
```

不要强制checkout、reset或删除旧目录。分支中已经加入16×16网格选项及严格逐元素重新量化核验，拉取后预期8项单元测试通过。

检查真实数据源是否还在磁盘：

```bash
ls -lh object10-work/source/data/chunk-000/file-213.parquet
ls -lh object10/features/episode_813/{z_q.npy,z_scale.npy,actions.npy,metadata.json,COMPLETE.json}
df -h /root/autodl-tmp
nvidia-smi
```

若Parquet缺失，**不要凭空生成RGB、不要自动重新下载全库**。先查找有没有其他已有原始归档或可用分片，再决定是否需要小体积单分片下载。

21帧小探针已成功完成。若要用新增的16×16指标重新测21帧，使用新的输出目录：

```bash
runtime/venv/bin/python scripts/corrector_multilayer_pilot.py \
  --episodes 813 --layers 2 5 8 --max-frames 21 \
  --output object10/multilayer_screen_813_21f_with16_001
```

下一步可先直接对Episode813全部155帧执行（新输出，不覆盖旧探针，包含16×16试验）：

```bash
runtime/venv/bin/python scripts/corrector_multilayer_pilot.py \
  --episodes 813 --layers 2 5 8 \
  --output object10/multilayer_screen_813_full_001
```

如果需要多个不同任务的比较，先在 `object10/manifest.json` 选出对应物理Parquet已经存在的2-3个不同 `task_id` 的完整Episode；必须保留训练/验证来源协议。示例命令的默认813属于同一任务，不能当作Object10全面结论。

仅看核心统计结果：

```bash
python3 - <<'PY'
import json
p="object10/multilayer_screen_813_full_001/report.json"
d=json.load(open(p))
for ep in d["episodes"]:
    print("episode",ep["episode"],"frames",ep["frames"],"pairs",ep["pairs"])
    print("deep cache relative L2 median/max",ep["old_deep_median_relative_l2"],ep["old_deep_max_relative_l2"])
    for l,grids in ep["layer_metrics"].items():
        for g,m in grids.items():
            print(l, "grid",g, "pool Δerror p90",m["pooled_vs_full_delta"]["p90_relative_l2"],
                  "FP16 Δerror p90",m["fp16_delta"]["p90_relative_l2"],
                  "INT8 Δerror p90",m["int8_delta"]["p90_relative_l2"])
PY
```

把报告JSON和标准输出发回ChatGPT。JSON不包含图片、模型、个人凭据；仍应注意检查其元数据。

## 判断原则

1. 首先确认旧缓存与新Connector兼容，检查`old_deep_exact_requantized_frames`、`old_deep_q_element_agreement_fraction`、`old_deep_scale_element_agreement_fraction`。上一轮21帧最大相对L2为0.02964，但仍可能来自量化。门限是**宽松的工程冒烟**（最大relative L2不大于0.06且最小cosine不小于0.997），不是科学最优阈值。
2. `pooled_vs_full_delta`：将4×4/8×8/16×16池化后的Δ通过最近邻扩展回32×32，对照全分辨率Δ；**度量局部动态信息的空间损失，不是预测器准确率**。应同时考虑p90及不同物体任务。
3. `fp16_delta` 与 `int8_delta`：同一固定网格中，存储压缩引入的Δ误差。**必须看Δ而不仅看单帧误差**。量化协议为逐Token INT8 + FP16 scale。
4. 小范围实验只能淘汰明显劣质的候选协议，**不能证明第2/5/8层哪层使LVM/OGG更准确**。选择最终层级仍需要冻结训练/验证Episode的预测基准和配对闭环。
5. 确定候选层后，另开发独立缓存目录例如 `object10/multilayer_features_v1`，按Episode/帧索引与旧Connector缓存对齐；不得向现有 `object10/features` 目录写新字段或覆盖SHA标记。

## 磁盘粗算（66,984帧、双相机、768维，每个中间层）

- 8×8 FP16约12.26GiB/层；8×8 INT8+FP16 scale约6.15GiB/层。
- 4×4 FP16约3.07GiB/层；4×4 INT8+FP16 scale约1.54GiB/层。
- 16×16 FP16约49.06GiB/层；16×16 INT8+FP16 scale约24.56GiB/层。仅适合当前不落盘的小样本诊断，不能直接全量缓存。
- 这是**每层**的未压缩二进制估算，尚未包含元数据、文件头、临时归档和训练快照；已有深层缓存不重复计算。原数据盘剩余约8GiB，不能直接承诺能装两个8×8层。
- 抽样检验后再决定外部持久存储、扩容、边提取边训练与最终缓存布局。禁止自动清理旧的训练结果和best/latest权重。

## 当前能证明与不能证明的内容

- 本地CPU单元测试：Hooks不改变假模型输出，测试异常清理；池化/量化数值与形状；只读报告不覆盖。
- 已有真实三帧验证：Hook的正确性与小型特征缓存规格。
- **已有21帧真实4090结果**，上述数值均为Episode813的11对变化统计；完整155帧、其他任务、训练/闭环收益尚未验证。不能单靠池化误差排序确定最优层级。
