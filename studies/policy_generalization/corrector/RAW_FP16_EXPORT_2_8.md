# Block 2 + Block 8：先做小范围未经池化的FP16特征提取

## 决策

- 固定零基索引2、8两层，保留各相机原始 `1024×768` Token。
- **不空间池化、不做INT8量化、不重复保存原Connector或动作。**
- 继续引用原`object10/features/episode_N/`的深层特征、归一化动作、帧序号及时间戳。
- 仅对已有真实示范在相同episode内自动生成监督目标 `F[t+10]-F[t]`，无需人工标注关键帧。
- 这是一组覆盖不同深度的初始实验选择，不是声称2和8已被证明最优。
- 旧接口已有155帧一致性核验：重新量化Connector的Q与Scale均为100%逐元素一致。

## 磁盘估算

一个FP16中间层：两相机 × 1024 token × 768维 × 2 bytes = 3 MiB/帧。
两层总计约6 MiB/帧，所以155帧约 **930 MiB**（不含.npy小头与报告），全66,984帧两层约 **392.5 GiB**。前面误写过“155帧两层465MiB”，那是**单层**容量，已纠正。

系统盘之前约20GiB可用，数据盘约8.1GiB可用，**不要把两个文件系统的剩余容量简单相加视为单个目录**。这一批先明确写到系统盘独立目录，并核查实例停止/释放后的持久性策略；重要结果建议另外备份。

## 4090运行说明

```bash
cd /root/autodl-tmp/corrector
git status -sb
git pull --ff-only
runtime/venv/bin/python -m unittest discover -s tests -p 'test_corrector_multilayer_pilot.py' -v
df -h / /root/autodl-tmp
nvidia-smi
```

此时测试数预期是**9项**（其中含新增导出选项CLI测试）。若有未提交修改或测试失败，停止，不强制重置Git，也不修改旧冻结目录。

只导出已确认存在原始Parquet和旧深层缓存的Episode813：

```bash
runtime/venv/bin/python scripts/corrector_multilayer_pilot.py \
  --episodes 813 \
  --layers 2 8 \
  --grids 8 \
  --save-fp16 \
  --output /root/multilayer_export_813_raw2_8_001

echo "EXIT_CODE=$?"
```

注意`--grids 8`仅控制**额外生成的压缩误差报告**；由于指定了`--save-fp16`，实际保存的数据始终为**未池化的32×32、FP16**。

目录结构：

```text
/root/multilayer_export_813_raw2_8_001/
├── report.json
└── episode_813/
    ├── block_02.fp16.npy     # [155,2,1024,768]，约465MiB
    ├── block_08.fp16.npy     # [155,2,1024,768]，约465MiB
    └── COMPLETE.json         # SHA、冻结权重、相机顺序、旧缓存引用
```

仅当`COMPLETE.json`存在并且SHA检查通过时，缓存才可被视作可用。脚本不覆盖旧输出、旧Connector缓存或Checkpoint；中途失败保留未完成目录以供排查，不会擅自删除任何文件。每次重新运行须指定新的输出路径。

检查结果：

```bash
runtime/venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
import numpy as np

root=Path('/root/multilayer_export_813_raw2_8_001')
meta=json.loads((root/'episode_813/COMPLETE.json').read_text())
print('episode:',meta['episode'],'frames:',meta['num_frames'])
print('pooling:',meta['pooling'],'dtype:',meta['dtype'])
print('old deep cache:',meta['deep_cache_path'])
for name, rec in sorted(meta['files'].items()):
    path=root/'episode_813'/name
    array=np.load(path, mmap_mode='r', allow_pickle=False)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    print(name,'shape:',array.shape,'dtype:',array.dtype,
          'GiB:',round(path.stat().st_size/2**30,3),
          'sha_ok:',h.hexdigest()==rec['sha256'])
print('report exists:',(root/'report.json').is_file())
PY
```

## 之后怎么扩大

原数据分片只确认Episode813对应`file-213.parquet`存在；**不要假定818/825的Parquet都在**。在检查来源存在后，可以选择来自Train/Validation的2～3个完整Episode，利用相同`--save-fp16 --layers 2 8`入口导出到另一个系统盘独立目录。

后续训练：同一Episode的`F_t^{(2)}`、`F_t^{(8)}`来自新增缓存，`Z_t^{deep}`与`A_{t:t+k}`来自旧缓存；监督目标取同一Episode的对应未来帧差，必须验证frame/timestamp连续且不跨Episode，保留训练/验证划分。构建**一个共享主干、按层输出的联合预测器**，避免把层数改成三个独立网络。

实验限制：目前这个增量只是**特征导出**，还没有真实运行此新增保存路径，也没有训练多层Corrector；请先检查磁盘、测试并跑Episode813，再逐步扩大。
