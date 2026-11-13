# 统一多层Corrector训练MVP

本实现基于`feature/multilayer-feature-screening-20261112`的`58dc922deb7d70b488330c857e41353e93500e78`。独立开发分支：`feature/unified-multilayer-training-mvp-20261112`。原任务书为上级目录的`MULTILAYER_TRAINING_MVP_CODEX_BRIEF.md`。

**已完成本地CPU开发验证；尚未执行真实五轨迹全量SHA预检、4090冒烟或真实训练。** 读取过远程813的真实COMPLETE/metadata以核对格式，没有复制FP16特征、加载VLA或占用GPU。本工具不会下载、提取、安装依赖或修改旧实验。

## 模型与数据

| 对照 | 输入 | 输出/监督 | 默认参数量 |
|---|---|---|---:|
| A Connector-only | 当前Connector、当前归一化7D动作 | Connector未来变化 | 273,120 |
| B joint | 当前Block2、Block8、Connector、同一动作 | 三层未来变化 | 767,328 |

Block索引**从0计数**。模型通过逐Token投影、空间3×3 depthwise卷积、跨流上下文和共享残差MLP主干交互，三个线性头输出变化。中间层保留双相机32×32×768、Connector保留128×960；没有目标池化、裁剪或全局二次复杂度自注意力。跨流上下文使用投影Token全局均值，这是显式的信息瓶颈，但**不会缩小输入文件或输出监督**。不假定Connector Token具有已证明的空间/相机对应关系。

本MVP没有接入在线LVM/OGG，也没有继承旧预测器权重；只复用冻结视觉特征及动作归一化协议。不能将它称为已验证的闭环改进。A与旧38M原生Corrector也不是同一架构，本次A是共同预算下的轻量Connector对照。

固定训练813/814/816/817共557对；验证815共142对，k10、数据10Hz、预测1秒。输入只含`t`的特征和动作；`t+10`只构造delta监督，没有未来实测动作。双相机顺序、冻结政策版本及权重SHA、帧/时间索引、episode和split、原始Parquet来源、raw对Connector COMPLETE的内容绑定全部校验。路径可迁移，内容绑定必须保持；所有数组以只读mmap逐帧访问，不整库载入RAM。

按训练557对流式估计每层`RMS(delta)`；损失为每层元素平均MSE除以该尺度平方，再对层等权平均。尺度不读取验证集。报告每层MSE、归一MSE、zero-delta基线、相对基线比值、逐转移余弦及有效/零向量计数，**不把零向量余弦伪装成0或1**。

默认A/B固定5轮、相同batch/seed/学习率/样本顺序，各1395次更新(batch2，最后不足batch保留)。按两者共同的Connector验证归一MSE选择best。B训练损失包含三个头，A只有Connector，不能直接比较总loss；先比较Connector指标和zero基线。参数量不同；验证仅一条Task21且与训练任务不同、相邻窗口重叠，不能声称显著泛化提升。提供单独fit的早停选项，但`paired`禁止早停以保证相同更新预算。

## 本地CPU验收

依赖仅已有Python、NumPy、PyTorch；不需要transformers/LeRobot/LIBERO/Pillow。Ubuntu下可用已有CPU环境：

```bash
cd /home/hhd/robot-lab/outputs/corrector_repository
/home/hhd/ws_lerobot/.venv/bin/python -m unittest discover -s tests/multilayer_mvp -v
```

测试在临时目录生成微型合成缓存，不是原始实验数据。覆盖完整真实输出shape、共享/各头梯度、过拟合、严格SHA、时序与split防泄漏、只读源文件、断点恢复等；详见`VALIDATION.md`。不得将合成loss当成真实预测质量。微型测试将空间保留阈值降至1MiB，仅为容纳本机/tmp空间；生产仍保留2GiB磁盘及1GiB有效RAM保护。

复用旧`corrector_data_workbench.digest/write_json`、`corrector_train.reference_components/reference_loaders/reference_update/atomic_save`；这些源码作为只读快照包含在部署包中。没有调用旧三轨迹白名单、旧数据切分、练习TODO或旧冻结模型工厂。数据适配/模型/指标/资源模块在`scripts/multilayer_mvp/`，入口`scripts/corrector_multilayer_train.py`。

## 远程完整操作

以下所有GPU操作**由用户执行**。同一远程终端保留变量；重新登录后需重新设置。占位`AUTODL_SSH_HOST`填当前SSH配置中的主机/别名，必要时按控制台设置端口；文档不存凭据。

### 1. 上传及独立部署

在本地Ubuntu执行（实际交付ZIP及checksum位于robot-lab的`outputs/autodl_setup/multilayer_mvp_20261112/`）：

```bash
cd /home/hhd/robot-lab/outputs/autodl_setup/multilayer_mvp_20261112
sha256sum -c corrector_multilayer_mvp_20261112.zip.sha256
scp corrector_multilayer_mvp_20261112.zip corrector_multilayer_mvp_20261112.zip.sha256 package_multilayer_mvp.py AUTODL_SSH_HOST:/root/
```

在远程执行，仅验证和解压，不启动实验：

```bash
cd /root
sha256sum -c corrector_multilayer_mvp_20261112.zip.sha256
MVP_SHA=$(awk '{print $1}' corrector_multilayer_mvp_20261112.zip.sha256)
MVP_CODE="/root/autodl-tmp/corrector/dev/multilayer-training-mvp-${MVP_SHA:0:12}"
python3 /root/package_multilayer_mvp.py install --archive /root/corrector_multilayer_mvp_20261112.zip --sha256 "$MVP_SHA" --destination "$MVP_CODE"
MVP_PY=/root/autodl-tmp/corrector/runtime/venv/bin/python
MVP_CONFIG="$MVP_CODE/studies/policy_generalization/corrector/multilayer_mvp/config.json"
MVP_RUN=/root/multilayer_training_runs/mvp_20261112_001
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

已存在的部署目录会拒绝覆盖；正常复用时无需再次install，直接设置`MVP_CODE`等变量。部署文件清单记录本地源码commit及逐文件SHA。ZIP只含源码、配置、测试、说明和必要旧工具快照，没有4.39GiB特征、模型或环境。所有新输出默认放系统盘`/root/multilayer_training_runs/`；不复制特征到数据盘。系统盘数据持久性和备份仍须自行确认。

### 2. 环境检查、CPU测试、真实数据预检

```bash
"$MVP_PY" -c 'import torch,numpy; print(torch.__version__,numpy.__version__,torch.cuda.is_available())'
cd "$MVP_CODE"
"$MVP_PY" -m unittest discover -s tests/multilayer_mvp -v
nvidia-smi
pgrep -af '[c]orrector_.*\.py|[s]molvla_.*\.py'
"$MVP_PY" "$MVP_CODE/scripts/corrector_multilayer_train.py" preflight --config "$MVP_CONFIG" --output "$MVP_RUN"
```

preflight不创建输出、不创建CUDA上下文；通过nvidia-smi报告GPU/进程，SHA顺序读约4.39GiB，可能需要数十秒至数分钟。期望557/142，五条episode的完成标记及所有列出的文件校验通过。若有旧实验/调度器活动，先停在此步；不要并发。smoke/fit还会拒绝其他GPU计算进程，并检查输出/数据盘空间及cgroup有效内存。不会擅自杀旧进程、清缓存或安装包。

### 3. GPU冒烟：先joint，再Connector

```bash
"$MVP_PY" "$MVP_CODE/scripts/corrector_multilayer_train.py" smoke --config "$MVP_CONFIG" --output "${MVP_RUN}_smoke_joint" --variant joint --device cuda --precision fp32 --batch-size 2 --width 96 --depth 2 --overfit-steps 30
"$MVP_PY" "$MVP_CODE/scripts/corrector_multilayer_train.py" smoke --config "$MVP_CONFIG" --output "${MVP_RUN}_smoke_connector" --variant connector --device cuda --precision fp32 --batch-size 2 --width 96 --depth 2 --overfit-steps 30
```

必须看到`status=passed`。各执行两个真实batch反向更新，检查共享主干及每个头非零梯度和权重改变，再检查同batch损失至少下降5%；写`smoke.json`，模型丢弃。这里的尺度只由smoke训练batch估计，与正式训练557对尺度明确分开。报告loss、梯度、更新标记、显存峰值；未通过不开始fit。如果失败或中断，保留目录，换独立smoke路径诊断，不改数据定义。

### 4. 小规模、可断点的A/B配对训练

先完成preflight及两个smoke，再执行：

```bash
mkdir -p /root/multilayer_training_runs
nohup "$MVP_PY" -u "$MVP_CODE/scripts/corrector_multilayer_train.py" paired --config "$MVP_CONFIG" --output "$MVP_RUN" --device cuda --precision fp32 --epochs 5 --patience 0 --batch-size 2 --width 96 --depth 2 --seed 42 --lr 0.0003 --weight-decay 0.0001 --cpu-threads 4 > "${MVP_RUN}.log" 2>&1 < /dev/null &
echo $!
```

依次A再B，不重新提取/训练VLA。参数与样本/尺度来源写`protocol.json/scales.json`；初始验证、逐轮历史JSON/CSV、best/latest、结果JSON独立保存。best/latest为原子发布，latest包含模型、优化器、调度器、随机状态、best和完整history。单轮是事务恢复边界；中途故障恢复该轮，不把半轮当完成。SSH断开不影响nohup任务。

默认FP32；若显存不足，保留旧输出，为**两种配置一起**换新RUN并显式使用`--precision bf16`，同样重做smoke；不静默降精度、减空间或裁剪监督。batch2/width96只是未经4090实测的保守起点；不要直接扩规模。单独fit可用相同参数和`--variant connector|joint`、`--patience N`，但早停后A/B步数可能不再相同，应另行报告。

### 5. 查看、暂停、恢复

```bash
tail -n 8 "${MVP_RUN}.log"
cat "$MVP_RUN/connector/status.json"
cat "$MVP_RUN/joint/status.json"
"$MVP_PY" "$MVP_CODE/scripts/corrector_multilayer_train.py" results --config "$MVP_CONFIG" --output "$MVP_RUN"
```

B尚未开始时没有joint/status.json是正常的。两者status为completed且comparison.json的matched_steps=true才是全部完成；result给出best/final每层指标、参数量、真实轮耗时及显存。缺文件不等于失败的实验结果。

安全暂停（只新建本次队列自己的请求文件）：

```bash
touch "$MVP_RUN/STOP"
```

当前完整epoch结束并保存后停；如A结束则不启动B。进程会记录ResourcePause退出，最后完整latest可恢复；也可向本次PID发SIGTERM，处理器在epoch边界响应。不用kill -9，不删除数据/权重。

恢复先确认GPU无其他进程和空间足够，保持**所有训练参数、源码、输入配置不变**：

```bash
rm -- "$MVP_RUN/STOP"
nohup "$MVP_PY" -u "$MVP_CODE/scripts/corrector_multilayer_train.py" paired --config "$MVP_CONFIG" --output "$MVP_RUN" --device cuda --precision fp32 --epochs 5 --patience 0 --batch-size 2 --width 96 --depth 2 --seed 42 --lr 0.0003 --weight-decay 0.0001 --cpu-threads 4 --resume >> "${MVP_RUN}.log" 2>&1 < /dev/null &
```

rm只用于自己创建的STOP；若不存在，不必运行。已完成A会校验并跳过更新，未开始B正常启动；半途B从完整epoch恢复。epoch0也有checkpoint。CUDA OOM/空间不足会暂停，不自动重试；若要改batch/精度/epoch预算，使用新输出目录重新进行公平A/B，禁止篡改旧protocol。SHA错误停止，核对对应导出/缓存来源，不能跳过验证。初始化阶段若尚未产生latest便失败，换新输出重试；不将其当可恢复训练。部分写出的`.partial`不算可用checkpoint，保留现场。

### 6. 回传轻量结果

完成后在远程生成仅JSON/CSV的独立结果ZIP（无权重、原始特征）：

```bash
"$MVP_PY" - "$MVP_RUN" <<'PY'
from pathlib import Path
import sys,zipfile
root=Path(sys.argv[1])
with zipfile.ZipFile(str(root)+'_reports.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv'):
            z.write(p,str(p.relative_to(root)))
print(str(root)+'_reports.zip')
PY
```

本地用`scp AUTODL_SSH_HOST:/root/multilayer_training_runs/mvp_20261112_001_reports.zip ./`取回。不要提交权重或原始数据到GitHub。下一步先看Connector是否胜过zero基线、joint对照的误差/耗时，以及跨层学习是否稳定，再决定是否扩大；不预设多层一定胜出。
