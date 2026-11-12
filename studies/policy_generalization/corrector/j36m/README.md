# 正式J-36M：实现、真实尺度统计与分阶段部署

2026-11-12。本任务最高优先级为`J36M_LVM_OGG_CODEX_TASK_20261112.md`，同步基分支`feature/multilayer-feature-screening-20261112@8dbd4638e15f39cb04de35ed0e50fcba364a3612`。开发分支`feature/j36m-lvm-ogg-20261112`，独立工作树；原MVP工作树未提交的报告排版保留。

## 首次交付时的证据边界（当前启动更新见下）

| 项目 | 状态 |
|---|---|
| 真实0.767M源码/ZIP/拓扑审查 | 完成，逐字保留721f2ba的MVP模型与数据模块 |
| 正式J-36M和原C0/S2/S8 | 已实现，CPU实数参数与完整形状/梯度测试 |
| 五条真实缓存尺度统计 | 已在远程CPU只读执行并取回，所有输入SHA重验 |
| J-LVM-only / J-LVM+OGG | 已实现，完整36.79M模型CPU合成观测接入原控制类测试 |
| GPU逐站点归因、J真实数据smoke、在线GPU smoke | **未执行**，只能由后续真实GPU报告确认 |
| 70轨迹提取、正式训练、Plus48/60回合 | **未启动** |
| 远程LIBERO-Plus源码/配置 | 当前未发现，须独立准备与核验，不替换普通LIBERO |

CPU测试清单与结果见`VALIDATION.md`。此处不是成功率报告，0.767M历史结果也不是正式J或机器人在线结果。

## 模型与协议

| 配置 | 参数量（真实sum） | 输入/输出 |
|---|---:|---|
| 历史J-small-MVP | 767,328 | width96/depth2旧工程模型，保留不用作正式对手 |
| J / J_equal / JC | **36,793,536** | 两中层+Connector+当前7D动作，三输出头 |
| C0 | **38,052,800** | 上游M20原MLP，Connector-only |
| S2 / S8 | **37,266,176** | 同上游原MLP，仅token_dim=768，原2048 Token |

J严格实现设计：4×4局部重组12288→512适配×2、Connector960→512、1536→768融合；7→256→1536动作FiLM；3个768/8head/FF1536的pre-norm Transformer；Connector头768→960；两中层解码768→256→12288并unpatchify；各自1024→128→768局部高频skip；两相机与64个区域位置编码。输入以训练集固定RMS归一，3个RMS为buffer不计入参数。没有width/depth缩小开关。

重组严格是置换：`[B,2,32,32,768]↔[B,128,4×4×768]`，完整32×32输出及监督不变。已核对[Transformers v5.5.4 SmolVLMConnector](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/smolvlm/modeling_smolvlm.py)及远程安装源码的排列；真实帧station命令还要数值核验实际connector.pixel_shuffle与分组一致，不能用CPU排列证明替代GPU图像核验。

C0/S2/S8使用`reference_mlp`中逐字上游MLP/config副本，附原LICENSE及文件SHA；没有修改安装包或冻结参考目录。旧训练器的优化器、显式Loader、梯度更新、原子checkpoint及MVP严格mmap加载器被复用。

训练10Hz/k10=存储时标1秒，在线20Hz/20步；此映射不补造原始采集物理时钟证据。只输入当前特征与当前归一化动作；未来帧仅构造delta。每层`MSE/(s_l²+1e-12)+β(1-cos)`，s只由训练集估计，β固定1；主J权重(.25,.25,.50)，J_equal固定等权，JC只监督Connector。验证报告每层raw/normalized MSE、zero基线、余弦、逐Task/Episode；J/J_equal/JC与C0均按验证Connector normalized MSE选best，S2/S8按各自层同指标选best，选择指标序列化进protocol；零向量不伪造余弦。每轮首batch记录共享`fusion.weight`的各头梯度范数和夹角，明确是共享参数探针，不是整网梯度相等的保证。

## 真实尺度统计：已测与未测

五条旧缓存813–817，精确遍历全部749帧及699个k10窗；训练仅813/814/816/817（557窗），验证815（142窗）。

| 层 | train原值RMS | val变化RMS | val zero raw MSE | 使用train尺度的zero normalized MSE |
|---|---:|---:|---:|---:|
| Block2 | 0.342600 | 0.195388 | 0.03817663 | 0.987051 |
| Block8 | 0.832624 | 0.492511 | 0.24256736 | 0.993901 |
| Connector | 7.044938 | 3.150711 | 9.92698071 | 0.996522 |

Connector已在原值空间上明显大于中层；MSE平方放大RMS差异，**不是Token数量导致逐元素均值变大，也不是较难预测的证据**。pixel shuffle为纯置换，不可能自行放大范数。究竟主要来自后续视觉层、post-norm或学习投影，必须等同帧逐站点GPU测量；本轮未作因果断言。原浮点Connector与INT8缓存的同帧/差量误差也尚待station实际测量。

`scale_summary_real5.json`为真实摘要；完整报告在`results/20261112/j36m_development/feature_scale_audit.json`，包含精确mean/std/RMS/max、相机/Token/通道RMS、最大变化窗口。元素分位数为**每帧分层随机抽样最多2048个元素**的估计，有固定seed与方法声明；不冒充十亿元素精确分位数。001旧采样报告留档，002修订采样避免按通道周期取样偏置；精确矩不变。

## 在线语义与公平性

`scripts/j36m/online.py`继承现有`OnlineCorrector`，通过打包三层Tensor桥接原到期队列、事件处理和去噪入口；不复制控制循环。robust状态机仍来自固定源文件的原两定义。仅替换多层观测/评分/引导接口。

- `j_monitor`：只评分，禁止清动作队列/OGG。
- `j_lvm`：同一评分、阈值及触发，原队列清空并重规划，guidance_calls必须0。
- `j_lvm_ogg`：仅在相同事件重规划的去噪过程中增加OGG。
- 三层小范数/非有限方向跳过并重新归一有效权重，全无效不更新状态机、不当成异常。
- 自然动作块刷新保留单动作条件的待到期预测，避免平白制造A10式监测空窗；纠偏中断会显式取消旧预测，再为新计划建预测。
- 原robust函数的EWMA/滞回/持续计数/冷却保留；正常历史基线冻结，测试回合不再更新用于阈值拟合的clean_history。独立default成功校准seed9000/9001，禁止使用1000–1002及view5/10。融合权重按正常覆盖率/(MAD+.05)估计并冻结，是启发式可靠性估计，不是失败分类器。
- OGG目标是已到期`expected-observed`，不是未来观测。候选动作只取首动作7维，损失对速度v求导，网络冻结；记录每层loss、梯度范数/夹角、动作更新量及同步GPU延迟，纠偏目标FP32另存本次回合文件。

Plus沿用旧语言、同物理初始状态、相机default/view15/view5/view10、固定Plus revision4976dc3、policy revisiona24f1f3、chunk50/10次去噪/20Hz。已从真实20261031默认回合导出小型`historical_plus_reference.json`：核对资产revision、基础BDDL SHA、机器人/settling及物理state；不是新生成初始状态。每回合还核对同条件的首噪声、策略输入，记录提交环境动作（明确是LiberoEnv.step输入，不冒称底层物理执行位移）。

正常校准→5方法default强制接口smoke（不作性能结果）→default/view15开发24回合→冻结→view5/view10保留24回合；主矩阵共48回合。显式加j_monitor则60回合。online smoke必须额外证实j_monitor与H50完整动作相同、两J方法首触发一致、仅OGG有引导；正式轨迹分叉后不强求分数相同。

## 70条清单和空间

manifest70按Task20–29每个最低ID的5train+2validation选择，完全不读取成绩；包含全部813–817。50训练/20验证，10265帧，6791/2774窗；23个源分片2,208,774,108字节。两层原始总60.15GiB，复用系统盘4.39GiB，新增数据盘55.76GiB。

实际重连只读df确认数据盘130GiB/约88.09GiB可用，系统盘约15.14GiB可用，GPU空闲。最低系统8GiB、数据20GiB；开提取前另计未完成raw、缺分片及8GiB检查点/分析预算。**新的Plus资产额外空间尚未核实，不包含在“70条能放下”之中**，缺资产或低空间会停，不能自动清旧数据/扩容。

提取复用已验证raw exporter，每次一条新Episode独立进程；完成并验SHA才原子发布到新raw70，已有5条不动。代价是每条重新加载冻结模型，比单常驻编码器慢，换取隔离及Episode级恢复；吞吐须实测，不保证今晚全部完成。失败的新`.work`尝试保留，不删除原始缓存。断点续接是完整Episode，不是把不完整FP16当已完成。

## 部署与启动

源码ZIP不含模型、原始FP16/Parquet、环境、视频。配套安装器只用标准库，检查ZIP与每文件SHA、路径及符号链接，拒绝覆盖。

本地：

```bash
cd /home/hhd/robot-lab/outputs/autodl_setup/j36m_20261112
sha256sum -c corrector_j36m_20261112.zip.sha256
scp corrector_j36m_20261112.zip corrector_j36m_20261112.zip.sha256 package_j36m.py AUTODL_SSH_HOST:/root/
```

远程（主机别名/端口按控制台，不保存密码）：

```bash
cd /root
sha256sum -c corrector_j36m_20261112.zip.sha256
J36_SHA=$(awk '{print $1}' corrector_j36m_20261112.zip.sha256)
JCODE="/root/autodl-tmp/corrector/dev/j36m-${J36_SHA:0:12}"
python3 /root/package_j36m.py install --archive /root/corrector_j36m_20261112.zip --sha256 "$J36_SHA" --destination "$JCODE"
JPY=/root/autodl-tmp/corrector/runtime/venv/bin/python
JROOT=/root/autodl-tmp/corrector/j36m_runs/formal_001
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$JCODE"
```

第一次仅检查并运行GPU smoke，**不提取70条，不长训，不评测Plus**：

```bash
"$JPY" scripts/corrector_j36m_pipeline.py plan --root "$JROOT"
nohup "$JPY" -u scripts/corrector_j36m_pipeline.py run --root "$JROOT" --until smoke --max-hours 2 > "${JROOT}.launch.log" 2>&1 < /dev/null &
```

顺序为：preflight（只读全454旧Connector及5raw、资源/参数）→CPU单测→5条cache统计→同帧GPU station→J/C0/S2/S8/J_equal真实smoke。station需同帧重算量化严格匹配旧cache、原始FP16匹配、实际shuffle匹配；任一失败即停。GPU smoke为真实2batch、30次单批更新、各头/共享梯度、权重更新、重载、显存与秒数，模型丢弃；不是正式训练结果。原始分辨率不变，默认microbatch1/FP32。

看日志与状态：

```bash
"$JPY" scripts/corrector_j36m_pipeline.py status --root "$JROOT"
tail -n 15 "$JROOT/gpu_J.log"
cat "$JROOT/receipts/gpu_J.json"
cat "$JROOT/smoke_J/smoke.json"
```

先根据真实smoke耗时估计S2/S8的2048Token训练成本，再决定后续范围；不能凭参数量估算墙钟。**未取得这些GPU结果前，本交付不声称显存或完成时间。**

### 只扩数据，或再开始正式训练

smoke全部成功后才可显式继续：

```bash
nohup "$JPY" -u scripts/corrector_j36m_pipeline.py resume --root "$JROOT" --until extract --max-hours 6 >> "${JROOT}.launch.log" 2>&1 < /dev/null &
```

默认不联网；所需分片缺失会报告并停止。若确需下载，必须在**最初plan/run时**加`--allow-source-download`（需代理再加`--proxy-url http://127.0.0.1:PORT`），之后保持同一选项恢复。已经冻结的计划不能临时偷偷加网络；可在独立阶段命令中显式执行extract70并记录新计划。不会下载模型或全量数据集。

70条完整SHA及新训练尺度审核完成，明确选择长训练：

```bash
nohup "$JPY" -u scripts/corrector_j36m_pipeline.py resume --root "$JROOT" --until train --max-hours 12 >> "${JROOT}.launch.log" 2>&1 < /dev/null &
```

默认正式预算每配置30轮、无早停，microbatch1×accumulate32、seed42、AdamW lr3e-4/WD1e-4、Cosine衰减；各配置每轮213更新、共6390次，C0/S2/S8/J/J_equal严格匹配。不是承诺12小时全部跑完；预算到达按完整epoch/episode/回合边界暂停。`--patience`仅独立train可显式设置，之后不能再称等更新预算比较。JC为可选独立消融，需自己的gpu_JC验收记录，不在默认长队列。

### Plus环境前置条件与明确启动

当前远程缺Plus源码/配置。**不要执行下面评测范围，直到固定Plus与历史资产已准备。** 不自动pip、不改libero-original、不自动下载6.4GB资产。

可在新的路径获取固定源码（不接触已安装普通LIBERO）：

```bash
PLUS=/root/autodl-tmp/corrector/runtime/libero-plus
mkdir "$PLUS"
git -C "$PLUS" init
git -C "$PLUS" remote add origin https://github.com/sylvestf/LIBERO-plus.git
git -C "$PLUS" fetch --depth 1 origin 4976dc30028e805ff8094b55501d532c48fec182
git -C "$PLUS" checkout --detach FETCH_HEAD
```

独立`runtime/libero-plus-config/config.yaml`必须指向该Plus的benchmark/bddl_files/init_files，以及**已经验证的**资产目录；资产目录需真实`.lerobot_camera_assets.json` revision=`dd2bd61b7d9a6fef1abc52d606e983b41886a149`、scenes/textures；不能新写假标记冒充资产。具体原协议见robot-lab的`libero_plus_protocol.md`、`replanning_results.md`和Notebook第5–6节。本方案不自动扩大资产占用；先核对展开大小/空间。运行时直接将新Plus放sys.path，无需重装第三方包。

环境准备好、正式J完成后：

```bash
nohup "$JPY" -u scripts/corrector_j36m_pipeline.py resume --root "$JROOT" --until development --max-hours 4 >> "${JROOT}.launch.log" 2>&1 < /dev/null &
```

Plus子阶段使用现有`runtime/sim-venv/bin/python`；从J best创建只允许监控的校准模板→default独立seed9000/9001正常监控→冻结norm门槛/融合权重/阈值→default五方法接口smoke→default/view15开发24回合。J-LVM与J-LVM+OGG共用同一J/阈值/队列/协议，只有引导开关不同。任何不一致停止。

开发结果及配置检查通过、保持模型和阈值冻结，再显式开始保留视角：

```bash
nohup "$JPY" -u scripts/corrector_j36m_pipeline.py resume --root "$JROOT" --until heldout --max-hours 4 >> "${JROOT}.launch.log" 2>&1 < /dev/null &
```

heldout强制读取完成的development receipt、同checkpoint/calibration SHA及development物理初始参考；保留view5/view10各3seed，24回合。不将校准/强制smoke混入48回合成功率，不据3seed推断显著性。

### STOP、恢复与错误处理

```bash
"$JPY" scripts/corrector_j36m_pipeline.py stop --root "$JROOT"
```

只写本次`JROOT/STOP`，不杀任务；训练在完整epoch、提取在完整Episode、评测在完整回合边界保存。SIGINT/SIGTERM也转为该请求，不用kill -9。

恢复前检查GPU/两盘、读失败日志，移除**本次自己创建**的STOP：

```bash
nvidia-smi
df -h / /root/autodl-tmp
rm -- "$JROOT/STOP"
nohup "$JPY" -u scripts/corrector_j36m_pipeline.py resume --root "$JROOT" --until train --max-hours 12 >> "${JROOT}.launch.log" 2>&1 < /dev/null &
```

如STOP不存在无需rm。已完成阶段保留，训练自动追加--resume，latest包含模型/optimizer/scheduler/best/完整历史/Torch+loader+Python+NumPy+CUDA RNG。未完成epoch回退重做，配置/代码/数据SHA变更拒绝恢复。提取已有完整raw验SHA跳过；新partial保留，不删除重建旧5条。Plus调用原run_plan按完整回合恢复。

**诊断阶段（cache-audit/station/gpu-smoke/CPU tests）中途失败不会无限重试或覆盖目录。** 队列会停并提示，使用独立新诊断output运行同一显式stage，审核通过再处理阶段交接；不要直接删除失败现场或伪造receipt。训练OOM可先用已声明的gradient checkpointing/microbatch/AMP在**新实验ID**做smoke，再另建冻结训练计划；本轮默认已经microbatch1。不能静默缩小J、分辨率或修改旧protocol。数据盘<20GiB、系统盘<8GiB停，不自动扩容/清旧结果。

阶段门槛绑定完整源码/数据选择哈希和实际证据文件SHA；错误/缺失gate不能跳过。日志在JROOT各stage.log，pipeline_status.json记录阶段、PID、秒数、退出码，experiment_index.json冻结命令/配置。checkpoint只留远程，不进入Git。

## 结果回传和分析

正式训练每模型`result.json/history.json/metrics.csv/protocol.json`；训练按Task/Episode报告，样本/参数/秒数/显存/梯度同存。Plus沿用batch_evaluation的trials、overall、执行状态和视频SHA，并额外存层分数/有效性、目标、OGG、原始提交动作及配对输入审计。先看相同条件下J-LVM-only↔OGG的成对救回/退化，不只比较总均值。

本地`j36m_development`报告只含代码/真实缓存CPU统计；正式训练和Plus结果当前为空，不得从离线MSE外推机器人成功率。后续仅取回JSON/CSV/小日志，原始FP16、Checkpoint、大视频继续留远程。

本轮最终源码树CPU测试：J36M专项28项、历史MVP23项、原多层导出9项，全部通过；独立部署包回归另记交付回执。

## 2026-11-13凌晨：已开始远程实际执行

用户授权32小时长期队列后，五条同帧逐站点GPU审计、J/C0/S2/S8/J_equal的batch1和batch8真实GPU smoke均已完成；新的70条完整FP16正在提取（早期快照已发布10条新Episode）。正式60轮/三种子训练尚在等待完整70条验证，不把排队写作已训练。

当前运行源码固定`1eb520f`，部署ZIP SHA=`35ecdd7488bfcf79b844eb2c2ede2b9d9a34722fe0d1efe458b89a623e2266f4`，监督器原始deadline=北京时间2026-11-14 08:51:56。实际状态以新独立status.json为准。主15模型/JC后续、双盘路径与恢复命令见[长期队列说明](LONGRUN_20261113.md)，GPU证据见[启动报告](../../results/20261113/j36m_remote_start_001/RESULTS.md)。

此更新没有在运行中修改源码/权重/原计划；初始单batch门槛曾遇LEROBOT_SRC路径优先级错误，保留日志后显式指定已有LeRobot，未安装新依赖。新长计划数据盘5GiB硬下限（按用户要求），系统盘8GiB，另有pending数据/Checkpoint预算；不再将20GiB固定余量当硬要求。Plus环境仍未准备，不声称闭环已完成。
