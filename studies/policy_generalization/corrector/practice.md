# 核心实现练习：写哪里、怎样调用、怎样验收

2026-11-07。只实现下面五处；不要求你再写环境、下载器、日志或批量评测调度。
这些方法用于在本地CPU主动理解算法并与上游对照，不作为正式训练的默认依赖；正式入口优先使用固定上游实现。

最新状态：用户已填写五处核心。2026-11-07审查修复了MLP漏最后一个block、损失的方向/批次维/返回值问题；五阶段检查通过，并在两种小构造结构中与上游同权重前向及参数梯度对齐。尚未用472帧真实缓存训练预测器。

| 次序 | 文件与方法 | 你需要完成 | 上游参考 |
| --- | --- | --- | --- |
| A | model.py：ResidualMLPBlock.forward | 用已注册层完成分支，最后与输入相加 | MLP.py：_MLPBlock.forward |
| B | model.py：VisualDynamicsMLP.forward | 动作投影/展开/拼接→残差块→输出 | MLP.py：SiglipResidualMLP.forward |
| C | losses.py：corrector_loss | MSE与逐样本展平后的余弦项，返回三个Tensor | train.py：_combined_loss |
| D | dataset.py：CachedTransitionDataset.__getitem__ | 反量化前后帧，取动作，计算delta，返回Tensor字典 | dataset.py：__getitem__ |
| E | training_step.py：train_corrector_batch | 将已学的单batch训练迁移到双输入模型及新标签 | 原train_one_batch与上游train循环 |

初始框架提供层结构、初始化、文件读取、形状检查、分割组合；核心方法现已由用户填写并经审查修正。
不要把答案写进Markdown，直接填写对应.py方法。参考源码位于outputs/source_audit/vla-corrector/src/siglip_dynamics。

真实调用关系：

```text
check_corrector_practice.py
  → 用户block/model/loss/getitem/train_corrector_batch

练习中的训练调用关系：
build_episode_splits → 用户Dataset/配对 → DataLoader
  → 用户train_corrector_batch → 用户model/loss → backward/step

正式入口corrector_train.py：固定上游Dataset/MLP/loss
  → 项目提供的逐轨迹划分、训练/验证与checkpoint编排
```

初始化数据集已经真正调用你的配对函数；不是把它留在独立演示文件里。
训练/验证按每条缓存的metadata.split组合；相同episode跨集合会报错，不对配对样本随机拆分。
自写MLP保留上游的层/权重命名，但第一版只接收带batch输入，默认宽度较小；不声称配置与论文规模相同。

```bash
python scripts/check_corrector_practice.py --stage block
python scripts/check_corrector_practice.py --stage model
python scripts/check_corrector_practice.py --stage loss
python scripts/check_corrector_practice.py --stage dataset
python scripts/check_corrector_practice.py --stage step
```

也可使用--stage all。未写的方法显示“待用户实现”，退出码2；这不是环境安装失败。
形状/数学断言失败则保留实际traceback，用它定位错误；不修改检查来迁就错误实现。
这些检查用明确的小构造张量，不是视觉编码器实测或预测器效果评测。通过后再与上游同权重输出做进一步对齐。

第一轮可以对照源码写，不要求默写。每完成一个方法，说明输入、每行的形状变化、输出给谁，再运行检查。
优先理解并亲自写通这几十行；数据工作台和Notebook内嵌的长环境代码不用逐行背诵。

## 本轮审查：用户注释中的问题与关键修正

1. `drop`是`nn.Dropout`：训练时以概率p将元素置零，保留的元素除以1-p，以保持期望值；不是删除token或改变shape。当前默认p=0，因此不改变数值；eval模式也不随机置零。例如p=0.5时，输入[2,4,6,8]的一种可能输出是[4,0,12,0]，不是固定丢掉哪几项。
2. 对[B,C]输入，`unsqueeze(1)`与`[:,None,:]`都得到[B,1,C]；随后expand到[B,L,C]，把一个样本的动作特征交给它的每个视觉token。每个样本仍使用自己的动作。`expand`通常用共享存储的视图，不要原地修改展开结果。
3. block在同一宽度内计算并做残差相加；transition连接相邻block，当宽度不同则投影，宽度相同则Identity。例如widths=(256,128,128)，计算为block0(256)→transition0(256→128)→block1(128)→transition1(Identity)→block2(128)。3个block只有2个transition；zip按较短序列截断，会漏block2。现在遍历全部block，按索引判断其后是否还有transition。
4. `reshape(-1)`和默认`flatten()`都将所有轴展成一维，数值顺序一致；这里应保留batch轴。实际特征为[L,D]=[128,960]，若batch=16，则余弦比较输入要是[16,122880]，而非[1966080]。前者逐样本比较再平均，后者混合全部样本后只比较一次，损失和权重含义不同。
5. 相似度越大越好，损失要越小越好，因此用`1-cosine_similarity`。例如预测[1,2]、目标[2,4]，MSE=2.5、cos=1、方向损失=0：方向对了但幅度仍不对；两项结合才兼顾方向与数值。直接最小化cos会鼓励反方向。实现返回(total,mse,cosine_loss)三个标量Tensor，和训练调用中的三变量接收对应。
6. Dataset的`scale[:,None]`是[L]→[L,1]；例如量化特征[[10,-20],[3,4]]与scale=[0.1,0.5]相乘，恢复为[[1,-2],[1.5,2]]。每个token有自己的scale，其各维共用这个scale。原用户实现正确，只去掉return后不可达的占位异常。

残差块和单batch更新顺序保留用户实现。`.item()`在backward与step之后用于返回日志是合适的；不能先把loss变成Python数值再反向传播。单步入口目前是无AMP/梯度裁剪的最小实现；参考训练器的梯度裁剪、验证、保存恢复等由完整训练入口另行接入，不声称已经复现整个上游训练配置。

本次58项CPU检查通过，其中含五阶段计算检查、逐样本余弦回归，以及同权重的等宽(8,8)/变宽(8,6,4)上游输出与各参数梯度对齐。这里是小构造数据的实现验证，不计作真实缓存训练或Corrector成功率提升。

用户后续明确：本轮重实现的主要目的为理解与语法补足，不要求将其作为生产分支。已据此调整正式调用链，见[真实训练运行说明](../corrector_training_runbook.md)。


## 2026-11-09：历史输入与Transformer设计练习（新任务，尚待用户实现）

本轮读`outputs/source_audit/vla-corrector/src/siglip_dynamics/transformer.py`与`dataset.py`，写`temporal_practice.py`。上游已有Transformer和history支持，本轮目标是亲手掌握设计与张量流，不把启用现成架构写成原创方法。当前已完成的MLP训练和300回合评测不导入此练习文件。

先说明建模问题：已有MLP输入当前视觉z_t与动作a_t，预测delta=z_(t+k)-z_t。单帧对运动方向和遮挡状态的信息可能不足，所以尝试增加过去H帧。未来z_(t+k)只进入监督标签，不能混进输入历史。k决定预测多久之后，H决定看多少过去，两者独立。

层与输入检查已经搭好，只留三个方法：

| 核心方法 | 用户需要写什么 | 验收重点 |
| --- | --- | --- |
| history_window（T1） | 计算截至t的H帧切片，检查同episode，返回[H,L,D] | 无未来帧、无负下标绕回、无跨轨迹 |
| TemporalPredictor.pack_tokens（T2） | 视觉线性投影、时间embedding、合并历史/token轴、在首部拼动作token | 不合并batch，保持顺序，层在__init__注册 |
| TemporalPredictor.last_frame_tokens（T3） | 去掉动作token，恢复[H,L]轴，只选最后一帧的L个输出 | 输出[B,L,C]，不直接对所有token求均值 |

实际形状（B=2,H=3,L=128,D=960,C=768）：

```text
视觉历史 [2,3,128,960]
  → 视觉投影与时间embedding [2,3,128,768]
  → 合并H和L [2,384,768]
动作 [2,7] → 线性投影并增加token轴 [2,1,768]
  → 拼接 [2,385,768]
  → TransformerEncoder [2,385,768]
  → 去动作token并取最后一帧 [2,128,768]
  → LayerNorm与输出投影 [2,128,960]
```

设计问题先自己回答：为什么768能分成12个head（每头64维）？为什么要保留时间embedding？为什么动作只投影为一个条件token？为什么保留最后一帧的128个位置，而不是平均成一个向量？这里输入全是过去，允许历史token彼此注意不等于使用未来信息；是否需要因果mask由预测任务决定，不能机械照搬自回归语言模型。

H=3在10FPS数据中覆盖过去0.2秒，在线20Hz应取t-4、t-2、t而非最近三个连续控制帧。扩大H会增加序列长度和注意力开销，不保证越多越好。当前框架支持的是单动作条件，不把它误称为完整动作序列条件预测器。

先显式运行上游小CPU示范（只用小张量，无下载/训练）：

```bash
python scripts/check_temporal_corrector.py --implementation reference --stage all
```

第一步只写T1，然后用学生实现验收：

```bash
python scripts/check_temporal_corrector.py --implementation student --stage history
```

再写T2/T3，用`--stage model`检查H1/H3的形状、动作梯度和上游同权重前向/反向一致性。未填写方法明确返回退出码2，不会悄悄调用reference补答案。测试默认D8/C16/L4/H3，避免本地CPU直接承受真实尺寸；真实960维/128token架构与训练吞吐要另做远端预检。

当前`corrector_train.py`正式入口仍固定MLP/H1。练习通过后再补模型工厂、历史Dataset、checkpoint模型类型和在线历史队列，不是给旧入口增加一个参数即可完成接入；对照固定共同有效时间锚点、数据划分、未来跨度和计算预算。
