# 当前主线：从相机扰动失败走向VLA-Corrector

2026-11-05。训练练习收束：真实batch、用户单步、官方五步、梯度/参数更新、checkpoint重载与离线评分均已完成。
不再把增加LoRA步数、在新base上做出成功视频作为Corrector前置条件。原训练Notebook保留作工程参考，不删除原结果。

2026-11-07：用户配对循环已完成且通过真实472帧检查。下一步改为[源码导读与数据实践](corrector_source_walkthrough.md)，
下方“待实现配对”保留为练习背景，不是当前重复任务。CPU三轨迹准备/重跑已完成，GPU提取尚未实测。

## 先把研究问题定准

原问题：相机扰动后，冻结SmolVLA可能抓空、抓取不稳或物体掉落，却继续执行原动作块。
新问题：**能否根据视觉变化与预测不一致，及时发现偏离、停止旧动作并重新规划；什么时候有用、什么时候无效？**

继续使用已有的smolvla_libero固定策略与LIBERO-plus相机设置，不混入这次smolvla_base的五步adapter。
原固定H50/H20实验已在view_5、view_10看到部分恢复，也有两种窗口都失败的案例，适合研究自适应执行窗口。
view_5/view_10已用于开发，不能随后重新命名为“从未见过的最终测试集”。保留全部配对seed与失败案例。

训练前失败、训练后成功可作为展示案例，但不是必须兑现的实验判据；还要报告整体结果与退化案例。
“官方任务”不意味着初始base已经掌握它；本轮不扩张训练，是因为两条示范/五步用于流程验收，未构成相机域适应实验。

## Corrector在做什么：四件事分开

冻结视觉编码器得到z_t=E(o_t)。先训练一个外部预测器：

```text
输入：当前视觉token z_t [B,L,D]，动作 a_t [B,7]
输出：预测视觉变化 delta_hat [B,L,D]
监督：delta = z_(t+k) - z_t
```

然后在线比较delta_hat与真正到来的delta。持续偏离时清空未执行动作队列，重新调用冻结策略。
论文还包含OGG（Online Gradient Guidance，在线梯度引导），用于下一次恢复查询。
只完成“特征预测→监控→普通重规划”，应称Corrector子机制复现，不称完整论文复现。

这个预测器不预测Flow Matching速度，不直接输出7维纠正动作，也不是人工抓取成功分类器。
训练标签由前后观测自动生成，不需要逐帧手标“失败”。评估报警是否对准失败时，仍需视频/物理诊断标签。
OGG是在冻结网络权重时对采样变量求梯度，与更新Corrector参数的训练不同。

## 今天只看三处源码

固定参考代码927a06da8f00d7df483459e9039219e076c846de，在outputs/source_audit/vla-corrector中，只读保留。

| 顺序 | 源码与方法 | 需要讲清楚 |
| --- | --- | --- |
| 1 | src/siglip_dynamics/dataset.py 的__getitem__ | 同一episode内选t与t+k；用z_next-z_t作为标签 |
| 2 | src/siglip_dynamics/MLP.py 的SiglipResidualMLP.forward | 动作先投影，再沿L个token复制，与各视觉token拼接，输出同形状delta |
| 3 | src/siglip_dynamics/train.py 的_combined_loss | MSE、余弦或二者组合；它监督的是视觉变化，不是动作误差 |

SmolVLA特征路径参考modeling_smolvla_modified.py的encode_siglip_only：prepare_images→embed_image→沿token轴拼接相机。
这里用的是实际策略的视觉路径，包括connector投影，不要因为文件叫siglip就另加载一个独立SigLIP替代。
L、D和相机顺序以真实缓存记录为准，config.py的token_dim=2048不是SmolVLA的通用维度。

先不用通读一千多行的检测脚本。随后再看lerobot_eval_modified_detection.py：
它将预测放进(t+k, delta_hat)队列，时间到达后与历史z_t对齐，计算1-cos(delta_real,delta_hat)，
再经过平滑、持续超限与冷却规则决定是否清空action_plan；不是单个误差一大就必然重规划。

## 没有失败相机视角的专家数据，并不等于不能做

1. **学习监控器：**可以从正常示范或冻结策略的真实仿真执行中，获得(观测、执行动作、下一观测)，训练视觉变化预测。
   不要求为失败状态补一条“正确动作”。但正常视角训练的预测器未必适应新视角，误报/漏报必须验证。
2. **后续补新视角示范：**若有原始LIBERO完整模拟器状态序列和模型XML，可在同一3D轨迹上重新设置相机并渲染，
   对纯被动相机变化保留原专家动作标签。无需真机；已有Parquet的8维机器人状态不含全部物体位姿，不能单凭它严格重建场景。
   2D裁剪/变暗不能等同于真实3D视角重渲染。这个路线暂不作为本阶段前置工作。
3. **验证是否有用：**利用既有抓空、掉落与H20/H50配对案例，看报警是否及时、重规划是否能恢复。
   若同一错误视角下重新预测仍抓空，正确报警也未必能救回；Corrector不是相机标定或3D理解的替代品。

## 数据与接口：先统一，再训练

元信息实际核对：alphabet soup任务有44条示范、6867帧；目前本地已核对并下载的是813/818/825三条，共472帧。
三条只用于接口/时序检查，不把它们视为已足够训练有效Corrector。
旧视频通常只存主相机，旧物理采集只覆盖部分时间，它们不等于完整的双相机视觉特征训练缓存。

11月7日按论文训练路径明确：先从正常专家示范提取**同一冻结策略、同一预处理**的特征和动作，而不是再微调VLA。
在线执行轨迹先用于监控评估与失败诊断；若后续加入训练，需区分正常/失败数据及其目的，不把全部失败轨迹默认为正常动力学标签。
缓存必须记录episode/frame、时间戳、相机顺序、特征来源权重SHA、动作尺度、成功/失败和有效长度；先完整保存再离线训练。
首次先做默认视角成功执行的数据链路，view_5/view_10用于开发期失败诊断；训练/验证按episode分组，阈值不在最终测试回合调。

动作先统一为同一策略processor下的7维归一化动作，离线/在线保持一致。
原辅助接口里的raw_actions来自sample_actions，尚未经过postprocessor，“raw”不等于环境尺度。
原extract.py有可选QUANTILES归一化，而现有SmolVLA配置为MEAN_STD，不能照搬参数名后假设一致。

先用k=1核对精确的一步动作/下一帧关系，再核对原实现的k=10。
原MLP的k步预测只条件于a_t，没有显式输入后面k步动作；它是一个需要评估的建模假设，不预先称为错误或创新。
10FPS的k=10与20Hz的k=10不是相同物理时长；现有数据的时间语义未确认前，不与20Hz在线缓存混训并声称一致。

## 你现在亲写的一小段：真实轨迹的时间配对

原仓库已有等价用途的配对：dataset.py的_load_quantized_frame_cache按episode连续段建立(t,t+k)，
__getitem__再取数据、反量化并计算delta。它还处理h_window历史与max_samples截断；我们的初版不宣称覆盖这些完整功能。
原缓存依赖行连续约定，我们的接口显式检查frame_index的时间差，避免漏帧时“隔k行”被误当成“隔k帧”。
frame连续、h_window=1且没有max_samples截断时，两种配对应一致，后续接入时要核验这一点。

实现文件：[corrector/transitions.py](corrector/transitions.py)。实际调用入口：[../../scripts/corrector_pair_audit.py](../../scripts/corrector_pair_audit.py)。
不需要GPU、模型或新下载；可以用熟悉的for循环，不要求写Transformer。

以下python需指向安装了pyarrow的本地解释器（已用项目实际环境核对）。

```bash
# 已由助手实际执行：读取三份真实Parquet，检查任务、frame连续性和轨迹长度，不调用你的函数。
python scripts/corrector_pair_audit.py --inspect-only
# 你完成函数后执行：明确调用make_transition_pairs，检查k=1/2的时序边界。
python scripts/corrector_pair_audit.py
```

真实边界例子：全表第153→154行是813号轨迹的最后一对；154→155跨到818，必须排除。
k=1时训练813/818应有154+163=317对，验证825有152对。不能把一个episode的最后一帧与下一条第一帧相减当监督。
你的函数只产生行号；以后真实特征缓存用这些行号取z_i、a_i、z_j并计算delta。当前没有假装已经训练了视觉预测器。
当前实际调用者是CPU审计脚本，不是训练器。实现通过后，下一步新增的特征缓存Dataset应直接调用此函数建立self.pairs，
然后__getitem__返回z_t、a_t、delta_z供DataLoader使用；不复制一套配对实现把你的代码架空，也不修改outputs中的上游源码。
我负责缓存、环境与归档，先审查你的配对函数，再由你实现MLP的短forward；不提前把整套Corrector写完。

## 后续验收顺序

- 特征/动作配对与预测：在独立轨迹上与“预测零变化（下一帧等于当前帧）”及去动作条件基线比较，不能只看训练loss下降。
- 先只监控、不干预：对照抓空/掉落时间，测误报、漏报、延迟；误差峰不自动等同于失败。
- 事件截断+普通重规划：与固定H50/H20对照，记录成功/恢复、动作块调用数、视觉监控耗时、总墙钟时间。
- 再接OGG：适配0.6.1的积分接口与梯度作用域。官方rollout的inference_mode不能直接容纳所需输入梯度；冻结权重不等于禁用全部梯度。

目前不承诺检测或重规划一定提高成功率，也不把挑出的单条成功视频当作完整复现结果。
参考：[论文v2](https://arxiv.org/abs/2607.01804v2)、[固定参考源码](https://github.com/ZJU-OmniAI/vla-corrector/tree/927a06da8f00d7df483459e9039219e076c846de)。
