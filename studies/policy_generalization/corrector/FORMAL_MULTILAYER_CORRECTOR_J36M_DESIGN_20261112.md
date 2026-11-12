# 纠偏设计：正式多层Corrector候选J-36M，不再把0.767M工程MVP当成正式模型

日期：2026-11-12。本文**仅是可检验的架构规范与实验计划，不是已经实现/训练的模型**。用户最初希望获得一个适合有限4090资源、对Block 2/8/Connector进行有意义融合、可用于LVM和OGG的正式Corrector架构。此前交给Codex的轻量任务书只要求MVP，结果使用width96/depth2，产生767,328参数。这个任务交付不足来自架构要求未明确，不应归咎于Codex；已跑通的MVP只证明工程闭环，不能代表正式多层Corrector算法效果。

## 1. 固定输入输出与时间

- 冻结同一`lerobot/smolvla_libero`权重，视觉层从零索引选Block 2和8，双相机顺序固定；
- `F2,F8: [B,2,1024,768]`，来自每路32×32特征；
- `Zc: [B,128,960]`，每路8×8，原pixel shuffle factor4 + connector投影；
- 当前动作`a_t:[B,7]`，冻结processor归一化；不输入`F_{t+10}`或未部署可得的未来实际动作；
- 预测`delta_F2,delta_F8:[B,2,1024,768]`与`delta_Zc:[B,128,960]`，数据10FPS、k=10，1秒跨度；在线20Hz对齐20控制步，不能误作10步；
- 原始中层FP16全分辨率文件**保持不变**，模型内部可进行有记录的下采样/patch grouping，输出和监督保持全32×32。

## 2. 候选J-36M架构（一个模型、真正跨层融合）

1. 两路相机各自将Block2/8 32×32 reshape成8×8的4×4**局部Patch组**。每组16×768=12,288维，共128个组（两相机各64）。Patch顺序必须与Connector pixel shuffle factor4空间布局核对；空间网格对齐不等于特征数值相等。
2. 各层独立Adapter，`Linear(12288,512)`×2，对`Zc`使用`Linear(960,512)`。三个路径输出都为`[B,128,512]`。对各层输入独立归一化，使用**只基于训练集**估计的固定RMS或LayerNorm（LayerNorm如计入参数需重算总量），禁止看validation统计。
3. 逐Token拼接3个Adapters得到`[B,128,1536]`，`Linear(1536,768)`得到`[B,128,768]`；加入8×8位置编码和camera embedding，**两个相机空间位置不能简单假定为同一坐标**。该融合层确实接受三个层的同一区域信息。
4. 动作`Linear(7,256)→SiLU→Linear(256,1536)`生成对主干的FiLM gamma/beta，保证OGG对7维动作可求梯度；动作的256维embedding同时供local skip使用。
5. 共享主干：3×`TransformerEncoderLayer(d_model=768,nhead=8,dim_feedforward=1536,dropout=0,batch_first=True,norm_first=True)`，上下文Token数128，避免把4096个中间层Token直接做全局attention。交互可跨空间区域、跨相机，但需记录注意力的成本与可能的伪对齐风险。
6. Connector Head：`Linear(768,960)`，输出`[B,128,960]`。
7. Block2/8 Head：各自`Linear(768,256)→SiLU→Linear(256,12288)`，按原4×4局部Patch布局unpatchify回`[B,2,1024,768]`。为了保留高频信息，各层额外有本地跳跃预测路径`Concat(F_l token[768],action_embed[256])→Linear(1024,128)→SiLU→Linear(128,768)`，局部输出与共享主干的对应局部Patch预测**相加**作为最终`ΔF_l`。共享主干输出均依赖于2/8/Connector，这才构成跨层融合。
8. 禁止在整个模型/损失中对`F2`/`F8`偷偷从32×32降为8×8目标；如果为了对照显式采用8×8监督，另写独立实验ID。

### 参数量：按上述torch.nn模块**仅初始化并数参**的本地计算（未做GPU训练）

- Block2 Adapter: 6,291,968；Block8 Adapter: 6,291,968；
- Connector Adapter: 492,032；
- Fusion linear: 1,180,416；
- Action FiLM MLP: 396,800；
- Shared 3个TransformerEncoderLayer: 14,181,120；
- Block2 decoder: 3,354,880；Block8 decoder: 3,354,880；
- Connector head: 738,240；
- Block2 local skip: 230,272；Block8 local skip: 230,272；
- Camera embed(2×768)+pos embed(64×768): 50,688；
- **合计36,793,536参数（约36.79M），对比原版单Connector38,052,800，属于同一量级；不是要追求恰好等参。**

上面的数值对应一个具体torch.Module配置，但尚未执行完整forward、GPU显存/性能、LVM与OGG验证；精确实现若增加norm/gate需要重新数参。对于两种中层Block原上游MLP只是`token_dim=768`、`scale=M20`，该版结构参数量应从模型初始化精确输出，不能将小模型替代并称为论文版。

## 3. 训练损失与公平对照

- 主评估任务Connector ΔZ，三层辅助目标分开报告。以训练集RMS归一MSE和可选cosine辅助，`L=λ2 L2 +λ8 L8+λc Lc`；初始候选`λc=0.5,λ2=0.25,λ8=0.25`与等权对照（不能事后只挑好看的）；每种层的真值目标形状完全匹配预测。
- C0: 原38.05M Connector-only MLP；S2: 上游原架构Block2-only MLP；S8: Block8-only；J-36M: 本架构全输入三层监督；
- JC: 同样J架构/输入但仅Connector监督，区分额外输入与辅助预测竞争；
- J-shuffle/J-no-fusion（可选）：打乱其他层输入或移除跨层融合，同时匹配容量/预算，检测是否真正使用额外信息；不得在validation上调参。
- 以固定Episode级train/validation、同seed、固定更新预算和同评估协议对照；不仅记录参数量，还要记录实际GPU时间和吞吐。S2/S8与Connector输出空间不同，原始MSE不可直接比较；比各自zero baseline及跨组在线表现。
- 保留旧0.767M模型作为`J-small-MVP`工程/轻量化对照，不作为正式架构对手。

## 4. 在线部署：三个硬性门槛

1. 动作梯度：模型`eval`且所有参数冻结，输入当前7维动作`requires_grad`，验证`∂loss/∂a`有限非零；不能`detach`掉动作用于融合/skip路径。OGG原方案通过动作相关变量梯度引导候选动作；先使用独立强制触发smoke。
2. LVM：不同层的误差量级差异显著；各层归一化与阈值必须在独立正常轨迹上定标，再于view15开发与view5/view10保留条件测试，不能沿用Connector阈值。
3. 资源：先在4090实测microbatch1/2前向反向峰值、GPU time和在线`1step monitor`+OGG引导时延；不能仅按36.8M参数量判断可部署性。必要时混合精度、梯度累积与局部重计算，显式记录。

## 5. 面向Codex的约束与验收

1. **先查现有真实MVP源码**，重新核对77万模型的每个真实模块及形状；原Source SHA为`721f2bae4c3cf2c9ef7abaf548caf40ee9d9f92d`，目前不在GitHub已知分支，不可凭任务书推测已经存在的实现。
2. 先只实现本J-36M模型类与单元测试，不启动70或454条数据提取；使用合成Batch检查形状、3层交叉梯度（扰动Block2应改变Connector预测，及逆向梯度）、动作梯度、Patch顺序、双相机位置编码区分、full-res目标loss与checkpoint重载一致。
3. 再在已有5条Episode做真实小batch和过拟合测试，报告实测显存和吞吐；若CUDA OOM，先调整microbatch和梯度累积，不能偷偷改成width96/depth2再当作J-36M。
4. GPU smoke全部通过后，才启动多任务数据提取和正式训练。具体Episode集合/分片/磁盘上限由冻结manifest生成，避免训练过程中修改数据选择。
5. 不承诺本设计最优或在线收益；对照、失败和局限同等保存。
