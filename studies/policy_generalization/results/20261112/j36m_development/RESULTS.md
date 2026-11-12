# 正式J-36M开发与真实缓存统计

任务2026-11-12，开发验收更新于2026-11-13凌晨。**这是本地实现与CPU统计报告，没有新J的GPU训练/机器人成功率。**

旧MVP源码从SHA=a98b5be6319054a532af7d35e0939d8684fea48c21485d8a215b808996515565的真实部署ZIP恢复，源commit721f2ba。审计确认767328参数、共享局部主干及三头；所有历史结果和原工作区用户修改保留。正式模型独立实现为36793536参数，C0/S2/S8原版对照仍为38052800/37266176/37266176，详细拓扑与模块数参见architecture_audit.json。

真实统计只读已有813–817，逐个校验新层/旧Connector COMPLETE及所有文件SHA，帧/相机/权重/动作/时间协议一致。749帧、557训练窗、142验证窗，未使用GPU、未重提取图像。所有量化Connector按q.float32×scale.float32反量化。

| 层 | train原值mean | train原值std | train原值RMS | val变化RMS | val zero raw MSE | val zero normalized MSE |
|---|---:|---:|---:|---:|---:|---:|
| Block2 | .014327 | .342300 | .342600 | .195388 | .03817663 | .987051 |
| Block8 | .016308 | .832464 | .832624 | .492511 | .24256736 | .993901 |
| Connector | -.232499 | 7.041100 | 7.044938 | 3.150711 | 9.92698071 | .996522 |

Connector变化RMS约为Block2的16.1倍、Block8的6.4倍，MSE将幅度差异平方放大到约260/40.9倍。原值RMS也已明显增大，不是预测器误差或总Token数导致的逐元素MSE差异。验证分位数抽样p99原值绝对值约为1.133/1.563/22.52；精确最大绝对值10.875/154/257.969，因此同时保留长尾、分组及异常窗口信息，不能只看均值。

pure pixel shuffle只置换值，已核对远程安装源码和CPU布局往返；**哪一站点主要放大**仍待同帧真实GPU Hook，不能从这张缓存表直接归因投影或LayerNorm。原浮点Connector的量化噪声亦须GPU同帧复算，本轮不把9.93推给量化。

完整feature_scale_audit.json为002修订抽样报告；feature_scale_audit_001.json保留早期采样；精确矩一致。source/provenance含原缓存内容SHA，抽样方法、相机/Token/通道RMS、分层delta分位数及最大变化窗口俱全。

正式训练以训练input RMS固定归一各流，delta训练尺度独立拟合。J主loss为(.25,.25,.50)加权的每层归一MSE+cos，J_equal为固定等权，JC可选。归一MSE解决数值幅度公平，不保证共享梯度等大；逐头fusion.weight梯度范数/夹角是明确探针。没有借验证值拟合scale或LVM阈值。

70条清单从原Object10固定manifest按Task20–29最低ID选5train+2val，不看成绩，恰好10265帧/6791训练窗/2774验证窗，23分片约2.06GiB；复用5缓存4.39GiB，新需55.76GiB。真实重连确认130GiB数据盘约88.09GiB空闲及GPU空闲。只完成清单，不冒称70缓存存在。Plus当前缺固定源码/配置，额外资产容量未知；没有购买、删除或启动长期任务。

代码、命令、阶段恢复与限制见[实施README](../../../corrector/j36m/README.md)和[验收说明](../../../corrector/j36m/VALIDATION.md)。

本轮源码CPU验收：正式J36M专项28项、历史MVP23项、多层导出9项全通过；实际秒数与日志SHA见`cpu_test_results.json`。新增部署SHA/路径/拒绝覆盖与NumPy RNG精确恢复检查。原始MVP文件没有为通过回归而改写。独立ZIP解压后的同套回归结果另存交付目录。
