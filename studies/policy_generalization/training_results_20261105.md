# 2026-11-05：首次真实SmolVLA LoRA训练核对

结论：inspect、用户单步、官方5步、保存重载和独立episode的成对预测均已完成真实T4执行。
用户指标已完成，本地CPU评分见[离线结果](offline_eval_results_20261105.md)；尚无机器人任务成功率结论。
Notebook重复运行同一阶段覆盖了输出，本记录结合Drive中的各阶段证据核对，不能仅看最后一个单元格。

## 已核对的证据

| 项目 | 结果 |
| --- | --- |
| 用户train_one_batch | 真实调用1次，loss=3.4351911545，梯度有限且参数变化，返回数值正确 |
| 官方训练 | 5步，每步一次optimizer.step，均有非零梯度与参数变化 |
| 可训练/全部参数 | 371328 / 450417504，约0.0824%，记录中的可训练名字均为LoRA参数 |
| 训练数据 | episode813/818，共319帧；825未参与训练 |
| 每步loss | 3.435191、2.163007、0.895386、2.109674、1.923499 |
| PyTorch训练峰值allocated | 约1.0761GiB；不是nvidia-smi显示的全部显存，也不包含后续所有推理峰值 |
| 保存与归档 | checkpoints/000005，COMPLETE.json含15个文件的SHA256清单 |
| 保存内容 | adapter/config、processor与归一化统计、训练配置、optimizer/scheduler/RNG/step |
| 固定batch/噪声重载 | 动作块最大误差0；后处理首动作最大误差0 |
| 训练续接 | 状态已保存，尚未实际resume验证 |
| 独立离线评测 | episode825的三个时刻，base MAE1.066619→LoRA1.061330，非任务成功率 |
| LIBERO执行/MP4 | 接入已补，完整GPU闭环待实测 |

用户单步与官方5步分别从初始策略加载，不是合计6步的同一条训练轨迹。
5步、batch_size=1只使用5个batch，尚未遍历319个训练样本一遍。
每步观测、噪声和Flow Matching时间可能不同，首尾loss差不能直接作为训练收益，更不能替代任务成功率。
本次短跑的scheduler将衰减周期缩到5步，最后学习率为2.5e-6；延长训练前需明确恢复/重设学习率的策略，不能盲目续接。

实际处理后图像为[1,1,3,256,256]，状态[1,1,8]，动作[1,50,7]，语言[1,48]。
图像/状态的第二个1是n_obs_steps=1的观测轴，不应误认为有多帧历史。
首个batch来自episode818、frame63、任务24。图像范围[0,1]，state/action已标准化。

检查发现：最初PNG导出仅处理4维图像，因此已有inspect目录没有PNG；两张图仍完整保存在first_processed_batch.pt。
已修复导出函数兼容4/5维，并用从Drive取回的真实batch在本地CPU补导，不要求重跑GPU。
原始证据、batch和PNG保存在outputs/training_audit_20261105/，不包含在公开Git产物中。

## 当前收束与转向Corrector

独立离线评分已完成：707个有效动作元素，MAE降低约0.50%，末帧略升。它验证了评测链路，不能证明任务成功率提高。
训练基础验收至此收束，不再为五步adapter追加训练或等待一条成功视频。
新主线见[Corrector机制实践与用户实现入口](corrector_mechanism_practice.md)，使用原先smolvla_libero冻结策略与LIBERO-plus对照，避免混入本次新base。

### 本次T4闭环入口实际失败在哪里

Notebook的libero-prepare输出表明，依赖安装和pip check通过，但check-sim导入libero.libero失败。
错误是ModuleNotFoundError: No module named 'libero'；尚未加载完整VLA，因此没有该checkpoint的执行视频/成功率。
固定原始LIBERO外层目录没有__init__.py，setup.py使用find_packages，本地实际结果为空列表；安装元信息存在不保证源码可导入。
此前CPU检查显式配置源码路径，漏测了远程editable安装分支，这一遗漏已补回归验证。

已为独立sim-venv注册专用libero_source_root.pth，增加repair-sim：只修复路径、核对版本/pip check、真实渲染复查，不重装、不训练。
本地实际CLI复查通过：两次新环境初始完整状态配对、reset/step、两路图像、MP4编码/解码。
日志在outputs/training_audit_20261105/libero_cpu_probe/repair-cli.log；29项CPU检查通过，包括干净解释器中“有安装元信息但不能import”的回归用例。
这仍是CPU仿真管线验证，不冒充远程T4完整VLA评测通过。
Notebook已同步新脚本并保留旧失败输出；下次先运行首格同步代码，再运行仿真准备，它会对已有环境选择repair-sim。
旧LoRA闭环评测入口保留为可选项，不再阻塞Corrector学习与实验。

## 云盘证据

- 用户单步证据（私人链接不公开，原始记录保存在私有归档）
- 官方五步证据（私人链接不公开，原始记录保存在私有归档）
- 重载证据（私人链接不公开，原始记录保存在私有归档）
- checkpoint目录（私人链接不公开，原始记录保存在私有归档）
