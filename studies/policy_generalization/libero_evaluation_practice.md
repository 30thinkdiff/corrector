# 保存的SmolVLA checkpoint怎样进入LIBERO评测

> 2026-11-05更新：训练练习已收束，本入口作为可选工程验证保留。当前研究转入[Corrector机制实践](corrector_mechanism_practice.md)，不要求先把五步LoRA跑成成功策略。

## 本次No module named libero与修复

远程安装完成、pip check通过，随后首次导入libero.libero失败，尚未进入VLA执行。
原始仓库外层libero没有__init__.py，其setup.py的find_packages在本地返回空列表；editable安装成功不代表正确注册了源码根目录。
此前本地CPU检查手工配置了源码路径，漏测此分支。现已用干净解释器复现并验证路径注册修复。

运行更新Notebook的首格同步代码，然后运行本节仿真准备格即可：已有sim-venv、源码和配置时选择repair-sim，
只注册专用.pth、核对固定依赖、运行pip check并做真实场景检查，不重复安装或训练；新会话则选择prepare-sim。
本地repair-sim CLI已实测通过，日志为outputs/training_audit_20261105/libero_cpu_probe/repair-cli.log。
Notebook保留旧报错输出，更新代码尚未在你的远程T4重新执行。场景检查视频也不能当成VLA执行视频。

以下是可选闭环评测的原操作说明。

本次真实训练和离线预测已完成，[CPU评分记录](offline_eval_results_20261105.md)显示三个验证时刻总体MAE小幅下降。
现在新增的是**闭环执行入口**：加载保存的adapter与processor，在原始LIBERO任务运行并录像。
默认两种条件各一个回合，仅检查协议与行为，不能据此声称统计显著的性能改善。

## 下一次连接T4怎么执行

不要“全部运行”：旧训练/预测单元格已有结果，会主动拒绝覆盖。

1. 运行Notebook最前面的准备单元格。新运行时会准备训练依赖/资产，已有运行时只同步代码。
2. 运行新增“LIBERO环境准备”单元格。独立sim-venv借用训练依赖，安装固定仿真包；不改训练venv。
   它先创建配置文件，避免LIBERO首次导入的input/EOFError；随后真实reset、保存两路图片、编码并解码两帧MP4。
   这一步没有加载VLA。场景/图形失败应在这里定位，不要靠重训排查。
3. 准备通过后，运行“LIBERO成对执行与录像”单元格。默认base/LoRA各一回合，最多280步；成功和失败都保存MP4。
   使用新RUN_NAME，已有实验目录不覆盖。下一轮才把EPISODES增加到3或更多，以相同seed配对比较。
4. 每回合结果、第一份实际策略输入、初始完整仿真状态、输入图片和视频直接写入挂载Drive。
   查看eval_info.json中的complete；断连时未完成结果保留，不能把缺失回合当作失败或成功计入完整对照。

本地已在独立Python3.12 CPU环境验证：原始任务匹配、真实reset/step、两次独立初始化的完整sim state一致，
两路图像/8维状态与20FPS MP4；额外用固定无操作策略跑通官方eval_policy的2步回合，reset调用1次、select_action调用2次，
超时结束、success读取和2帧20FPS视频均通过。这是实际物理环境与评测管线检查，**没有运行完整SmolVLA**。
585个场景资产文件逐一与固定提交的Git blob哈希核对一致。本地实测发现BDDL未声明完整的future依赖，已补入准备脚本。
完整模型闭环仍需远程T4实测；CPU检查不能冒充GPU评测已完成。
单次回合结果是环境提供的success，成功率是成功回合数/实际计划且完成的回合数，不由视频观感或训练loss推断。

CPU检查产物：outputs/training_audit_20261105/libero_cpu_probe/，含实际输入PNG、render_probe.mp4、
render_check.json、noop_rollout/verification.json和完整安装/运行日志；无操作回合的0%不是LoRA成绩。
上游LIBERO导入旧Gym的提示仍保留；本入口实际使用Gymnasium向量环境，已验证reset/step，没有用全局屏蔽隐藏错误。

额外将验证数据825/frame0与CPU仿真输入作了图像对照：相机朝向/布局相近，但地板与背景亮度有可见差异。
两者不是同一初始状态；尚不能确定来自渲染版本、灯光还是数据生成配置，不擅自改纹理或归一化来“调成一样”。
对照页面为CPU检查目录的camera_comparison.html。T4准备单元格会显示本次实际相机输入，先核对它们，再解释执行成绩。

## 对齐了哪些东西

| 项目 | 本次协议 |
| --- | --- |
| 策略 | 官方五步checkpoint的同一冻结base，LoRA禁用/启用；不使用旧smolvla_libero |
| 仿真 | 官方原始LIBERO固定提交8f1084e，不引入Plus的view后缀 |
| 任务 | 按语言精确匹配alphabet soup放篮子；数据task_index=24不是suite的task_id |
| 相机 | agentview→image，手腕→image2，256×256；环境processor翻转一次，保持数据约定 |
| 状态 | 末端位置3+轴角3+夹爪位置2，共8维；不是7维关节角 |
| 处理器 | 直接加载checkpoint保存的归一化统计与tokenizer；不临时拿验证集重算统计 |
| 动作 | 反归一化后的7维相对末端控制量进入env.step；不是把标准化动作直接给模拟器 |
| 初始条件 | 同一任务、初始状态序列、seed；比较两个条件首次输入前的完整sim state |
| 缓存 | 每回合官方rollout调用policy.reset清空动作队列；队列仍控制何时重新预测 |
| 超时 | 实例级TimeLimit让达到280步正确truncated，避免没有done时的聚合歧义 |
| 频率 | 显式20Hz控制、20FPS录像；修正类默认录像80FPS，不改安装包源码 |

每回合记录control_steps，并核对官方录像帧数与实际控制步数一致；额外保存final_frame.png，方便检查最后状态。

**数据10FPS与控制20Hz的关系尚未由转换脚本确认。**本次20Hz是明确记录的初始执行协议，沿用此前仿真设置，
不等于证明与示范时间语义完全一致。不要因为看到10就把视频改10FPS而保持20Hz控制，那只会改变播放速度。
执行/抓取行为检查后，若确认示范动作转换需要其他控制频率，再单独设计对照；不无声插值、重复或缩放动作。

初始噪声：两个条件每回合重设同一随机seed；闭环轨迹分叉后，观测和预测次数可能不同，
因此不能声称后面每个动作块都严格共享同一个噪声Tensor。这与之前固定观测的离线成对预测不同。

## 核心代码只看这条链，不必再读整个准备脚本

官方rollout已经实现下列逻辑，本项目调用它，而不是让你重新写一套控制循环：

```text
raw observation（像素uint8、嵌套机器人状态）
  → preprocess_observation：转Tensor、图像[0,1]、整理字段
  → env_preprocessor：图像翻转、状态拼接
  → checkpoint preprocessor：状态标准化、语言tokenize、设备移动
  → policy.select_action：队列有动作则取出，否则生成动作块
  → checkpoint postprocessor：动作反归一化
  → env_postprocessor → env.step(action_numpy)
  → 新观测、reward、terminated/truncated、is_success
```

`select_action()`每步返回[1,7]，并不代表每步都重新预测整段动作。
`eval_policy()`负责reset、循环、收集环境success、录像；本项目负责加载版本、成对条件和结果归档。
禁用adapter的作用域包住一整个回合，不能在动作队列还留有旧条件动作时切换LoRA。

你刚写的指标已经有实际用途：CPU的score-eval确实调用它。其关键索引过程是：
`is_pad [B,H] → ~is_pad → unsqueeze(-1) [B,H,1] → expand_as [B,H,D]`；
布尔索引选出一维有效误差，再mean。本次选中707个数，排除了49×7个末帧补齐数。

## 留给你掌握的下一小段核心

第一次观看两个条件视频后，先核对首次抓取和搬运，不能只看最终success。
然后用本地eval_info.json，按seed取出base和trained状态，先检查每对是否齐全，
再统计“都成功、都失败、仅trained成功、仅base成功”四种情况。
等真实结果产生后，我们明确输入与预期输出，再由你写这个短的CPU分析函数；当前不新增待填脚本，也没有暗中调用尚未实现的方法。
这项练习不挡GPU实验。我先审查你的实现，再按错误给提示。
“仅trained成功”与“仅base成功”比把两个小样本成功率机械相减更便于追踪具体案例。
环境与日志代码由助手维护，你重点掌握输入/输出语义、对照条件、指标和失败定位。

源码参考：[官方LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)。
本项目入口是scripts/smolvla_libero_tools.py；实际控制循环仍在固定LeRobot的lerobot_eval.rollout中。
