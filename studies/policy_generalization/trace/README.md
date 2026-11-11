# SmolVLA：本地开发，Colab 记录第一次真实推理

目标是把真实 Tensor 对应到源码方法。记录器复用官方 `lerobot_eval.main()`，
不复制模型、不改变像素、不额外运行一次模型、不重采样噪声，也不修改安装包。
当前只支持一个策略实例、顺序评测；先用一个 LIBERO 任务、一个环境、一个 episode。

## 先理解三个文件

1. `tensor_summary.py`：你写的摘要函数，保留了取 shape、detach、取值和范围检查的结构。
   现在支持 bfloat16、空张量和严格 JSON；导入时不执行示例。
2. `policy_trace.py`：在真实 Policy 实例上安装 wrapper。先读
   `_observe_output()`：调用原方法 → 记录摘要 → 返回原对象。
   再读 `select_wrapper()`：只在第一次实际生成时开启记录。
3. `eval_with_trace.py`：暂时包装官方 `make_policy()` 与处理器工厂，
   拿到实例后安装记录器，随后执行原来的官方评测入口；`finally` 恢复包装和 hook。

普通 Python 方法不能使用 `register_forward_hook()`；本次采用方法 wrapper。
预处理／后处理使用官方 Pipeline 的 hook 列表，返回的仍是原来的 Pipeline 对象。

## 运行前先填写预期

实际参数以本次加载的 checkpoint 为准；表中数字是既有 LIBERO 实验的待验证预期。

| 记录位置 | 预期 | 运行后要解释什么 |
| --- | --- | --- |
| `preprocessor.input` 中的状态 | LIBERO 特征通常8维 | 这是策略归一化之前的环境状态特征 |
| `policy_input.observation.images.*` | 每路通常 `[1,3,256,256]`、`[0,1]` | 已经过官方处理器，是实际交给策略的图像 |
| `policy_input.observation.state` | 通常 `[1,8]` | 已经过策略预处理器，通常已归一化 |
| `prepare_images.output.0.*` | 每路通常 `[1,3,512,512]`、`[-1,1]` | 从配置确认 Resize，检查范围变化 |
| `prepare_images.output.1.*` | 每路 `[1]` bool | 相机级有效性标记 |
| `prepare_state.output` | 通常 `[1,32]` | 尾部是否补零 |
| `visual_tokens.0`、`.1` | 读取真实 shape，不猜 | 每台相机编码后的 token 数与特征宽度 |
| `embed_prefix.output.0/.1/.2` | 特征／有效性／注意力分组 | token 数以真实语言长度为准 |
| `embed_suffix.first_output.0` | 第一积分步的动作特征 | 整块动作有多少行，每行多少特征 |
| `initial_noise` | 通常 `[1,50,32]` | 原始噪声只在首次生成时保存一次 |
| `sample_actions.output` | 通常 `[1,50,32]` | 模型统一动作维度，仍在归一化空间 |
| `action_chunk` | 通常 `[1,50,7]` | 裁掉统一维度中的额外部分 |
| `select_action.output` | 通常 `[1,7]` | 队列第一次返回动作块第0行 |
| `postprocessor.output` 中的 action | 通常 `[1,7]` | 后处理后的动作，尚未经过环境后处理 |

`preprocessor.input` 和 `postprocessor.output` 是处理器 hook 看到的 transition；
其内部字段名称由当前 LeRobot 的 TransitionKey 定义，以实际 JSON 为准。
记录器也会保存语言 IDs 和 mask，但不额外运行 tokenizer。
PNG 仅展示 batch 中第0个样本，完整张量保存在 `.pt` 中。

## 本地可以验证什么

在安装了 CPU PyTorch 和 Pillow 的环境中，从 robot-lab 根目录运行：

```bash
python studies/policy_generalization/trace/tensor_summary.py
python -m unittest discover -s studies/policy_generalization/trace -p 'test_trace.py' -v
python studies/policy_generalization/trace/eval_with_trace.py --check-env
EVAL_DRIVER=trace DRY_RUN=1 bash scripts/eval_libero.sh
```

本地测试验证摘要、噪声随机状态不被记录器改变、队列行为、返回值和异常后的清理。
测试使用明确的模型替身，不等于成功运行完整 SmolVLA／LIBERO。
`--check-env` 只检查环境，不下载 checkpoint，不要求 CUDA 或完整模型依赖。

## 在已有 Colab 环境中运行

`scripts/lerobot.ipynb` 末尾已追加 Trace 单元，原有单元保持不变。
优先复用你已跑通的依赖与 LeRobot 源码，不为本次 Trace 主动升级整套环境。
Notebook 中先设置项目路径，确保新 Trace 文件和本地开发代码已同步到 Colab。

特别注意：Notebook 内修改 `sys.path` 不会传给 `!python` 或 `subprocess` 启动的新进程。
新增单元显式传入 `PYTHONPATH`，并使用 `sys.executable` 启动同一 Python 环境。
Notebook 目前固定了 `ae2b57f`，本地包报告为0.6.1，不能仅凭版本名假定两者源码相同。
这两个实现的关键包装接口已检查；每次仍记录实际导入路径与 Git commit。

终端中也可以沿用现有脚本：

```bash
EVAL_DRIVER=trace RUN_NAME=trace_day2_001 N_EPISODES=1 \
  bash scripts/eval_libero.sh
```

Colab 使用 shell 时同样要确保 `python` 指向预期环境，且 LeRobot 源码目录在
`PYTHONPATH` 中。默认 Trace 目录是本次评测输出下的 `trace/`，可用 `TRACE_DIR` 改变。
重复运行应换一个实验目录；已有 `trace.json` 时记录器拒绝覆盖。

## 如何读结果

```text
trace/
  trace.json                       环境、真实配置、Tensor摘要、前三次队列长度
  policy_input.pt                  策略第一次生成所用的预处理后输入Tensor
  initial_noise.pt                 此次实际采样的初始噪声
  action_chunk.pt                  裁剪后的完整动作块，通常最后一维7
  selected_action.pt               首次返回的单步动作
  observation_images_*_batch0.png  策略输入图像；只为值域确认为[0,1]的RGB图保存
```

如果采用已有动作队列、显式外部噪声或不同策略扩展，记录项可能不同；当前入口通过
官方 rollout 重置策略，默认不传显式噪声，因此会进入首次采样路径。
`complete=true` 表示捕获了动作块与首次返回动作；若完整评测后来失败，另有 `error`
字段，不能把它解释成整个 episode 成功。hook 不可用时不会伪造处理器记录。

优先完成三项核对：

1. 比较 `prepare_images` 前后的 shape、范围，与实际配置对应。
2. 用 `torch.testing.assert_close(chunk[:, 0], selected_action)` 验证第一次取队列。
3. 检查 `queue_steps`；若 `n_action_steps>=3`，前三次通常只有第一次生成动作块。

所有 `.pt` 只从自己本次实验的目录读取：

```python
chunk = torch.load(trace_dir / "action_chunk.pt", map_location="cpu", weights_only=True)
action = torch.load(trace_dir / "selected_action.pt", map_location="cpu", weights_only=True)
torch.testing.assert_close(chunk[:, 0], action)
```

摘要计算会同步 GPU，Trace 时间不能用来衡量纯模型推理性能。
完整运行仍需在 Colab 上验证；没有真实运行结果前，不把表中的预期数字写成结论。

## 你负责的短练习

先不读完全部包装代码，集中练三个小任务：

1. 在运行前预测图像、状态、动作块和单步动作的 shape，写下理由。
2. 自己用摘要中的 shape 写一个断言，检查动作块和单步动作最后一维一致。
3. 解释 `_observe_output()` 中为什么必须 `return result`，而不能再次调用原方法。

先完成单次真实 Trace。亮度配对、K/V 可视化和训练不放进本轮入口。
