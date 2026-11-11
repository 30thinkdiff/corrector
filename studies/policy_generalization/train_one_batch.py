"""你的单batch训练练习；由workbench的--practice-step-file显式调用。

输入：已加载的LoRA策略、已经预处理并移到正确设备的batch、已配置的optimizer。
SmolVLA的policy(batch)返回(loss, loss_dict)。返回Python float形式的loss。
这里不再加载模型、不重复归一化、不调用其他训练器。
"""


def train_one_batch(policy, batch, optimizer):
    # TODO 1：设置训练模式。
    policy.train()
    # TODO 2：清理上一步梯度。
    optimizer.zero_grad()
    # TODO 3：前向并取得loss。
    loss, loss_dict = policy(batch)
    # TODO 4：反向传播。
    loss.backward()
    # TODO 5：更新参数。
    optimizer.step()
    # TODO 6：返回与计算图分离的数值loss。
    return loss.detach().item()
