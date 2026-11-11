"""从已掌握的train_one_batch迁移到Corrector，供实践入口真实调用。"""
from .losses import corrector_loss


def train_corrector_batch(model, batch, optimizer, beta=1.0):
    """练习E：一次更新，返回Python float的total loss。

    batch已有z_t、a_t、delta_z，且已在正确设备；这里不加载模型或数据。
    1. train模式，清梯度。
    2. model(z_t,a_t)前向。
    3. corrector_loss(pred,delta_z,beta=beta)取total。
    4. backward、optimizer.step()。
    5. 返回与计算图分离的float日志。
    已写过单batch训练，这是接口迁移练习，不需要重写训练平台。
    """
    model.train()
    optimizer.zero_grad()
    pred = model(batch["z_t"], batch["a_t"])
    total_loss, mse_loss, cosine_loss = corrector_loss(pred, batch["delta_z"], loss_type="both", beta=beta)
    total_loss.backward()
    optimizer.step()
    return total_loss.item()
