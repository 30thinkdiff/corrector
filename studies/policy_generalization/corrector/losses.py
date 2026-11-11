"""用户实现视觉变化监督；来源为参考train.py的_combined_loss。"""
import torch
import torch.nn.functional as F


def corrector_loss(pred, target, loss_type="both", beta=1.0):
    """练习C：返回(total, mse, cosine_loss)，均为标量Tensor。

    pred/target：[B,L,D]。mse使用所有标量的mean。
    cosine_loss：先将每个样本展平为[B,L*D]，再沿最后一维计算相似度，最后对batch取mean。
    mse/cosine/both三种loss_type；both= mse + beta*cosine_loss。
    可用：F.mse_loss、reshape、F.cosine_similarity、mean。
    不要item()：训练要对total反向，数值日志由外层取。
    """
    if pred.shape != target.shape or pred.ndim != 3 or pred.numel() == 0:
        raise ValueError("需要相同、非空的[B,L,D]")
    if not pred.is_floating_point() or not target.is_floating_point() or pred.device != target.device:
        raise ValueError("需要同设备的浮点Tensor")
    if loss_type not in {"mse", "cosine", "both"} or beta < 0:
        raise ValueError("loss_type须为mse/cosine/both，beta非负")
    mse = F.mse_loss(pred, target)
    # reshape(-1)与flatten()默认均展平全部轴；这里要保留batch轴，逐样本比较方向。
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    cosine_loss = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
    if loss_type == "mse":
        total = mse
    elif loss_type == "cosine":
        total = cosine_loss
    else:
        total = mse + beta * cosine_loss
    return total, mse, cosine_loss
