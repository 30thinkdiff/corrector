"""用户实现的离线动作指标；由check-metric/score-eval显式调用。"""

import torch


def masked_action_mae(pred, target, is_pad):
    """返回有效动作元素上的平均绝对误差，类型为Python float。

    pred/target: 同形状浮点Tensor [B,H,D]，本次真实D=7。
    is_pad: bool Tensor [B,H]；True代表补齐位置，必须排除。
    按所有有效动作元素求平均，不能先对各样本平均再等权平均。
    全部为padding、形状不匹配或mask非bool时，抛出ValueError/TypeError。
    本次比较标准化动作，不在此函数中反归一化或加载模型。
    """
    # 1：检查输入形状、mask类型，以及是否存在有效位置。
    if not all(isinstance(value, torch.Tensor) for value in (pred, target, is_pad)):
        raise TypeError("pred/target/is_pad must be torch.Tensor")
    if pred.ndim != 3 or target.ndim != 3 or is_pad.ndim != 2:
        raise ValueError(f"pred/target/is_pad must be (B,H,D) or (B,H), got {tuple(pred.shape)}/{tuple(target.shape)}/{tuple(is_pad.shape)}")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target must have same shape, got {tuple(pred.shape)}/{tuple(target.shape)}")
    if is_pad.shape != pred.shape[:2]:
        raise ValueError(f"is_pad must match (B,H)={tuple(pred.shape[:2])}, got {tuple(is_pad.shape)}")
    if pred.numel() == 0:
        raise ValueError("pred/target must contain action elements")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred/target must be floating-point tensors")
    if pred.device != target.device or pred.device != is_pad.device:
        raise ValueError("pred/target/is_pad must be on the same device")
    if is_pad.dtype != torch.bool:
        raise TypeError(f"is_pad must be bool, got {is_pad.dtype}")
    if is_pad.all():
        raise ValueError("is_pad must have at least one False value")
    # 2：计算绝对误差，并排除补齐位置。保留用户实现。
    valid_mask = ~is_pad  # True for valid positions
    abs_error = torch.abs(pred - target)  # (B,H,D)
    valid_abs_error = abs_error[valid_mask.unsqueeze(-1).expand_as(abs_error)]
    if not torch.isfinite(valid_abs_error).all():
        raise ValueError("valid action errors contain NaN or Inf")
    # 3：对有效动作元素求平均，返回Python float。
    mean_abs_error = valid_abs_error.mean().item()
    return mean_abs_error
