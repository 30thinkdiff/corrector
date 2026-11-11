import json
import math

import torch


def summarize_tensor(name: str, tensor: torch.Tensor, *, verbose: bool = True):
    """查看一个真实数值 Tensor；返回可直接写入 JSON 的摘要。"""
    # 1. 检查输入是不是 Tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} is not a torch.Tensor, got {type(tensor)}")
    if tensor.is_complex():
        raise TypeError("This summary supports real-valued and bool tensors")
    # 2. 取得 shape、dtype、device
    shape = list(tensor.shape)
    dtype = tensor.dtype
    device = tensor.device

    # 3. detach 后展平，提取前几个数
    detached = tensor.detach()
    flat_tensor = detached.flatten()

    # tolist() 也支持 bfloat16；numpy() 不能直接转换这种类型。
    # JSON 不支持 NaN/Inf 数字，用字符串保留它们的含义。
    def json_value(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    first_few = [json_value(value) for value in flat_tensor[:5].cpu().tolist()]

    # 4. 对浮点 Tensor 检查 NaN/Inf、min/max
    #    注意空 Tensor 和 bool Tensor 的情况
    has_nan = bool(torch.isnan(detached).any().item()) if tensor.is_floating_point() else False
    has_inf = bool(torch.isinf(detached).any().item()) if tensor.is_floating_point() else False
    min_val = json_value(detached.min().item()) if tensor.numel() else None
    max_val = json_value(detached.max().item()) if tensor.numel() else None

    # 5. 返回一个字典，或打印结构化结果
    summary = {
        "name": name,
        "shape": shape,
        "dtype": str(dtype),
        "device": str(device),
        "first_few": first_few,
        "numel": tensor.numel(),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": min_val,
        "max": max_val,
    }
    true_num = int(detached.sum().item()) if tensor.dtype == torch.bool else None
    false_num = tensor.numel() - true_num if true_num is not None else None
    true_fraction = true_num / tensor.numel() if true_num is not None and tensor.numel() > 0 else None
    if tensor.dtype == torch.bool:
        summary.update({
            "true_count": true_num,
            "false_count": false_num,
            "true_fraction": true_fraction,
        })
    if verbose:
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    # 直接运行本文件才执行示例；被 Trace 脚本导入时不执行。
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([1.0, float("nan")])
    z = torch.tensor([1, 2, 3], dtype=torch.int32)
    summarize_tensor("x", x)
    summarize_tensor("y", y)
    summarize_tensor("z", z)
