
import torch


def apply_brightness(observation: dict, factor: float) -> dict:
    """仅修改观察字典中的 RGB 图像，不改变机器人状态。"""

    if factor <= 0:
        raise ValueError("Brightness factor must be positive.")

    result = dict(observation)

    for key, value in observation.items():
        if not key.startswith("observation.images."):
            continue

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a Tensor")

        if not value.is_floating_point():
            raise TypeError(f"{key} must be a floating-point image")

        result[key] = (value * factor).clamp(0.0, 1.0)

    return result
