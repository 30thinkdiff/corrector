import os
from pathlib import Path

from torchvision.utils import save_image

from brightness import apply_brightness
from lerobot.scripts import lerobot_eval

FACTOR = float(os.environ.get("BRIGHTNESS_FACTOR", "1.0"))
DEBUG_DIR = Path(os.environ.get("DEBUG_DIR", "outputs/obs_debug"))

if FACTOR <= 0:
    raise ValueError("Brightness factor must be positive")

# 保存官方函数，后面先按官方方式处理观察
original_preprocess = lerobot_eval.preprocess_observation

first_observation = True

def preprocess_with_brightness(raw_observation):
    global first_observation

    #1.官方处理：uint8 → float，BHWC → BCHW
    observation = original_preprocess(raw_observation)

    #2. 找出全部相机图像
    image_keys = [
        key for key in observation
        if key.startswith("observation.images.")
    ]

    if not image_keys:
        raise RuntimeError("No image features found")

    #3.修改图像，但不修改其他观察
    modified = apply_brightness(observation, FACTOR)

    #4.只在第一次观察时验证并保存图像
    if first_observation:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        for key in image_keys:
            name = key.replace(".", "_")

            before = observation[key][0].detach().cpu()
            after = modified[key][0].detach().cpu()

            save_image(before, DEBUG_DIR / f"{name}_before.png")
            save_image(after, DEBUG_DIR / f"{name}_after.png")

            print(
                f"{key}:shape={tuple(before.shape)},"
                f"mean_before={before.mean().item():.4f},"
                f"mean_after={after.mean().item():.4f}"
            )

        print(f"Brightness factor: {FACTOR}")
        first_observation = False

    return modified

# 只修改当前 Python 进程中的评测函数引用
lerobot_eval.preprocess_observation = preprocess_with_brightness

if __name__ == "__main__":
    lerobot_eval.main()