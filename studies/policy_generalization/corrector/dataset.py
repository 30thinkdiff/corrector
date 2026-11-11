"""读取本项目逐episode特征缓存；IO/配对已接入，getitem由用户实现。"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .transitions import make_transition_pairs


class CachedTransitionDataset(Dataset):
    def __init__(self, directory, k_step=1):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "metadata.json").read_text())
        self.z_q = np.load(self.directory / "z_q.npy", mmap_mode="r")
        self.z_scale = np.load(self.directory / "z_scale.npy", mmap_mode="r")
        self.actions = np.load(self.directory / "actions.npy", mmap_mode="r")
        self.episode_index = np.load(self.directory / "episode_index.npy", mmap_mode="r")
        self.frame_index = np.load(self.directory / "frame_index.npy", mmap_mode="r")
        n = len(self.episode_index)
        if (self.z_q.ndim != 3 or self.z_scale.shape != self.z_q.shape[:2]
                or self.z_q.shape[0] != n or self.actions.ndim != 2 or len(self.actions) != n
                or self.frame_index.shape != (n,)):
            raise ValueError("缓存各数组的帧数/形状不一致")
        self.pairs = make_transition_pairs(self.episode_index.tolist(), self.frame_index.tolist(), k_step)
        if not self.pairs:
            raise ValueError("没有可用时序对；检查轨迹长度和k")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        """练习D：返回z_t[L,D]、a_t[A]、delta_z[L,D]的float32 Tensor字典。

        1. 从self.pairs[index]取当前/未来行号。
        2. 反量化两帧：int8特征转float32，乘该帧scale[:,None]。
        3. 读取当前行action并复制为float32；不要原地修改只读mmap。
        4. 未来特征减当前特征作为delta_z；转换为Tensor并返回。
        本实践h_window=1，不要求构造z_hist，也不要求GPU移动。
        """
        t_now, t_future = self.pairs[index]
        # scale[L]→[L,1]，每个token的一个缩放系数广播到它的D个特征。
        z_now = self.z_q[t_now].astype(np.float32) * self.z_scale[t_now][:, None]
        z_future = self.z_q[t_future].astype(np.float32) * self.z_scale[t_future][:, None]
        a_now = self.actions[t_now].astype(np.float32)
        delta_z = z_future - z_now
        return {
            "z_t": torch.from_numpy(z_now),
            "a_t": torch.from_numpy(a_now),
            "delta_z": torch.from_numpy(delta_z)
        }


def build_episode_splits(feature_root, k_step=1):
    """按metadata.split组合轨迹，不在相邻时序对上random_split。

    调用者先用数据工作台verify校验COMPLETE与SHA，再把本轮选定轨迹缓存复制到临时本地盘。
    """
    groups = {"train": [], "validation": []}
    episodes = {"train": set(), "validation": set()}
    for path in sorted(Path(feature_root).glob("episode_*")):
        if not (path / "COMPLETE.json").is_file():
            continue
        ds = CachedTransitionDataset(path, k_step)
        split = ds.meta.get("split")
        if split not in groups:
            raise ValueError(f"未知split：{split}")
        groups[split].append(ds)
        episodes[split].update(ds.episode_index.tolist())
    if episodes["train"] & episodes["validation"]:
        raise ValueError("训练和验证存在相同episode")
    if any(not v for v in groups.values()):
        raise ValueError("至少需要训练和验证轨迹各一条")
    return {key: ConcatDataset(value) for key, value in groups.items()}
