"""用户亲写的Corrector时序配对；由scripts/corrector_pair_audit.py实际调用。"""

from numbers import Integral


def make_transition_pairs(episode_index, frame_index, k_step=1):
    """返回[(当前行号, 未来行号), ...]，不是episode内部的frame编号。

    两个输入为等长整数列表；同一episode连续排列，内部frame逐个递增且无漏帧。
    一对必须属于同一episode，且未来frame恰好比当前frame大k_step。
    不足k_step的轨迹尾部不能配对；k_step必须为正整数。
    示例：[813,813,818,818]、[0,1,0,1]、k=1 → [(0,1),(2,3)]。
    这里只有索引，不提视觉特征、不训练网络。监督标签随后才由z[j]-z[i]构造。
    """
    result = []
    # 1. 检查输入长度和k_step。
    length_episode = len(episode_index)
    length_frame = len(frame_index)
    if length_episode != length_frame:
        raise ValueError("episode_index和frame_index长度不一致")
    if isinstance(k_step, bool) or not isinstance(k_step, Integral):
        raise TypeError("k_step必须为正整数，不能是bool或小数")
    if k_step <= 0:
        raise ValueError("k_step必须为正整数")
    # i+k是行号，只有连续帧缓存中才等价于未来第k帧；遇到漏帧先报错。
    seen_episodes = set()
    for i in range(length_episode):
        if i == 0 or episode_index[i] != episode_index[i - 1]:
            if episode_index[i] in seen_episodes:
                raise ValueError("同一episode必须连续排列")
            seen_episodes.add(episode_index[i])
        elif frame_index[i] != frame_index[i - 1] + 1:
            raise ValueError("当前实现要求episode内frame连续；不能将行距当作帧距")
    # 2. 遍历候选行，检查episode相同、frame相差k_step。
    for i in range(length_episode):
        if (
            i + k_step < length_episode
            and episode_index[i] == episode_index[i + k_step]
            and frame_index[i + k_step] - frame_index[i] == k_step
        ):
            result.append((i, i + k_step))
    # 3. 收集并返回行号对。你可以先使用熟悉的for循环，不要求向量化。
    return result
