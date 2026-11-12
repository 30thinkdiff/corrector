"""用户完成的LVM/OGG核心练习；不等同于已接入的在线算法。

上游参考：lerobot_eval_modified_detection.py的到期比较/robust_threshold_step，
以及smolvla_modified/modeling_smolvla_modified.py的sample_actions引导分支。
"""
import numpy as np
import torch
import torch.nn.functional as F


def visual_mismatch(predicted_delta, observed_delta):
    """练习L1：[B,L,D]和[B,L,D]→每样本误差[B]。

    先各自reshape(B,-1)，再计算1-F.cosine_similarity(...,dim=-1)。
    observed_delta只能在t+k观测到达后用z_(t+k)-z_t构造，不得提前读取未来帧。
    本方法只计算分数，不决定是否中止、不对batch取平均。
    """
    if predicted_delta.shape != observed_delta.shape or predicted_delta.ndim != 3:
        raise ValueError("需要同形状的[B,L,D]")
    if predicted_delta.numel() == 0 or not predicted_delta.is_floating_point() or not observed_delta.is_floating_point():
        raise ValueError("需要非空浮点特征")
    predicted_flat = predicted_delta.reshape(predicted_delta.shape[0], -1)
    observed_flat = observed_delta.reshape(observed_delta.shape[0], -1)
    cosine_sim = F.cosine_similarity(predicted_flat, observed_flat, dim=-1)
    # 零变化向量没有可定义的方向；PyTorch的eps使分数为1，不代表已确认失败。
    return 1 - cosine_sim  # 返回每个样本的视觉变化不一致分数，不聚合batch。


def robust_limits(history, k_on=3., k_off=2., noise_floor=.25):
    """练习L2：历史误差列表→(T_on,T_off)两个float。

    np.median求中位数；MAD=median(abs(history-median))，再max(MAD,1e-6)。
    两个阈值分别max(noise_floor,median+k*MAD)。
    这只是源码状态机的阈值核心；尚未包含EWMA/bootstrap/持续计数/冷却。
    正式状态机需与固定上游robust_threshold_step逐项对齐，不能冒充完成整个LVM。
    """
    history = np.asarray(history, dtype=np.float64)
    if (history.ndim != 1 or history.size == 0 or not np.isfinite(history).all()
            or not np.isfinite([k_on, k_off, noise_floor]).all()
            or k_on <= k_off or k_off < 0 or noise_floor < 0):
        raise ValueError("需要非空历史、k_on>k_off及非负floor")
    median = np.median(history)
    mad = np.median(np.abs(history - median))
    mad = max(mad, 1e-6)
    T_on = max(noise_floor, median + k_on * mad)
    T_off = max(noise_floor, median + k_off * mad)
    return float(T_on), float(T_off)


def guided_velocity(predictor, z_t, x_tau, velocity, timestep, target_delta, eta=1.):
    """练习G1：冻结预测器，通过候选动作影响视觉残差，计算速度引导。

    z_t/target_delta：[B,L,D]；x_tau/velocity：[B,H,A_pad]；timestep：[B]。
    predictor是可微Torch模块（不是返回NumPy的predict_delta_z封装）。
    已在eval且参数requires_grad=False，但仍应保留对输入动作的梯度。
    返回(v_guided, loss_float)，v_guided同velocity形状并已detach。

    在torch.enable_grad()内：
      1. velocity.detach().clone().requires_grad_(True)建立独立速度叶子v。
      2. clean=x_tau-timestep[:,None,None]*v，取第0个动作的前predictor.cfg.action_dim维。
      3. predictor(z_t,action)预测delta；与target_delta按样本展平算平均余弦损失。
      4. torch.autograd.grad(loss,v)取引导梯度；v-eta*gradient，最后detach。
    target_delta由外部构造为期望变化减实际偏离；此处不训练预测器或VLA。
    tau是Flow Matching生成进度，H是未来动作位置，两种时间不能混淆。
    """
    if x_tau.shape != velocity.shape or velocity.ndim != 3 or not np.isfinite(eta) or eta < 0:
        raise ValueError("需要相同[B,H,A_pad]及非负eta")
    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError("调用前执行predictor.eval().requires_grad_(False)，本函数不隐式修改模型")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("OGG需要输入梯度；使用no_grad包围策略，不要使用inference_mode")
    batch_size, horizon, padded_dim = velocity.shape
    if (horizon < 1 or padded_dim < predictor.cfg.action_dim or timestep.shape != (batch_size,)
            or z_t.ndim != 3 or z_t.shape[0] != batch_size or target_delta.shape != z_t.shape):
        raise ValueError("检查[B,H,A_pad]、timestep[B]及z_t/target_delta[B,L,D]")
    if not torch.isfinite(timestep).all() or ((timestep < 0) | (timestep > 1)).any():
        raise ValueError("生成进度timestep应为[0,1]内的有限数，不是秒数")
    # 参数冻结只是不为参数求导；计算图仍穿过预测器，保留loss对action、v的导数。
    # 策略通常在no_grad下采样；这里临时启用记录，否则loss与v之间不会建图。
    with torch.enable_grad():
        # detach切断原计算图，但共享存储；clone复制数据，建立独立的新速度叶子。
        v = velocity.detach().clone().requires_grad_(True)  # [B,H,A_pad]，例如[1,50,32]。
        # 计算clean动作
        clean = x_tau.detach() - timestep[:, None, None] * v
        # 取第0个动作的前predictor.cfg.action_dim维
        action = clean[:, 0, :predictor.cfg.action_dim]
        # 使用预测器预测delta
        delta = predictor(z_t.detach(), action)
        # 计算损失
        # 单机器人仍保留B=1；mean此时不改数值。这里比较候选动作预测与纠偏目标，
        # 不是重新执行LVM的历史/当前观测比较。
        loss = visual_mismatch(delta, target_delta.detach()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("OGG损失出现NaN/Inf")
        # 计算梯度
        # 返回元组，取其中loss对v的导数；不向模型参数.grad累加。
        # create_graph=False：无需再对这次导数求导，不建立高阶梯度图。
        gradient = torch.autograd.grad(loss, v, create_graph=False)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("OGG输入梯度出现NaN/Inf")
        # 更新速度
        v_guided = (v - eta * gradient).detach()
    return v_guided, loss.item()
