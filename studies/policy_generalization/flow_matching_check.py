import torch

# 假设一段示教动作，有 3 个动作维度
action = torch.tensor([0.2, -0.5, 0.8])

# 初始随机噪声
noise = torch.tensor([1.2, 0.5, -0.2])

# 一个 Flow Matching 时间点
t = 0.01

def make_training_target(action, noise, t):
    #TODO1:按照 SmolVLA 的公式构造 x_t
    x_t = t * noise + (1 - t) * action

    #TODO2:计算目标速度u_t
    velocity = noise - action  # 这里假设速度是噪声和动作的差值，实际公式可能更复杂

    return x_t, velocity

def integrate_from_noise(noise, velocity, steps=10):
    x = noise.clone()
    dt = -1.0 / steps

    for _ in range(steps):
        #TODO3:执行一次欧拉积分
        x = x + velocity * dt

    return x

x_t, target_velocity = make_training_target(action, noise, t)

reconstructed = integrate_from_noise(
    noise,
    target_velocity,
    steps=10
)

print(f"真实动作：", action)
print(f"中间状态：", x_t)
print("目标速度:", target_velocity)
print("积分结果:", reconstructed)

assert torch.allclose(reconstructed, action, atol=1e-6)