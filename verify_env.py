"""环境验证脚本：检查 MuJoCo / Gymnasium / MPS"""
import sys

print(f"Python: {sys.version.split()[0]}")

# 1. MuJoCo 核心
import mujoco
print(f"MuJoCo: {mujoco.__version__}")

# 2. Gymnasium + 经典机器人环境（CPU 仿真）
import gymnasium as gym
env = gym.make("Humanoid-v5", render_mode=None)
obs, _ = env.reset()
for _ in range(10):
    obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
print(f"Gymnasium: Humanoid-v5 OK, obs_dim={obs.shape}, action_dim={env.action_space.shape[0]}")

# 3. 渲染能力（MuJoCo 原生 viewer 需要窗口；这里测试离屏渲染）
from mujoco import MjData, MjModel
model = MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 3"/>
    <geom type="sphere" size="0.5" pos="0 0 0.5" rgba="1 0 0 1"/>
    <geom type="plane" size="5 5 0.1" rgba="0 0 1 1"/>
  </worldbody>
</mujoco>
""")
data = MjData(model)
mujoco.mj_step(model, data)
renderer = mujoco.Renderer(model, height=240, width=320)
renderer.render()
print("MuJoCo: 离屏渲染 OK (RGB 图像生成成功)")

# 4. PyTorch MPS (Apple GPU)
import torch
print(f"PyTorch: {torch.__version__}, MPS 可用: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device="mps")
    y = (x @ x).sum().item()
    print(f"MPS 矩阵运算 OK: {y:.2f}")

print("\n✅ 全部通过")
