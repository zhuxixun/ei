"""
入门脚本 2：模仿学习 —— Diffusion Policy 训练 PushT

LeRobot 经典入门任务：控制机械臂把 T 形积木推入圆环。
- 数据：从 HuggingFace 下载官方演示数据集 lerobot/pusht（首次约 300MB）
- 策略：Diffusion Policy（具身智能最主流的动作生成模型之一）
- 设备：MPS (Apple GPU)

用法：python 02_lerobot_pusht.py [训练步数]
"""
import sys
from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

# 训练步数：200 步冒烟测试跑通流程；5000 步能学到东西；20000 步效果不错
training_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
output_directory = Path(f"outputs/train/pusht_diffusion_{training_steps}steps")
output_directory.mkdir(parents=True, exist_ok=True)

# 数据集：优先用本地下载好的，避免从 HuggingFace 反复下载
dataset_path = Path("data/pusht")  # 本地数据集（含 meta/data/videos）
if dataset_path.exists():
    dataset_root, dataset_repo = dataset_path.resolve(), "local/pusht"
    print(f"📁 使用本地数据集: {dataset_root}")
else:
    dataset_root, dataset_repo = None, "lerobot/pusht"  # 从 HuggingFace 拉取（需要能访问 huggingface.co）
    print("📡 数据集不存在，将尝试从 HuggingFace 下载")

# 1. 设备选择：Mac 上用 MPS，有 NVIDIA 卡时自动用 CUDA
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  设备: {device}")

# 2. 读取数据集元信息，确定策略输入/输出维度
dataset_metadata = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
print(f"📦 数据集: lerobot/pusht, 观测: {list(input_features)}, 动作: {list(output_features)}")

# 3. 创建 Diffusion Policy（输入输出维度由数据集自动确定）
cfg = DiffusionConfig(
    input_features=input_features,
    output_features=output_features,
)
policy = DiffusionPolicy(cfg)
policy.train()
policy.to(device)
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

# 4. 计算 delta_timestamps（策略需要的历史帧/未来动作帧的时间偏移）
fps = dataset_metadata.fps
delta_timestamps = {
    "observation.image": [i / fps for i in range(-(cfg.n_obs_steps - 1), 1)],
    "observation.state": [i / fps for i in range(-(cfg.n_obs_steps - 1), 1)],
    # 动作窗口 = 扩散模型的预测 horizon
    "action": [i / fps for i in range(-1, cfg.horizon - 1)],
}

dataset = LeRobotDataset(dataset_repo, root=dataset_root, delta_timestamps=delta_timestamps)
print(f"🚀 开始训练 {training_steps} 步...（MPS 上约 200 步需 3-5 分钟）")

optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr)
dataloader = torch.utils.data.DataLoader(
    dataset, num_workers=0, batch_size=16, shuffle=True, drop_last=True
)

# 5. 训练循环
step = 0
for batch in dataloader:
    if step >= training_steps:
        break
    # 数据预处理（归一化等），tensor 已在正确设备上
    batch = preprocessor(batch)

    # 前向 + 损失
    loss, _ = policy.forward(batch)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"  step {step:>5d}/{training_steps}  loss: {loss.item():.4f}")
    step += 1

# 6. 保存 checkpoint
policy.save_pretrained(output_directory)
preprocessor.save_pretrained(output_directory)
postprocessor.save_pretrained(output_directory)
print(f"💾 Checkpoint 已保存: {output_directory}")

# 7. 在仿真里评估：让学到的策略真的去推 T 块！
print("🎬 开始在 PushT 仿真中评估...")
import gymnasium as gym
import gym_pusht  # noqa: F401 注册环境

# obs_type="pixels_agent_pos"：与训练数据完全一致（agent_pos 2 维 + 图像）
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
    observation_width=96,
    observation_height=96,
    max_episode_steps=300,
)
policy.eval()

obs, _ = env.reset()  # obs = {"pixels": (96,96,3) uint8, "agent_pos": (2,) float32}
total_reward, step = 0.0, 0
while step < 300:  # PushT 一个 episode 最长 300 步
    img = obs["pixels"]  # (96, 96, 3) uint8
    # 转成与训练数据一致的 CHW float32 [0,1] 格式
    img_tensor = torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    batch = {
        "observation.image": img_tensor,
        "observation.state": torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        action = policy.select_action(preprocessor(batch))
    action = postprocessor(action)  # 反归一化
    action_np = action.squeeze(0).cpu().numpy()  # (2,) 推块器位置

    obs, reward, terminated, truncated, _ = env.step(action_np)
    total_reward += reward
    step += 1
    if step % 50 == 0:
        print(f"  评估 step {step}: 累计奖励 {total_reward:.1f}")
    if terminated or truncated:
        break

print(f"📊 评估完成: 累计奖励 {total_reward:.1f}（>30 说明策略学会了；完整训练 20000 步可超 60）")
