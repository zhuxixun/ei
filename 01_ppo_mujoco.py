"""
入门脚本 1：经典强化学习 —— PPO 训练 MuJoCo 机器人

用 stable-baselines3 + MPS(Apple GPU) 训练 HalfCheetah 跑起来。
约 2-5 分钟出结果，是理解"RL 训练循环"的最小闭环。
"""
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "HalfCheetah-v5"   # 或换成 Humanoid-v5（更慢更有挑战）
TOTAL_TIMESTEPS = 50_000    # 训练总步数（可调大）
EVAL_FREQ = 5_000

def make_env():
    return gym.make(ENV_ID, render_mode="rgb_array")

# 1. 创建向量化环境
env = DummyVecEnv([make_env])

# 2. 训练（注意：SB3 内部用 float64，而 MPS 不支持 float64，
#    所以 SB3 统一用 CPU 训练；低维状态观测下 CPU 已足够快）
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=256,
    learning_rate=3e-4,
    verbose=0,
    device="cpu",
)
print(f"✅ 开始训练 {ENV_ID}，设备: {model.device}")

# 3. 训练 + 定期评估
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
model.save(f"outputs/ppo_{ENV_ID}")

# 4. 回放：让训练好的策略跑一遍并录视频
print("🎬 录制演示视频...")
eval_env = DummyVecEnv([make_env])
eval_env = VecVideoRecorder(
    eval_env, "outputs/videos",
    record_video_trigger=lambda x: x == 0,
    video_length=200,
    name_prefix=f"ppo_{ENV_ID}",
)
obs = eval_env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = eval_env.step(action)
eval_env.close()

# 5. 最终评估
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, DummyVecEnv([make_env]), n_eval_episodes=10)
print(f"📊 平均回报: {mean_reward:.1f} ± {std_reward:.1f}")
print(f"💾 模型已保存: outputs/ppo_{ENV_ID}.zip")
print(f"🎥 视频已保存: outputs/videos/ppo_{ENV_ID}-step-0-to-step-200.mp4")
