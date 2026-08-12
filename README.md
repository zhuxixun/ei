# 🤖 具身智能入门环境（Apple Silicon 友好版）

> 没有 NVIDIA GPU？没问题。这套环境在 **MacBook (M5 Pro / M 系列芯片)** 上原生跑通，
> 覆盖强化学习（RL）与模仿学习（Imitation Learning）两条主流学习路线，
> 让你在 Mac 上完成从零到"仿真机器人学会做任务"的完整闭环。

## ✨ 特性

- 🍎 **Apple Silicon 原生支持**：PyTorch MPS 加速、MuJoCo 离屏渲染，无需 CUDA
- 🎮 **两个开箱即用的入门脚本**：
  - `01_ppo_mujoco.py` — 经典强化学习（PPO 训练机器人奔跑）
  - `02_lerobot_pusht.py` — 模仿学习（Diffusion Policy 学习推积木）
- 📦 **数据集已内置**：PushT 演示数据集（7.7MB）随仓库分发，clone 即用，无需翻墙下载
- 🧩 **主流技术栈**：MuJoCo + Gymnasium + stable-baselines3 + HuggingFace LeRobot

## 📋 硬件与系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | macOS（Apple Silicon）或 Linux |
| 芯片 | M1/M2/M3/M4/M5 系列（训练走 MPS）；Linux 用 CPU 或 NVIDIA GPU 均可 |
| 内存 | 8GB 以上（推荐 16GB+） |
| Python | 3.12（3.13/3.14 暂不兼容 LeRobot 生态） |
| 磁盘 | ~5GB（虚拟环境 + 依赖） |

## 🚀 快速开始（3 步）

### 第 1 步：克隆仓库

```bash
git clone https://github.com/zhuxixun/ei.git
cd ei
```

### 第 2 步：一键安装环境

```bash
# 普通环境（自动安装 uv → 创建 venv → 装 MuJoCo/SB3/LeRobot）
bash setup.sh

# 中国大陆网络环境（使用清华 PyPI 镜像，更快更稳）
bash setup.sh --china
```

> 不想用脚本？手动安装见文末 [手动安装](#-手动安装方式)。

### 第 3 步：验证 + 开跑

```bash
source .venv/bin/activate

# 环境体检（MuJoCo / Gymnasium / MPS 全检查）
python verify_env.py

# 🎮 脚本 1：强化学习 —— 训练四足机器人学会奔跑（约 10 秒）
python 01_ppo_mujoco.py

# 🎮 脚本 2：模仿学习 —— 让机械臂学会把 T 形积木推入圆环
python 02_lerobot_pusht.py 200     # 冒烟测试，验证全流程（约 1 分钟）
python 02_lerobot_pusht.py 5000    # 策略开始学到东西（约 12 分钟）
python 02_lerobot_pusht.py 20000   # 完整训练，奖励可达 60+（约 45 分钟）
```

## 🎯 两个脚本学什么

### 脚本 1：PPO 强化学习（`01_ppo_mujoco.py`）

用 stable-baselines3 的 PPO 算法，在 MuJoCo 的 HalfCheetah 环境里通过"试错 + 奖励"学出奔跑策略。

```
┌─────────┐  观测(obs)  ┌──────────────────┐  动作(action)  ┌─────────────┐
│ MuJoCo  │ ──────────► │ PPO 神经网络策略  │ ─────────────► │  环境推进    │
│ 仿真环境 │ ◄────────── │  (策略梯度更新)   │ ◄───────────── │  + 奖励反馈   │
└─────────┘  奖励(reward)└──────────────────┘  更新策略参数   └─────────────┘
```

**预期结果**（M5 Pro 实测）：

| 指标 | 数值 |
|---|---|
| 训练耗时（5 万步） | ~5 秒 |
| 平均回报（随机策略约 -280） | **~740** |
| 产出 | 模型 + 演示视频 `outputs/videos/` |

### 脚本 2：Diffusion Policy 模仿学习（`02_lerobot_pusht.py`）

不靠奖励信号，而是**学习人类的演示数据**：从官方 80 个演示片段（图像 + 机械臂状态）中学会"推 T 块"这一动作模式。Diffusion Policy 是当前具身智能最主流的动作生成模型之一。

**预期结果**（M5 Pro 实测，训练步数与评估奖励的关系）：

| 训练步数 | 耗时 | 评估奖励 | 说明 |
|---|---|---|---|
| 200 | ~1 分钟 | ~2 | 冒烟测试，仅验证流程 |
| 3000 | ~7 分钟 | ~13 | 策略明显在学习 |
| 20000 | ~45 分钟 | 60+ | 任务基本掌握（>30 即算学会） |

> 奖励 0-100：T 块进入圆环的覆盖率。跑完训练后脚本会自动在仿真里评估并打印累计奖励。

## 📦 数据集说明

PushT 演示数据集已随仓库分发（`data/pusht/`，7.7MB），**无需手动下载**。

如果删除了想重新获取，二选一：

```bash
# 方式 A：脚本下载（需要能访问 huggingface.co，或设置代理）
python download_pusht.py

# 方式 B：中国大陆用户，从 hf-mirror 手动下载（注意必须带 /datasets 前缀）
BASE="https://hf-mirror.com/datasets/lerobot/pusht/resolve/main"
mkdir -p data/pusht/{meta/episodes/chunk-000,data/chunk-000,videos/observation.image/chunk-000}
for f in meta/info.json meta/stats.json meta/tasks.parquet \
         meta/episodes/chunk-000/file-000.parquet \
         data/chunk-000/file-000.parquet \
         videos/observation.image/chunk-000/file-000.mp4; do
  curl -sL -o "data/pusht/$f" "$BASE/$f"
done
```

## 📁 目录结构

```
ei/
├── setup.sh                 # 一键环境安装脚本
├── verify_env.py            # 环境体检脚本
├── 01_ppo_mujoco.py         # RL：PPO 训练 HalfCheetah + 录视频
├── 02_lerobot_pusht.py      # 模仿学习：Diffusion Policy 训练 PushT + 评估
├── download_pusht.py        # 数据集下载脚本
├── data/pusht/              # PushT 数据集（内置）
│   ├── meta/                #   元数据（info/stats/tasks/episodes）
│   ├── data/                #   状态轨迹 parquet
│   └── videos/              #   80 个演示片段视频
└── outputs/                 # 训练产出（模型 / 视频，已 gitignore）
```

## ❓ 常见问题（FAQ）

### 1. 脚本 1 训练很快，但脚本 2 很慢，正常吗？
正常。脚本 1 是低维状态观测（17 维向量），CPU 都极快；脚本 2 是图像输入 + 扩散模型（数百次去噪迭代），计算量大得多，M5 Pro 上 20000 步约 45 分钟。想提速可减小 `cfg.down_dims` 或减少 `num_train_timesteps`。

### 2. 为什么 stable-baselines3 用 CPU 而不是 MPS？
SB3 内部使用 float64 张量，而 Apple MPS **不支持 float64**，强行启用会直接崩溃。低维状态观测下 CPU 训练速度已足够（5 万步仅 5 秒），故脚本 1 默认 CPU。LeRobot（脚本 2）使用 float32，可以完整走 MPS 加速。

### 3. 报错 `MUJOCO_GL` 相关？
**不要设置 `MUJOCO_GL` 环境变量**。macOS 上 MuJoCo 默认走 cgl（Core OpenGL 离屏渲染），手动设成 `osmesa`/`egl`（仅 Linux 可用）反而报错。如果看到渲染黑屏/报错，检查是否在 shell 配置里设过它：`unset MUJOCO_GL`。

### 4. 运行脚本 2 时提示 `HF_ENDPOINT` / 下载数据集失败？
- 数据集已随仓库分发，正常流程**不会触发下载**
- 若报错与 `hf-mirror` 有关：你的 shell 配置（如 `~/.zshrc`）可能设了 `export HF_ENDPOINT=https://hf-mirror.com`，新版 huggingface_hub 对镜像响应要求 `X-Repo-Commit` 头，镜像经常缺失导致失败。临时 `unset HF_ENDPOINT` 可绕过
- 中国大陆网络直连 huggingface.co 不稳定时，参考上方"数据集说明"用 hf-mirror 手动下载

### 5. 控制台刷屏 `SDL` / `torchcodec` 警告，要紧吗？
不要紧，都是 macOS 上的无害警告：
- `SDLApplication is implemented in both...`：cv2 与 pygame 的 SDL 动态库重复加载，不影响运行
- `torchcodec ... libavutil ... could not load`：torchcodec 缺 FFmpeg 库，LeRobot 会自动回退到 pyav 解码器

### 6. 想用 NVIDIA GPU / 云服务器跑？
脚本里设备选择是自动的：有 CUDA 时 LeRobot 自动用 CUDA。部署到云 GPU（如 AutoDL）时：`bash setup.sh` 安装（Linux 需确保 Python 3.12），然后把 `02_lerobot_pusht.py` 的 `device` 逻辑改为 `torch.device("cuda")` 即可，训练速度可提升 10-30 倍。

### 7. 修改脚本后如何更新到 GitHub？
```bash
git add -A && git commit -m "描述改动" && git push
```

## 🛠 手动安装方式

不想用 `setup.sh` 的话，手动执行等价命令：

```bash
# 1. 创建 Python 3.12 虚拟环境（需要先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh）
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. 基础依赖（中国大陆可加 --index-url https://pypi.tuna.tsinghua.edu.cn/simple）
uv pip install mujoco "gymnasium[mujoco]" stable-baselines3 tqdm rich moviepy imageio imageio-ffmpeg

# 3. LeRobot（源码安装，main 分支才有最新 API）
mkdir -p src
curl -sL -o /tmp/lerobot.tar.gz https://github.com/huggingface/lerobot/archive/refs/heads/main.tar.gz
tar xzf /tmp/lerobot.tar.gz -C src && mv src/lerobot-main src/lerobot && rm /tmp/lerobot.tar.gz
uv pip install "./src/lerobot[pusht,diffusion]"
```

> ⚠️ 注意：LeRobot 必须装 GitHub main 分支（PyPI 上的 0.2.x 是 2024 年的旧版，API 已完全不一样）。

## 🧭 下一步学习路径

1. **完整训练一次**：`python 02_lerobot_pusht.py 20000`，观察奖励随步数增长
2. **换策略对比**：Diffusion Policy 之外，LeRobot 还支持 ACT、VLA 等策略，参考 `src/lerobot/examples/training/`
3. **可视化数据**：`python -m lerobot.scripts.lerobot_dataset_viz --dataset.repo_id local/pusht --dataset.root data/pusht`（需替换为实际命令参数，详见 LeRobot 文档）
4. **仿真自主数据收集**：`lerobot-rollout` 让策略在仿真里闭环滚动，自主收集新数据
5. **上真机**：有预算后可接 SO-101/乐高臂等廉价机械臂（LeRobot 官方支持），把仿真策略迁移到真机

## 📚 参考

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) — 机器人学习框架
- [MuJoCo](https://mujoco.org/) — 物理仿真引擎
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — RL 算法库
- [Gymnasium](https://gymnasium.farama.org/) — RL 环境标准接口
- [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137) — 动作生成的扩散模型方法

## 📄 License

MIT
