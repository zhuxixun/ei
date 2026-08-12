# 具身智能入门环境（MacBook M5 Pro）

MuJoCo + LeRobot + stable-baselines3 学习环境，全部在 Apple Silicon 上原生运行。

## 环境信息

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.13 | uv 虚拟环境 `.venv` |
| PyTorch | 2.11.0 | MPS (Apple GPU) 加速 |
| MuJoCo | 3.11.0 | 物理仿真 |
| Gymnasium | 1.3.0 | RL 环境标准接口 |
| stable-baselines3 | 2.9.0 | 经典 RL 算法库 |
| LeRobot | 0.6.2 | HuggingFace 机器人学习框架（源码装自 GitHub main 分支） |

## 常用命令

```bash
source .venv/bin/activate

# 入门脚本 1：PPO 强化学习训练 HalfCheetah（约 10 秒，CPU 即可）
python 01_ppo_mujoco.py

# 入门脚本 2：Diffusion Policy 模仿学习训练 PushT（MPS 加速）
python 02_lerobot_pusht.py 200     # 冒烟测试（~1 分钟）
python 02_lerobot_pusht.py 5000    # 能学到东西（~12 分钟）
python 02_lerobot_pusht.py 20000   # 效果不错，奖励可超 60（~45 分钟）

# 环境验证
python verify_env.py
```

## 目录结构

```
.
├── .venv/                  # 虚拟环境
├── 01_ppo_mujoco.py        # RL：PPO 训练 HalfCheetah + 录视频
├── 02_lerobot_pusht.py     # 模仿学习：Diffusion Policy 训练 PushT + 仿真评估
├── verify_env.py           # 环境体检脚本
├── download_pusht.py       # 数据集下载脚本（已用，可留作参考）
├── data/pusht/             # 本地数据集（从 hf-mirror 手动下载，~7.7MB）
│   ├── meta/               #   元数据（info/stats/tasks/episodes）
│   ├── data/               #   状态数据 parquet
│   └── videos/             #   演示视频（80 个 episode 的机械臂推块录像）
├── outputs/
│   ├── videos/             # RL 训练演示视频
│   └── train/              # 模仿学习 checkpoint
└── src/lerobot/            # LeRobot 源码（GitHub main 分支）
```

## macOS 注意事项（踩坑记录）

1. **MPS 不支持 float64**：stable-baselines3 内部用 float64，在 MPS 上会崩 → SB3 用 CPU 训练（低维状态观测下 CPU 足够快，5 万步仅 5 秒）
2. **不要设置 `MUJOCO_GL` 环境变量**：macOS 默认走 cgl（Core OpenGL 离屏渲染），设成 osmesa/egl 反而报错；gymnasium 新版渲染器只认 glfw/egl/osmesa
3. **`HF_ENDPOINT` 冲突**：`~/.zshrc` 里设了 `HF_ENDPOINT=https://hf-mirror.com`，但新版 huggingface_hub 对镜像返回的响应要求 `X-Repo-Commit` 头，huggingface.co 直连又被代理出口 IP 拒绝（401）→ 最终用 curl 从 `https://hf-mirror.com/datasets/lerobot/pusht/resolve/main/...`（注意带 `/datasets` 前缀）手动下载数据集到本地，LeRobot 用 `root=` 参数加载本地数据，完全离线
4. **PyPI 源**：用清华镜像 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 装包更快更稳
5. **torchcodec 警告**（`libavutil` 加载失败）：harmless，已自动回退 pyav 解码器；SDL 重复警告同理无害
6. **LeRobot 0.6.2 API 变化**：`DiffusionConfig` 没有 `delta_timestamps`/`batch_size` 字段；action 窗口长度 = `cfg.horizon`；`save_to_json` 不存在；`lerobot.envs.gymnasium` 模块已移除

## 下一步学习建议

1. 把 `02_lerobot_pusht.py` 的步数调到 20000 完整训练一次，观察奖励变化曲线
2. 改 `--policy.type`（换 ACT / VLA 等策略）对比效果 —— 见 `src/lerobot/examples/training/`
3. 在 `data/pusht` 里用 `lerobot_dataset_viz` 看演示数据
4. 跑 `lerobot_rollout` 让策略在仿真里自主滚动收集数据（闭环）
5. 有 NVIDIA 卡后，把同一套代码丢到 AutoDL 云 GPU 上，`device` 会自动切 CUDA
