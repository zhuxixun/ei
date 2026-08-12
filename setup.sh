#!/usr/bin/env bash
# 一键环境安装脚本（macOS / Linux）
# 用法: bash setup.sh [--china]
#   --china  使用国内镜像源（清华 PyPI），适合中国大陆网络环境
set -euo pipefail

PYPI_INDEX="https://pypi.org/simple"
if [[ "${1:-}" == "--china" ]]; then
  PYPI_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "🌏 使用清华 PyPI 镜像源"
fi

echo "🔍 检查 Python 3.12..."
if command -v uv &>/dev/null; then
  echo "✅ 已安装 uv"
else
  echo "⚠️  未找到 uv，正在安装（Astral 官方脚本）..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "🔧 创建虚拟环境 (.venv, Python 3.12)..."
uv venv .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

echo "📦 安装基础依赖（MuJoCo / Gymnasium / SB3 / 视频工具）..."
uv pip install mujoco "gymnasium[mujoco]" stable-baselines3 tqdm rich moviepy imageio imageio-ffmpeg --index-url "$PYPI_INDEX"

echo "📥 获取 LeRobot 源码（GitHub main 分支）..."
if [[ ! -d src/lerobot ]]; then
  mkdir -p src
  curl -sL -o /tmp/lerobot.tar.gz https://github.com/huggingface/lerobot/archive/refs/heads/main.tar.gz
  tar xzf /tmp/lerobot.tar.gz -C src
  mv src/lerobot-main src/lerobot
  rm -f /tmp/lerobot.tar.gz
else
  echo "  已存在 src/lerobot，跳过下载"
fi

echo "📦 安装 LeRobot（含 PushT 仿真 + Diffusion Policy 依赖）..."
uv pip install "./src/lerobot[pusht,diffusion]" --index-url "$PYPI_INDEX"

echo ""
echo "✅ 安装完成！"
echo "   下一步: source .venv/bin/activate && python verify_env.py"
