#!/usr/bin/env bash
# 一键环境安装脚本（macOS / Linux），基于 uv 项目管理（pyproject.toml + uv.lock）
# 用法: bash setup.sh [--china]
#   --china  使用国内镜像源（清华 PyPI），适合中国大陆网络环境
set -euo pipefail

echo "🔍 检查 uv..."
if command -v uv &>/dev/null; then
  echo "✅ 已安装 uv $(uv --version | awk '{print $2}')"
else
  echo "⚠️  未找到 uv，正在安装（Astral 官方脚本）..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

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

echo "🔧 按 pyproject.toml + uv.lock 同步依赖（自动创建 .venv，Python 3.12）..."
if [[ "${1:-}" == "--china" ]]; then
  export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "🌏 使用清华 PyPI 镜像源"
  # 镜像同步会临时把 uv.lock 中的下载地址改写为镜像地址，结束后还原，避免污染仓库
  if [[ -f uv.lock ]]; then
    cp uv.lock /tmp/uv.lock.bak.$$
  fi
  uv sync
  if [[ -f /tmp/uv.lock.bak.$$ ]]; then
    cp /tmp/uv.lock.bak.$$ uv.lock && rm -f /tmp/uv.lock.bak.$$
    echo "♻️  已还原 uv.lock（保持官方 PyPI 记录）"
  fi
else
  uv sync
fi

echo ""
echo "✅ 安装完成！"
echo "   下一步: source .venv/bin/activate && python verify_env.py"
