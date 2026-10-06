#!/usr/bin/env bash
# AnimalTracker - macOS 一键启动脚本
#
# 用法:
#   1. 打开「终端」(Terminal)
#   2. cd 到本项目目录,例如:
#        cd "/Users/chengneng/Project/AnimalTracker/AnimalTracker legacy"
#   3. 运行:  bash run_mac.sh
#
# 说明:
#   - 项目里已有的 venv/ 是 Windows 环境创建的,在 macOS 上无法使用,
#     本脚本会另建一个 .venv_mac 虚拟环境,不影响原有 venv/。
#   - 首次运行会下载 torch / ultralytics 等依赖(约 1~2GB),请耐心等待。
#   - 需要 Python 3.9 及以上版本(macOS 自带或从 python.org 安装均可)。

set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "==> 使用 Python: $($PY --version 2>&1 || echo '未找到 python3,请先安装 Python 3')"

# 1. 创建虚拟环境
if [ ! -d ".venv_mac" ]; then
    echo "==> [1/3] 创建虚拟环境 .venv_mac ..."
    "$PY" -m venv .venv_mac
fi

# 2. 安装依赖
echo "==> [2/3] 安装依赖(首次较慢,需下载 torch/ultralytics)..."
.venv_mac/bin/python -m pip install --upgrade pip --quiet
.venv_mac/bin/python -m pip install -r requirements.txt

# 3. 启动 Streamlit
echo "==> [3/3] 启动 Streamlit ..."
echo "    浏览器打开: http://localhost:8501"
exec .venv_mac/bin/streamlit run app.py
