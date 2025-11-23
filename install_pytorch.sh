#!/bin/bash

echo "=== 安装 PyTorch 和 torchvision for JetPack 6.2.1 ==="

# 更新系统
sudo apt-get update
sudo apt-get install -y python3-pip libjpeg-dev libpng-dev libtiff-dev

# 安装 numpy
pip3 install "numpy<2.0"

# 从 Jetson AI Lab 安装
echo "正在安装 PyTorch 和 torchvision..."
python3 -m pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# 验证安装
echo "=== 验证安装 ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "=== 安装完成！==="