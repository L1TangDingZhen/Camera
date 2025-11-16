#!/bin/bash
# Jetson Docker 快速启动脚本
# 用法: ./run_docker_jetson.sh [lite|balanced|performance]

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Docker Hub配置
# TODO: 替换为你的Docker Hub用户名
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-your-dockerhub-username}"
IMAGE_NAME="${IMAGE_NAME:-life-tracker}"
IMAGE_TAG="${IMAGE_TAG:-jetson-latest}"

FULL_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# 默认模式
MODE=${1:-balanced}

# 配置文件映射
case $MODE in
    lite)
        CONFIG_FILE="config_jetson_lite.yaml"
        echo -e "${GREEN}[模式] Lite (7W, 45+ FPS)${NC}"
        ;;
    balanced)
        CONFIG_FILE="config_jetson_balanced.yaml"
        echo -e "${GREEN}[模式] Balanced (15W, 30-35 FPS) - 推荐${NC}"
        ;;
    performance)
        CONFIG_FILE="config_jetson_performance.yaml"
        echo -e "${GREEN}[模式] Performance (25W, 20-25 FPS)${NC}"
        ;;
    *)
        echo -e "${RED}[错误] 无效模式: $MODE${NC}"
        echo "用法: $0 [lite|balanced|performance]"
        exit 1
        ;;
esac

# 创建必要的目录
echo -e "${YELLOW}[准备] 创建数据目录...${NC}"
mkdir -p data logs models training_data

# 检查摄像头
echo -e "${YELLOW}[检查] 摄像头设备...${NC}"
if [ -e /dev/video0 ]; then
    CAMERA_DEVICE="/dev/video0"
    echo -e "${GREEN}[检查] 找到摄像头: /dev/video0${NC}"
else
    echo -e "${RED}[警告] 未找到 /dev/video0，请检查摄像头连接${NC}"
    CAMERA_DEVICE="/dev/video0"
fi

# 拉取最新镜像
echo -e "${YELLOW}[Docker] 拉取镜像: ${FULL_IMAGE}${NC}"
docker pull ${FULL_IMAGE}

# 停止旧容器（如果存在）
if [ "$(docker ps -a -q -f name=life-tracker)" ]; then
    echo -e "${YELLOW}[Docker] 停止并删除旧容器...${NC}"
    docker stop life-tracker 2>/dev/null || true
    docker rm life-tracker 2>/dev/null || true
fi

# 运行容器
echo -e "${GREEN}[Docker] 启动容器...${NC}"
docker run -d \
    --name life-tracker \
    --runtime nvidia \
    --restart unless-stopped \
    --device=${CAMERA_DEVICE}:/dev/video0 \
    --network host \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/training_data:/app/training_data \
    -e DISPLAY=$DISPLAY \
    ${FULL_IMAGE} \
    python3 main.py --config config/${CONFIG_FILE}

# 等待容器启动
echo -e "${YELLOW}[等待] 容器启动中...${NC}"
sleep 3

# 检查状态
if [ "$(docker ps -q -f name=life-tracker)" ]; then
    echo -e "${GREEN}[成功] 容器已启动！${NC}"
    echo ""
    echo "=========================================="
    echo "  Life Tracker 已运行"
    echo "=========================================="
    echo ""
    echo "查看日志:"
    echo "  docker logs -f life-tracker"
    echo ""
    echo "查看状态:"
    echo "  docker ps"
    echo ""
    echo "停止容器:"
    echo "  docker stop life-tracker"
    echo ""
    echo "访问Web Dashboard:"
    echo "  http://$(hostname -I | awk '{print $1}'):5000"
    echo ""
    echo "=========================================="
else
    echo -e "${RED}[错误] 容器启动失败！${NC}"
    echo "查看错误日志:"
    echo "  docker logs life-tracker"
    exit 1
fi
