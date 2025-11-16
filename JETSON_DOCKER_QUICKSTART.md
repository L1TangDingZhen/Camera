# Jetson Docker 快速部署指南

> 从Docker Hub下载预构建镜像并运行

---

## 📋 前置条件

### 硬件
- NVIDIA Jetson Orin Nano Super (或其他Jetson设备)
- USB摄像头或CSI摄像头
- 网络连接

### 软件
- JetPack 5.1.2+ (预装在Jetson上)
- 已安装Docker和NVIDIA Container Toolkit

---

## 🚀 快速开始（3步）

### 步骤 1: 安装Docker环境

```bash
# SSH登录到Jetson
ssh your-jetson-username@jetson-ip

# 安装Docker
sudo apt-get update
sudo apt-get install -y docker.io

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 添加当前用户到docker组（避免每次sudo）
sudo usermod -aG docker $USER
# 注销重新登录使生效
```

**验证安装**：
```bash
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi
# 应该能看到GPU信息
```

---

### 步骤 2: 下载启动脚本

```bash
# 克隆仓库（只需要配置文件和脚本）
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# 赋予执行权限
chmod +x run_docker_jetson.sh
```

**或者手动创建脚本**：
如果无法git clone，可以手动创建 `run_docker_jetson.sh`（内容见仓库）

---

### 步骤 3: 配置Docker Hub用户名

编辑 `run_docker_jetson.sh`，修改第13行：

```bash
# 修改前
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-your-dockerhub-username}"

# 修改后（替换为你的Docker Hub用户名）
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-your-actual-username}"
```

**或者**设置环境变量：
```bash
export DOCKERHUB_USERNAME=your-actual-username
```

---

### 步骤 4: 运行！

```bash
# 推荐：Balanced模式（15W, 30-35 FPS）
./run_docker_jetson.sh balanced

# 或者其他模式
./run_docker_jetson.sh lite          # 省电模式（7W, 45+ FPS）
./run_docker_jetson.sh performance   # 高性能（25W, 20-25 FPS）
```

**首次运行会**：
1. 从Docker Hub下载镜像（~2GB，需要5-15分钟）
2. 创建必要的目录
3. 启动容器
4. 显示访问URL

**输出示例**：
```
[Docker] 拉取镜像: yourname/life-tracker:jetson-latest
[Docker] 启动容器...
[成功] 容器已启动！

==========================================
  Life Tracker 已运行
==========================================

查看日志:
  docker logs -f life-tracker

访问Web Dashboard:
  http://192.168.1.100:5000

==========================================
```

---

## 📊 使用命令

### 查看运行状态

```bash
# 查看容器状态
docker ps

# 查看实时日志
docker logs -f life-tracker

# 查看资源使用
jtop  # 或 sudo jtop
```

### 停止/重启

```bash
# 停止
docker stop life-tracker

# 重启
docker restart life-tracker

# 删除容器（保留数据）
docker rm life-tracker
```

### 更新镜像

```bash
# 拉取最新镜像
docker pull <your-dockerhub-username>/life-tracker:jetson-latest

# 重新运行脚本
./run_docker_jetson.sh balanced
```

---

## 📁 数据持久化

以下目录会自动挂载到宿主机，数据持久保存：

```
Camera/
├── data/              ← SQLite数据库
├── logs/              ← 日志文件
├── models/            ← SVM模型文件
└── training_data/     ← 训练数据
```

即使删除容器，这些数据也不会丢失。

---

## 🔧 配置摄像头

### 检查摄像头设备

```bash
# 列出所有摄像头
ls -l /dev/video*

# 测试摄像头
v4l2-ctl --list-devices
```

### 修改摄像头设备

如果你的摄像头不是 `/dev/video0`，修改 `run_docker_jetson.sh` 第38行：

```bash
# 修改前
CAMERA_DEVICE="/dev/video0"

# 修改后（比如你的摄像头是video1）
CAMERA_DEVICE="/dev/video1"
```

---

## 🌐 访问Web Dashboard

### 1. 找到Jetson的IP地址

```bash
hostname -I
# 输出: 192.168.1.100 ...
```

### 2. 在浏览器中访问

```
http://192.168.1.100:5000
```

### 3. 如果无法访问

检查防火墙：
```bash
# 临时关闭防火墙测试
sudo ufw disable

# 或者开放端口
sudo ufw allow 5000/tcp
```

---

## 🐛 故障排查

### 问题1: 镜像拉取失败

```bash
# 错误: failed to pull image
# 解决: 检查网络连接，或手动登录Docker Hub

docker login
# 输入你的Docker Hub用户名和密码
```

### 问题2: 容器启动失败

```bash
# 查看详细错误
docker logs life-tracker

# 常见原因:
# - 摄像头设备不存在 → 检查 ls /dev/video*
# - 端口被占用 → 检查 sudo netstat -tulpn | grep 5000
# - 权限不足 → 确保 docker 命令不需要 sudo
```

### 问题3: 看不到摄像头画面

```bash
# 进入容器调试
docker exec -it life-tracker bash

# 在容器内测试摄像头
python3 -c "import cv2; print(cv2.VideoCapture(0).read())"

# 如果返回 (True, array(...)) 说明摄像头正常
```

### 问题4: FPS太低

```bash
# 切换到Lite模式
docker stop life-tracker
./run_docker_jetson.sh lite

# 或者降低分辨率
# 编辑 config/config_jetson_balanced.yaml
# resolution: [1280, 720] → [640, 480]
```

---

## 🔄 迁移训练数据

如果你在PC上训练了SVM模型，想在Jetson上使用：

```bash
# 在PC上
scp models/pose_classifier_svm.pkl your-jetson-user@jetson-ip:~/Camera/models/
scp -r training_data/ your-jetson-user@jetson-ip:~/Camera/

# Jetson会自动加载这些模型
```

---

## 📈 性能监控

### 实时监控

```bash
# 安装jtop（如果没有）
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service

# 运行监控
jtop
```

**关键指标**：
- **GPU使用率**: 应该 50-80%（说明GPU在工作）
- **CPU使用率**: 30-60%（MediaPipe在CPU上）
- **内存**: <6GB（8GB Jetson）
- **功耗**: 7W/15W/25W（取决于模式）

---

## 🚦 模式选择建议

| 模式 | 功耗 | FPS | 精度 | 适用场景 |
|------|------|-----|------|---------|
| **Lite** | 7W | 45+ | 良好 | 电池供电、演示 |
| **Balanced** | 15W | 30-35 | 优秀 | **日常使用（推荐）** |
| **Performance** | 25W | 20-25 | 最佳 | 高精度需求 |

**建议**：
- 初次使用：先用 **Balanced** 模式
- 如果觉得FPS够用：切换到 **Lite** 省电
- 如果识别不准：切换到 **Performance**

---

## 📚 下一步

容器运行后，你可以：

1. **收集训练数据**：
   ```bash
   # 进入容器
   docker exec -it life-tracker bash

   # 运行数据收集
   python3 collect_data.py
   ```

2. **训练SVM模型**：
   ```bash
   python3 train_svm.py
   ```

3. **查看统计数据**：
   ```bash
   python3 query_stats.py
   ```

4. **测试鲁棒性**：
   ```bash
   python3 test_robustness.py --label sitting --duration 30
   ```

所有命令都在容器内运行，数据会自动保存到宿主机。

---

## 💾 备份和恢复

### 备份数据

```bash
# 备份所有数据
tar -czf life-tracker-backup-$(date +%Y%m%d).tar.gz \
    data/ logs/ models/ training_data/

# 传输到PC
scp life-tracker-backup-*.tar.gz your-pc-user@pc-ip:~/backups/
```

### 恢复数据

```bash
# 解压备份
tar -xzf life-tracker-backup-20241116.tar.gz

# 重新运行容器
./run_docker_jetson.sh balanced
```

---

## ✅ 完整流程总结

```bash
# 1. 安装环境（一次性）
sudo apt-get install -y docker.io nvidia-container-toolkit

# 2. 克隆仓库
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# 3. 配置Docker Hub用户名
export DOCKERHUB_USERNAME=your-username

# 4. 运行！
./run_docker_jetson.sh balanced

# 5. 查看日志
docker logs -f life-tracker

# 6. 浏览器访问
# http://<jetson-ip>:5000
```

**就这么简单！** 🎉
