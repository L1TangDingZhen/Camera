#!/usr/bin/env python3
"""
RTMPose模型下载脚本

功能：
- 自动下载RTMPose模型文件（配置+权重）
- 支持多个模型大小（tiny/s/m/l）
- 自动创建目录结构
- 验证下载完整性

使用方法：
    python download_rtmpose_models.py --model rtmpose-s
    python download_rtmpose_models.py --model rtmpose-tiny
    python download_rtmpose_models.py --all  # 下载所有模型
"""

import argparse
import urllib.request
from pathlib import Path
import hashlib
import sys


# RTMPose模型配置
MODELS = {
    'rtmpose-tiny': {
        'config': {
            'filename': 'rtmpose-t_8xb256-420e_coco-256x192.py',
            'url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py',
        },
        'checkpoint': {
            'filename': 'rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth',
            'md5': 'cfc8f33d',  # 部分MD5（用于验证）
        },
        'size': '~5MB',
        'speed': '~8ms (Jetson)',
        'accuracy': 'AP 65.9%',
    },
    'rtmpose-s': {
        'config': {
            'filename': 'rtmpose-s_8xb256-420e_coco-256x192.py',
            'url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py',
        },
        'checkpoint': {
            'filename': 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
            'md5': 'fcb2599b',
        },
        'size': '~18MB',
        'speed': '~12ms (Jetson FP16)',
        'accuracy': 'AP 68.6%',
    },
    'rtmpose-m': {
        'config': {
            'filename': 'rtmpose-m_8xb256-420e_coco-256x192.py',
            'url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py',
        },
        'checkpoint': {
            'filename': 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
            'md5': '63eb25f7',
        },
        'size': '~55MB',
        'speed': '~20ms (Jetson)',
        'accuracy': 'AP 72.7%',
    },
    'rtmpose-l': {
        'config': {
            'filename': 'rtmpose-l_8xb256-420e_coco-256x192.py',
            'url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py',
        },
        'checkpoint': {
            'filename': 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth',
            'md5': 'f016ffe0',
        },
        'size': '~110MB',
        'speed': '~35ms (Jetson)',
        'accuracy': 'AP 75.3%',
    },
}


def download_file(url: str, dest_path: Path, desc: str = ""):
    """下载文件并显示进度"""
    print(f"\n[下载] {desc}")
    print(f"  URL: {url}")
    print(f"  保存到: {dest_path}")

    try:
        def reporthook(count, block_size, total_size):
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r  进度: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print("\n  下载完成 ✓")
        return True

    except Exception as e:
        print(f"\n  下载失败: {e}")
        return False


def verify_file(file_path: Path, expected_md5_partial: str) -> bool:
    """验证文件完整性（部分MD5匹配）"""
    if not file_path.exists():
        return False

    # 读取文件并计算MD5
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    file_md5 = md5_hash.hexdigest()[:8]  # 取前8位

    if file_md5 == expected_md5_partial:
        print(f"  校验成功 ✓ (MD5: {file_md5})")
        return True
    else:
        print(f"  校验失败 ✗ (期望: {expected_md5_partial}, 实际: {file_md5})")
        return False


def download_model(model_name: str, base_dir: Path = Path('models/rtmpose')):
    """下载指定模型"""
    if model_name not in MODELS:
        print(f"❌ 不支持的模型: {model_name}")
        print(f"   支持的模型: {list(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]

    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"  大小: {model_info['size']}")
    print(f"  速度: {model_info['speed']}")
    print(f"  精度: {model_info['accuracy']}")
    print(f"{'='*60}")

    # 创建目录
    config_dir = base_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 下载配置文件
    config_path = config_dir / model_info['config']['filename']
    if config_path.exists():
        print(f"\n[跳过] 配置文件已存在: {config_path}")
    else:
        if not download_file(
            model_info['config']['url'],
            config_path,
            f"配置文件 - {model_info['config']['filename']}"
        ):
            return False

    # 下载权重文件
    checkpoint_path = base_dir / model_info['checkpoint']['filename']
    if checkpoint_path.exists():
        print(f"\n[跳过] 权重文件已存在: {checkpoint_path}")
        print(f"  正在验证...")
        if verify_file(checkpoint_path, model_info['checkpoint']['md5']):
            print(f"  文件完整 ✓")
        else:
            print(f"  文件损坏，重新下载...")
            checkpoint_path.unlink()

    if not checkpoint_path.exists():
        if not download_file(
            model_info['checkpoint']['url'],
            checkpoint_path,
            f"权重文件 - {model_info['checkpoint']['filename']} ({model_info['size']})"
        ):
            return False

        # 验证下载
        if not verify_file(checkpoint_path, model_info['checkpoint']['md5']):
            print(f"  ⚠️ 校验失败，文件可能损坏")
            return False

    print(f"\n✅ {model_name} 下载完成！")
    print(f"\n配置路径: {config_path}")
    print(f"权重路径: {checkpoint_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='下载RTMPose模型文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载推荐模型（rtmpose-s）
  python download_rtmpose_models.py --model rtmpose-s

  # 下载轻量模型（用于低功耗）
  python download_rtmpose_models.py --model rtmpose-tiny

  # 下载所有模型
  python download_rtmpose_models.py --all

推荐配置:
  - 标准部署: rtmpose-s (12ms, AP 68.6%) ⭐⭐⭐⭐⭐
  - 低功耗:   rtmpose-tiny (8ms, AP 65.9%)
  - 高精度:   rtmpose-m (20ms, AP 72.7%)
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODELS.keys()),
        help='要下载的模型'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='下载所有模型'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/rtmpose'),
        help='输出目录（默认: models/rtmpose）'
    )

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        return

    # 检查网络连接
    print("正在检查网络连接...")
    try:
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("网络连接正常 ✓")
    except:
        print("⚠️ 网络连接失败，请检查网络")
        return

    # 下载模型
    if args.all:
        print(f"\n开始下载所有模型...")
        success_count = 0
        for model_name in MODELS.keys():
            if download_model(model_name, args.output):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"总结: 成功下载 {success_count}/{len(MODELS)} 个模型")
        print(f"{'='*60}")

    else:
        download_model(args.model, args.output)

    # 使用提示
    print(f"\n{'='*60}")
    print("下一步:")
    print("  1. 修改配置文件 (config/config_gpu.yaml):")
    print("       models:")
    print("         pose:")
    print("           backend: rtmpose  # 改这里")
    print(f"           model: {args.model or 'rtmpose-s'}")
    print("")
    print("  2. 运行测试:")
    print("       python main.py --config config/config_gpu.yaml")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
