"""
Webhook 通知工具
使用 apprise 库发送通知，支持多种服务

安装: pip install apprise requests
"""
import os
import sys
import subprocess
import threading
from pathlib import Path

import apprise
import requests


# 默认配置（可通过环境变量或参数覆盖）
# Apprise API 服务器地址
DEFAULT_WEBHOOK_URL = os.environ.get(
    "DMTG_WEBHOOK_URL",
    "https://ntfy.jangit.me/notify/notifytg"
)


def send_notification(
    title: str,
    body: str,
    webhook_url: str = None,
    image_path: str = None,
) -> bool:
    """
    发送通知到 webhook

    Args:
        title: 消息标题
        body: 消息内容
        webhook_url: Webhook URL（默认使用环境变量或配置）
        image_path: 可选的图片路径

    Returns:
        是否发送成功
    """
    url = webhook_url or DEFAULT_WEBHOOK_URL

    # 使用 requests 直接调用 Apprise API
    try:
        files = {}
        data = {
            "title": title,
            "body": body,
        }

        # 添加附件
        if image_path and Path(image_path).exists():
            files["attach"] = open(image_path, "rb")

        if files:
            response = requests.post(url, data=data, files=files)
            # 关闭文件
            for f in files.values():
                f.close()
        else:
            response = requests.post(url, data=data)

        if response.status_code == 200:
            print(f"[Notify] Sent: {title}")
            return True
        else:
            print(f"[Notify] Failed: {title} (HTTP {response.status_code}: {response.text[:100]})")
            return False

    except Exception as e:
        print(f"[Notify] Error: {title} ({e})")
        return False


def send_training_update(
    epoch: int,
    loss: float,
    best_loss: float,
    is_best: bool = False,
    webhook_url: str = None,
    extra_info: str = None,
) -> bool:
    """
    发送训练进度更新
    """
    if is_best:
        title = f"DMTG Best Model - Epoch {epoch}"
    else:
        title = f"DMTG Training - Epoch {epoch}"

    body = f"Loss: {loss:.4f}\nBest: {best_loss:.4f}"
    if extra_info:
        body += f"\n{extra_info}"

    return send_notification(title, body, webhook_url)


def send_image_result(
    title: str,
    image_path: str,
    description: str = None,
    webhook_url: str = None,
) -> bool:
    """
    发送图片结果
    """
    body = description or f"Image: {Path(image_path).name}"
    return send_notification(title, body, webhook_url, image_path)


def run_script_async(
    name: str,
    cmd: list,
    cwd: str = None,
    webhook_url: str = None,
    label: str = None,
) -> threading.Thread:
    """
    在后台线程运行脚本

    Args:
        name: 脚本名称（用于日志）
        cmd: 命令列表
        cwd: 工作目录
        webhook_url: Webhook URL（会自动添加 --webhook 参数）
        label: 标签（用于日志）

    Returns:
        启动的线程对象
    """
    def _run():
        try:
            # 添加 webhook 参数
            run_cmd = cmd.copy()
            if webhook_url:
                run_cmd.extend(["--webhook", webhook_url])

            label_str = f": {label}" if label else ""
            print(f"[{name}] Starting{label_str}")
            result = subprocess.run(run_cmd, cwd=cwd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"[{name}] Completed{label_str}")
            else:
                print(f"[{name}] Failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"[{name}] Error: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def run_visualization_scripts(
    checkpoint_path: str,
    label: str,
    webhook_url: str = None,
    human_data_dir: str = None,
    cwd: str = None,
) -> list:
    """
    运行所有可视化脚本（t-SNE、轨迹生成等）

    Args:
        checkpoint_path: 模型检查点路径
        label: 输出文件标签
        webhook_url: Webhook URL
        human_data_dir: 人类数据目录
        cwd: 工作目录（脚本所在目录）

    Returns:
        启动的线程列表
    """
    if cwd is None:
        cwd = str(Path(__file__).parent.parent.parent)  # 项目根目录

    threads = []

    # 1. 空间特征 t-SNE (plot_tsne_distribution.py)
    cmd_tsne_spatial = [
        sys.executable, "plot_tsne_distribution.py",
        "-c", checkpoint_path,
        "-l", label,
        "--num_human", "300",
        "--num_model", "300",
        "--device", "cpu",
        "--no_display",
    ]
    if human_data_dir:
        cmd_tsne_spatial.extend(["--human_data", human_data_dir])
    threads.append(run_script_async("t-SNE-Spatial", cmd_tsne_spatial, cwd, webhook_url, label))

    # 2. 时间特征 t-SNE (plot_tsne_temporal.py)
    cmd_tsne_temporal = [
        sys.executable, "plot_tsne_temporal.py",
        "--checkpoint", checkpoint_path,
        "--label", label,
        "--num_samples", "300",
        "--device", "cpu",
        "--no_display",
    ]
    threads.append(run_script_async("t-SNE-Temporal", cmd_tsne_temporal, cwd, webhook_url, label))

    # 3. 轨迹生成测试 (test_generate.py)
    cmd_generate = [
        sys.executable, "test_generate.py",
        "--checkpoint", checkpoint_path,
        "-l", label,
        "--num_samples", "3",
        "--device", "cpu",
        "--no_display",
    ]
    threads.append(run_script_async("Generate", cmd_generate, cwd, webhook_url, label))

    return threads


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test notification")
    parser.add_argument("--url", type=str, default=DEFAULT_WEBHOOK_URL)
    parser.add_argument("--title", type=str, default="Test Notification")
    parser.add_argument("--body", type=str, default="This is a test message")
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    success = send_notification(
        args.title,
        args.body,
        webhook_url=args.url,
        image_path=args.image,
    )
    print(f"Success: {success}")
