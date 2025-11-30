"""
Webhook 通知工具
使用 apprise 库发送通知，支持多种服务

安装: pip install apprise
"""
import os
from pathlib import Path

import apprise


# 默认配置（可通过环境变量或参数覆盖）
DEFAULT_WEBHOOK_URL = os.environ.get(
    "DMTG_WEBHOOK_URL",
    "ntfys://ntfy.jangit.me/notifytg"
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

    # 创建 Apprise 实例
    apobj = apprise.Apprise()
    apobj.add(url)

    # 准备附件
    attach = None
    if image_path and Path(image_path).exists():
        attach = apprise.AppriseAttachment()
        attach.add(image_path)

    # 发送通知
    result = apobj.notify(
        title=title,
        body=body,
        attach=attach,
    )

    if result:
        print(f"[Notify] Sent: {title}")
    else:
        print(f"[Notify] Failed: {title}")

    return result


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
