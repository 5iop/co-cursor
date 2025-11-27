"""
数据集下载工具
支持下载:
1. BOUN Mouse Dynamics Dataset
2. SapiMouse Dataset
3. Localized Narratives (Open Images, COCO, Flickr30k, ADE20k)
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
DATASETS_DIR = Path("datasets")
DOWNLOAD_DIR = Path("downloads")

# 数据集 URL 配置
DATASETS = {
    "boun": {
        "name": "BOUN Mouse Dynamics Dataset",
        "url": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/w6cxr8yc7p-2.zip",
        "output_dir": "boun-mouse-dynamics-dataset",
        "type": "multipart_zip",
    },
    "sapimouse": {
        "name": "SapiMouse Dataset",
        "url": "https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.zip",
        "output_dir": "sapimouse",
        "type": "zip",
    },
    "localized_narratives": {
        "name": "Localized Narratives (Full)",
        "output_dir": "open_images_v6",  # 与 train.py 的 --open_images_dir 对应
        "type": "jsonl_files",
        "files": {
            # Open Images V6
            "open_images_train": [
                f"https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-{i:05d}-of-00010.jsonl"
                for i in range(10)
            ],
            "open_images_val": [
                "https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl"
            ],
            "open_images_test": [
                "https://storage.googleapis.com/localized-narratives/annotations/open_images_test_localized_narratives.jsonl"
            ],
            # COCO
            "coco_train": [
                f"https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-{i:05d}-of-00004.jsonl"
                for i in range(4)
            ],
            "coco_val": [
                "https://storage.googleapis.com/localized-narratives/annotations/coco_val_localized_narratives.jsonl"
            ],
            # Flickr30k
            "flickr30k_train": [
                "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_train_localized_narratives.jsonl"
            ],
            "flickr30k_val": [
                "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_val_localized_narratives.jsonl"
            ],
            "flickr30k_test": [
                "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_test_localized_narratives.jsonl"
            ],
            # ADE20k
            "ade20k_train": [
                "https://storage.googleapis.com/localized-narratives/annotations/ade20k_train_localized_narratives.jsonl"
            ],
            "ade20k_val": [
                "https://storage.googleapis.com/localized-narratives/annotations/ade20k_validation_localized_narratives.jsonl"
            ],
        },
    },
}


def download_file(url: str, output_path: Path, desc: str = None) -> bool:
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        output_path: 保存路径
        desc: 进度条描述

    Returns:
        是否下载成功
    """
    if desc is None:
        desc = output_path.name

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"下载失败 {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """解压 ZIP 文件"""
    print(f"正在解压: {zip_path.name}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="解压进度") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

    print(f"解压完成: {extract_to}")


def check_and_install_7z():
    """检查并提示安装 7z 工具"""
    import platform

    for cmd in ['7z', '7za', 'p7zip']:
        try:
            subprocess.run([cmd], capture_output=True, check=False)
            return cmd
        except FileNotFoundError:
            continue

    print("\n" + "=" * 70)
    print("未检测到 7-Zip 工具，需要安装才能解压 BOUN 数据集")
    print("=" * 70)

    system = platform.system()
    if system == 'Linux':
        print("请运行: sudo apt install -y p7zip-full")
    elif system == 'Darwin':
        print("请运行: brew install p7zip")
    elif system == 'Windows':
        print("请下载安装: https://www.7-zip.org/download.html")

    return None


def extract_multipart_zip(base_path: Path, extract_to: Path):
    """使用 7z 解压分卷 ZIP 文件"""
    zip_cmd = check_and_install_7z()
    if not zip_cmd:
        raise Exception("未安装 7-Zip 工具")

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"使用 {zip_cmd} 解压...")

    cmd = [zip_cmd, 'x', str(base_path), f'-o{extract_to}', '-y']
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    for line in process.stdout:
        if line.strip():
            print(f"  {line.strip()}")

    process.wait()

    if process.returncode != 0:
        stderr_output = process.stderr.read()
        raise Exception(f"7-Zip 解压失败: {stderr_output}")


def download_sapimouse():
    """下载 SapiMouse 数据集"""
    print("\n" + "=" * 60)
    print("下载 SapiMouse Dataset")
    print("=" * 60)

    config = DATASETS["sapimouse"]
    output_dir = DATASETS_DIR / config["output_dir"]

    if output_dir.exists():
        response = input(f"数据集已存在: {output_dir}\n是否重新下载? (y/n): ")
        if response.lower() != 'y':
            print("跳过下载")
            return
        shutil.rmtree(output_dir)

    # 下载
    zip_path = DOWNLOAD_DIR / "sapimouse.zip"
    if not zip_path.exists():
        print(f"下载 {config['url']}")
        if not download_file(config["url"], zip_path, "sapimouse.zip"):
            return

    # 解压
    extract_zip(zip_path, DATASETS_DIR)

    # 验证
    if output_dir.exists():
        user_dirs = list(output_dir.glob("user*"))
        print(f"\n下载完成! 用户数: {len(user_dirs)}")
    else:
        print("下载或解压失败")


def download_boun():
    """下载 BOUN 数据集"""
    print("\n" + "=" * 60)
    print("下载 BOUN Mouse Dynamics Dataset")
    print("=" * 60)

    config = DATASETS["boun"]
    output_dir = DATASETS_DIR / config["output_dir"]

    if output_dir.exists():
        response = input(f"数据集已存在: {output_dir}\n是否重新下载? (y/n): ")
        if response.lower() != 'y':
            print("跳过下载")
            return
        shutil.rmtree(output_dir)

    # 下载主 ZIP
    main_zip = DOWNLOAD_DIR / "w6cxr8yc7p-2.zip"
    if not main_zip.exists():
        print(f"下载 {config['url']}")
        if not download_file(config["url"], main_zip, "boun-dataset.zip"):
            return

    # 解压主 ZIP
    temp_extract = DOWNLOAD_DIR / "temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    extract_zip(main_zip, temp_extract)

    # 查找数据集分卷文件
    main_dataset_zip = temp_extract / f"{config['output_dir']}.zip"
    if main_dataset_zip.exists():
        part_files = list(temp_extract.glob(f"{config['output_dir']}.z*"))
        if part_files:
            extract_multipart_zip(main_dataset_zip, DATASETS_DIR)
        else:
            extract_zip(main_dataset_zip, DATASETS_DIR)

    # 清理
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    if output_dir.exists():
        users_dir = output_dir / "users"
        if users_dir.exists():
            print(f"\n下载完成! 用户数: {len(list(users_dir.iterdir()))}")
    else:
        print("下载或解压失败")


def download_localized_narratives(subset: str = "all", num_threads: int = 4):
    """
    下载 Localized Narratives 数据集

    Args:
        subset: 下载的子集 ("all", "open_images", "coco", "flickr30k", "ade20k")
        num_threads: 并行下载线程数
    """
    print("\n" + "=" * 60)
    print("下载 Localized Narratives Dataset")
    print("=" * 60)

    config = DATASETS["localized_narratives"]
    output_dir = DATASETS_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集要下载的 URL
    urls_to_download = []
    files_config = config["files"]

    for key, urls in files_config.items():
        # 根据 subset 过滤
        if subset != "all":
            if not key.startswith(subset):
                continue

        for url in urls:
            filename = url.split("/")[-1]
            output_path = output_dir / filename

            if output_path.exists():
                print(f"已存在，跳过: {filename}")
                continue

            urls_to_download.append((url, output_path))

    if not urls_to_download:
        print("所有文件已下载完成")
        return

    print(f"\n需要下载 {len(urls_to_download)} 个文件")
    print(f"使用 {num_threads} 个线程并行下载\n")

    # 并行下载
    def download_worker(args):
        url, output_path = args
        return download_file(url, output_path, output_path.name)

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_worker, args): args for args in urls_to_download}

        for future in as_completed(futures):
            url, output_path = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"下载失败 {output_path.name}: {e}")
                fail_count += 1

    print(f"\n下载完成: 成功 {success_count}, 失败 {fail_count}")

    # 统计文件
    jsonl_files = list(output_dir.glob("*.jsonl"))
    total_size = sum(f.stat().st_size for f in jsonl_files)
    print(f"JSONL 文件数: {len(jsonl_files)}")
    print(f"总大小: {total_size / (1024**3):.2f} GB")


def show_menu():
    """显示交互式菜单"""
    print("\n" + "=" * 60)
    print("数据集下载工具")
    print("=" * 60)
    print("\n可用数据集:")
    print("  1. SapiMouse Dataset (~100MB)")
    print("  2. BOUN Mouse Dynamics Dataset (~2GB)")
    print("  3. Localized Narratives - 全部 (~15GB)")
    print("  4. Localized Narratives - 仅 Open Images (~10GB)")
    print("  5. Localized Narratives - 仅 COCO (~3GB)")
    print("  6. Localized Narratives - 仅 Flickr30k (~500MB)")
    print("  7. Localized Narratives - 仅 ADE20k (~200MB)")
    print("  8. 下载全部数据集")
    print("  0. 退出")
    print()

    choice = input("请选择 (0-8): ").strip()
    return choice


def main():
    parser = argparse.ArgumentParser(description="数据集下载工具")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sapimouse", "boun", "localized_narratives", "open_images", "coco", "flickr30k", "ade20k", "all"],
        help="要下载的数据集"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="并行下载线程数 (默认: 4)"
    )

    args = parser.parse_args()

    # 创建目录
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        # 命令行模式
        if args.dataset == "sapimouse":
            download_sapimouse()
        elif args.dataset == "boun":
            download_boun()
        elif args.dataset == "localized_narratives":
            download_localized_narratives("all", args.threads)
        elif args.dataset == "open_images":
            download_localized_narratives("open_images", args.threads)
        elif args.dataset == "coco":
            download_localized_narratives("coco", args.threads)
        elif args.dataset == "flickr30k":
            download_localized_narratives("flickr30k", args.threads)
        elif args.dataset == "ade20k":
            download_localized_narratives("ade20k", args.threads)
        elif args.dataset == "all":
            download_sapimouse()
            download_boun()
            download_localized_narratives("all", args.threads)
    else:
        # 交互式菜单模式
        while True:
            choice = show_menu()

            if choice == "0":
                print("退出")
                break
            elif choice == "1":
                download_sapimouse()
            elif choice == "2":
                download_boun()
            elif choice == "3":
                download_localized_narratives("all")
            elif choice == "4":
                download_localized_narratives("open_images")
            elif choice == "5":
                download_localized_narratives("coco")
            elif choice == "6":
                download_localized_narratives("flickr30k")
            elif choice == "7":
                download_localized_narratives("ade20k")
            elif choice == "8":
                download_sapimouse()
                download_boun()
                download_localized_narratives("all")
            else:
                print("无效选择，请重试")

            input("\n按 Enter 继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
