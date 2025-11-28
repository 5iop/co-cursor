#!/bin/bash
# 数据集下载脚本
# 使用 aria2c 多线程并行下载

set -e

# 安装依赖
install_deps() {
    echo "=========================================="
    echo "安装依赖..."
    echo "=========================================="
    sudo add-apt-repository -y ppa:savoury1/backports
    sudo apt-get update
    sudo apt-get -y install aria2 vim p7zip-full
    pip install -r requirements.txt
}

# 下载数据集
download_datasets() {
    echo "=========================================="
    echo "开始下载数据集..."
    echo "=========================================="

    # 创建目录
    mkdir -p datasets
    mkdir -p downloads

    # 使用 aria2c 下载
    aria2c -i datasets/download.txt \
        --dir=downloads \
        --continue=true \
        --max-concurrent-downloads=5 \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --console-log-level=notice \
        --summary-interval=10

    echo "=========================================="
    echo "下载完成!"
    echo "=========================================="
}

# 解压数据集
extract_datasets() {
    echo "=========================================="
    echo "解压数据集..."
    echo "=========================================="

    # 解压 SapiMouse
    if [ -f "downloads/sapimouse.zip" ]; then
        echo "解压 SapiMouse..."
        unzip -o downloads/sapimouse.zip -d datasets/
    fi

    # 移动 BOUN Parquet 文件
    if [ -f "downloads/boun_trajectories.parquet" ]; then
        echo "移动 BOUN Parquet 文件..."
        mkdir -p datasets/boun-processed
        mv downloads/boun_trajectories.parquet datasets/boun-processed/
    fi

    # 移动 Open Images Parquet 文件
    mkdir -p datasets/open_images_v6
    mv downloads/open_images_v6-*.parquet datasets/open_images_v6/ 2>/dev/null || true

    echo "=========================================="
    echo "解压完成!"
    echo "=========================================="
}

# 主函数
main() {
    case "${1:-all}" in
        deps)
            install_deps
            ;;
        download)
            download_datasets
            ;;
        extract)
            extract_datasets
            ;;
        all)
            install_deps
            download_datasets
            extract_datasets
            ;;
        *)
            echo "用法: $0 {deps|download|extract|all}"
            echo "  deps    - 仅安装依赖"
            echo "  download - 仅下载数据集"
            echo "  extract  - 仅解压数据集"
            echo "  all     - 全部执行 (默认)"
            exit 1
            ;;
    esac
}

main "$@"
