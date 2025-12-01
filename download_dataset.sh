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
    sudo apt-get -y install aria2 vim btop nvtop
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

# 移动数据集到正确位置
organize_datasets() {
    echo "=========================================="
    echo "整理数据集..."
    echo "=========================================="

    # 移动 BOUN Parquet 文件
    if [ -f "downloads/boun_trajectories.parquet" ]; then
        echo "移动 BOUN Parquet 文件..."
        mkdir -p datasets/boun-processed
        mv downloads/boun_trajectories.parquet datasets/boun-processed/
        echo "  -> datasets/boun-processed/boun_trajectories.parquet"
    fi

    # 移动 Open Images V6 Parquet 文件
    if ls downloads/open_images_v6-*.parquet 1> /dev/null 2>&1; then
        echo "移动 Open Images V6 Parquet 文件..."
        mkdir -p datasets/open_images_v6
        for f in downloads/open_images_v6-*.parquet; do
            if [ -f "$f" ]; then
                mv "$f" datasets/open_images_v6/
                echo "  -> datasets/open_images_v6/$(basename $f)"
            fi
        done
    fi

    echo "=========================================="
    echo "整理完成!"
    echo "=========================================="
}

# 显示数据集状态
show_status() {
    echo "=========================================="
    echo "数据集状态"
    echo "=========================================="

    echo ""
    echo "BOUN 数据集:"
    if [ -f "datasets/boun-processed/boun_trajectories.parquet" ]; then
        ls -lh datasets/boun-processed/boun_trajectories.parquet
    else
        echo "  未找到"
    fi

    echo ""
    echo "Open Images V6 数据集:"
    if ls datasets/open_images_v6/*.parquet 1> /dev/null 2>&1; then
        ls -lh datasets/open_images_v6/*.parquet
    else
        echo "  未找到"
    fi

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
        organize)
            organize_datasets
            ;;
        status)
            show_status
            ;;
        all)
            install_deps
            download_datasets
            organize_datasets
            show_status
            ;;
        *)
            echo "用法: $0 {deps|download|organize|status|all}"
            echo "  deps     - 仅安装依赖"
            echo "  download - 仅下载数据集"
            echo "  organize - 仅整理数据集（移动到正确位置）"
            echo "  status   - 显示数据集状态"
            echo "  all      - 全部执行 (默认)"
            exit 1
            ;;
    esac
}

main "$@"
