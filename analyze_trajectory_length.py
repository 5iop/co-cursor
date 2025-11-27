"""
轨迹长度分布分析脚本
用于检查数据集中每个轨迹包含多少点，帮助决定mask长度
支持多线程加速
"""
import argparse

import orjson
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

NUM_THREADS = 16


def analyze_jsonl_file(file_path: Path) -> list:
    """分析单个JSONL文件，返回所有轨迹的点数列表"""
    point_counts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = orjson.loads(line)
                    traces = record.get('traces', [])

                    for trace in traces:
                        if trace:
                            point_counts.append(len(trace))

                except orjson.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return point_counts


def print_distribution(point_counts: list, dataset_name: str):
    """打印点数分布统计"""
    if not point_counts:
        print(f"\n{dataset_name}: 没有找到轨迹数据")
        return

    counts = np.array(point_counts)

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    print(f"轨迹总数: {len(counts):,}")
    print(f"\n--- 基本统计 ---")
    print(f"最小点数: {counts.min()}")
    print(f"最大点数: {counts.max()}")
    print(f"平均点数: {counts.mean():.1f}")
    print(f"中位数:   {np.median(counts):.1f}")
    print(f"标准差:   {counts.std():.1f}")

    print(f"\n--- 百分位数 ---")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(counts, p)
        print(f"  {p:2d}%: {value:,.0f} 点")

    print(f"\n--- 分布区间 ---")
    bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, float('inf')]
    bin_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000-2000', '2000-5000', '5000+']

    for i in range(len(bins) - 1):
        mask = (counts >= bins[i]) & (counts < bins[i+1])
        count = mask.sum()
        pct = count / len(counts) * 100
        bar = '#' * int(pct / 2)
        print(f"  {bin_labels[i]:>10}: {count:>6,} ({pct:5.1f}%) {bar}")

    # 推荐的mask长度
    print(f"\n--- 推荐的 mask/seq_length ---")
    print(f"  保守 (覆盖50%): {int(np.percentile(counts, 50))}")
    print(f"  平衡 (覆盖75%): {int(np.percentile(counts, 75))}")
    print(f"  宽松 (覆盖90%): {int(np.percentile(counts, 90))}")
    print(f"  激进 (覆盖95%): {int(np.percentile(counts, 95))}")


def analyze_dataset(data_dir: Path, dataset_name: str, num_threads: int = NUM_THREADS) -> list:
    """分析整个数据集目录（多线程）"""
    jsonl_files = list(data_dir.glob("**/*.jsonl"))

    if not jsonl_files:
        print(f"在 {data_dir} 中没有找到JSONL文件")
        return []

    print(f"\n正在分析 {dataset_name}...")
    print(f"找到 {len(jsonl_files)} 个JSONL文件，使用 {num_threads} 线程")

    all_counts = []

    # 多线程并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(analyze_jsonl_file, f): f for f in jsonl_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="读取文件"):
            counts = future.result()
            all_counts.extend(counts)

    return all_counts


def main():
    parser = argparse.ArgumentParser(description="分析轨迹长度分布")
    parser.add_argument(
        "--boun",
        type=str,
        default="datasets/boun-processed",
        help="BOUN处理后的数据集目录"
    )
    parser.add_argument(
        "--open-images",
        type=str,
        default="datasets/open_images_v6",
        help="Open Images数据集目录"
    )
    parser.add_argument(
        "--custom",
        type=str,
        default=None,
        help="自定义数据集目录"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=NUM_THREADS,
        help=f"线程数 (默认: {NUM_THREADS})"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="分析所有数据集的合并统计"
    )

    args = parser.parse_args()

    all_counts = []
    datasets_analyzed = []

    # 分析BOUN数据集
    boun_path = Path(args.boun)
    if boun_path.exists():
        counts = analyze_dataset(boun_path, "BOUN (处理后)", args.threads)
        if counts:
            print_distribution(counts, "BOUN (处理后)")
            all_counts.extend(counts)
            datasets_analyzed.append("BOUN")
    else:
        print(f"BOUN目录不存在: {boun_path}")

    # 分析Open Images数据集
    open_images_path = Path(args.open_images)
    if open_images_path.exists():
        counts = analyze_dataset(open_images_path, "Open Images V6", args.threads)
        if counts:
            print_distribution(counts, "Open Images V6")
            all_counts.extend(counts)
            datasets_analyzed.append("Open Images")
    else:
        print(f"Open Images目录不存在: {open_images_path}")

    # 分析自定义数据集
    if args.custom:
        custom_path = Path(args.custom)
        if custom_path.exists():
            counts = analyze_dataset(custom_path, f"自定义 ({custom_path.name})", args.threads)
            if counts:
                print_distribution(counts, f"自定义 ({custom_path.name})")
                all_counts.extend(counts)
                datasets_analyzed.append("Custom")
        else:
            print(f"自定义目录不存在: {custom_path}")

    # 合并统计
    if args.all and len(datasets_analyzed) > 1:
        print_distribution(all_counts, f"合并统计 ({', '.join(datasets_analyzed)})")

    # 总结
    if all_counts:
        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        print(f"分析的数据集: {', '.join(datasets_analyzed)}")
        print(f"总轨迹数: {len(all_counts):,}")
        print(f"\n根据数据分布，建议:")
        p50 = int(np.percentile(all_counts, 50))
        p90 = int(np.percentile(all_counts, 90))
        print(f"  - 如果使用固定长度重采样: seq_length = 50-100 (当前模型默认)")
        print(f"  - 如果使用变长mask: 最大长度建议设为 {p90} (覆盖90%数据)")
        print(f"  - 中位数点数: {p50}")


if __name__ == "__main__":
    main()
