"""
轨迹长度分布分析脚本
用于检查数据集中每个轨迹包含多少点，帮助决定mask长度
支持 Parquet/CSV 格式，使用 PyArrow 读取
"""
import argparse
import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pv
from pathlib import Path
from tqdm import tqdm

DISTANCE_THRESHOLD = 100.0  # SapiMouse 轨迹分割阈值


def analyze_parquet_file(file_path: Path) -> list:
    """分析单个Parquet文件（新格式：x, y 为 list 列）"""
    point_counts = []

    try:
        table = pq.read_table(file_path, columns=['x'])
        x_col = table.column('x')

        for i in range(len(table)):
            x_list = x_col[i].as_py()
            if x_list:
                point_counts.append(len(x_list))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return point_counts


def analyze_csv_file(file_path: Path, min_length: int = 10) -> list:
    """分析单个SapiMouse CSV文件，按距离分割轨迹"""
    point_counts = []

    try:
        table = pv.read_csv(
            file_path,
            read_options=pv.ReadOptions(use_threads=True),
            convert_options=pv.ConvertOptions(
                include_columns=['x', 'y', 'state']
            )
        )

        x = table.column('x').to_numpy()
        y = table.column('y').to_numpy()
        states = table.column('state').to_pylist()

        # 只保留Move事件
        move_mask = np.array([s == 'Move' for s in states])
        coords = np.column_stack([x[move_mask], y[move_mask]])

        if len(coords) < 2:
            return point_counts

        # 按距离分割轨迹
        diffs = np.diff(coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        split_indices = np.where(distances > DISTANCE_THRESHOLD)[0] + 1
        split_indices = np.concatenate([[0], split_indices, [len(coords)]])

        for i in range(len(split_indices) - 1):
            length = split_indices[i + 1] - split_indices[i]
            if length >= min_length:
                point_counts.append(length)

    except Exception:
        pass  # 静默处理错误

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


def analyze_dataset(data_dir: Path, dataset_name: str, file_type: str = "auto") -> list:
    """
    分析整个数据集目录

    Args:
        data_dir: 数据集目录
        dataset_name: 数据集名称
        file_type: 文件类型 ("parquet", "csv", "auto")
    """
    # 根据文件类型查找文件
    if file_type == "auto":
        parquet_files = list(data_dir.glob("**/*.parquet"))
        csv_files = list(data_dir.glob("**/*.csv"))

        if parquet_files:
            files, analyze_func, file_type_name = parquet_files, analyze_parquet_file, "Parquet"
        elif csv_files:
            files, analyze_func, file_type_name = csv_files, analyze_csv_file, "CSV"
        else:
            print(f"在 {data_dir} 中没有找到数据文件")
            return []
    elif file_type == "parquet":
        files = list(data_dir.glob("**/*.parquet"))
        analyze_func, file_type_name = analyze_parquet_file, "Parquet"
    elif file_type == "csv":
        files = list(data_dir.glob("**/*.csv"))
        analyze_func, file_type_name = analyze_csv_file, "CSV"
    else:
        print(f"未知文件类型: {file_type}")
        return []

    if not files:
        print(f"在 {data_dir} 中没有找到 {file_type_name} 文件")
        return []

    print(f"\n正在分析 {dataset_name}...")
    print(f"找到 {len(files)} 个 {file_type_name} 文件")

    all_counts = []

    for file_path in tqdm(files, desc="读取文件"):
        counts = analyze_func(file_path)
        all_counts.extend(counts)

    return all_counts


def main():
    parser = argparse.ArgumentParser(description="分析轨迹长度分布")
    parser.add_argument(
        "--sapimouse",
        type=str,
        default="datasets/sapimouse",
        help="SapiMouse数据集目录"
    )
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
        "--all",
        action="store_true",
        help="分析所有数据集的合并统计"
    )

    args = parser.parse_args()

    # 项目根目录
    base_dir = Path(__file__).parent.parent

    all_counts = []
    datasets_analyzed = []

    # 分析SapiMouse数据集
    sapimouse_path = base_dir / args.sapimouse
    if sapimouse_path.exists():
        counts = analyze_dataset(sapimouse_path, "SapiMouse", file_type="csv")
        if counts:
            print_distribution(counts, "SapiMouse")
            all_counts.extend(counts)
            datasets_analyzed.append("SapiMouse")
    else:
        print(f"SapiMouse目录不存在: {sapimouse_path}")

    # 分析BOUN数据集
    boun_path = base_dir / args.boun
    if boun_path.exists():
        counts = analyze_dataset(boun_path, "BOUN (处理后)")
        if counts:
            print_distribution(counts, "BOUN (处理后)")
            all_counts.extend(counts)
            datasets_analyzed.append("BOUN")
    else:
        print(f"BOUN目录不存在: {boun_path}")

    # 分析Open Images数据集
    open_images_path = base_dir / args.open_images
    if open_images_path.exists():
        counts = analyze_dataset(open_images_path, "Open Images V6")
        if counts:
            print_distribution(counts, "Open Images V6")
            all_counts.extend(counts)
            datasets_analyzed.append("Open Images")
    else:
        print(f"Open Images目录不存在: {open_images_path}")

    # 分析自定义数据集
    if args.custom:
        custom_path = base_dir / args.custom
        if custom_path.exists():
            counts = analyze_dataset(custom_path, f"自定义 ({custom_path.name})")
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
