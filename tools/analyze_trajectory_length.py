"""
轨迹长度和复杂度分布分析脚本
用于检查数据集中每个轨迹包含多少点，以及轨迹复杂度分布
支持 Parquet/CSV 格式，使用 PyArrow 读取

复杂度计算:
  path_ratio = path_length / straight_dist
  complexity = (path_ratio - 1) / path_ratio = 1 - 1/path_ratio
"""
import argparse
import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pv
from pathlib import Path
from tqdm import tqdm

DISTANCE_THRESHOLD = 100.0  # SapiMouse 轨迹分割阈值


def compute_complexity(coords: np.ndarray) -> float:
    """
    计算轨迹复杂度

    Args:
        coords: (N, 2) 轨迹坐标

    Returns:
        complexity: 复杂度值 [0, 1)
    """
    if len(coords) < 2:
        return 0.0

    # 计算路径长度
    segments = coords[1:] - coords[:-1]
    path_length = np.sum(np.linalg.norm(segments, axis=1))

    # 计算起终点直线距离
    straight_dist = np.linalg.norm(coords[-1] - coords[0])

    if straight_dist < 1e-8:
        # 起终点重合，使用路径长度作为复杂度指标
        return min(path_length, 1.0)

    # path_ratio = path_length / straight_dist
    path_ratio = path_length / straight_dist

    # complexity = (ratio - 1) / ratio = 1 - 1/ratio
    complexity = (path_ratio - 1.0) / path_ratio

    return complexity


def analyze_parquet_file(file_path: Path) -> tuple:
    """分析单个Parquet文件（新格式：x, y 为 list 列）

    Returns:
        (point_counts, complexities): 点数列表和复杂度列表
    """
    point_counts = []
    complexities = []

    try:
        table = pq.read_table(file_path, columns=['x', 'y'])
        x_col = table.column('x')
        y_col = table.column('y')

        for i in range(len(table)):
            x_list = x_col[i].as_py()
            y_list = y_col[i].as_py()
            if x_list and y_list:
                point_counts.append(len(x_list))
                coords = np.column_stack([x_list, y_list]).astype(np.float32)
                complexities.append(compute_complexity(coords))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return point_counts, complexities


def analyze_csv_file(file_path: Path, min_length: int = 10) -> tuple:
    """分析单个SapiMouse CSV文件，按距离分割轨迹

    Returns:
        (point_counts, complexities): 点数列表和复杂度列表
    """
    point_counts = []
    complexities = []

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
        coords = np.column_stack([x[move_mask], y[move_mask]]).astype(np.float32)

        if len(coords) < 2:
            return point_counts, complexities

        # 按距离分割轨迹
        diffs = np.diff(coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        split_indices = np.where(distances > DISTANCE_THRESHOLD)[0] + 1
        split_indices = np.concatenate([[0], split_indices, [len(coords)]])

        for i in range(len(split_indices) - 1):
            start, end = split_indices[i], split_indices[i + 1]
            length = end - start
            if length >= min_length:
                point_counts.append(length)
                traj_coords = coords[start:end]
                complexities.append(compute_complexity(traj_coords))

    except Exception:
        pass  # 静默处理错误

    return point_counts, complexities


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
    print(f"\n--- 基本统计 (点数) ---")
    print(f"最小点数: {counts.min()}")
    print(f"最大点数: {counts.max()}")
    print(f"平均点数: {counts.mean():.1f}")
    print(f"中位数:   {np.median(counts):.1f}")
    print(f"标准差:   {counts.std():.1f}")

    print(f"\n--- 百分位数 (点数) ---")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(counts, p)
        print(f"  {p:2d}%: {value:,.0f} 点")

    print(f"\n--- 分布区间 (点数) ---")
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


def print_complexity_distribution(complexities: list, dataset_name: str):
    """打印复杂度分布统计"""
    if not complexities:
        print(f"\n{dataset_name}: 没有复杂度数据")
        return

    comp = np.array(complexities)

    print(f"\n--- 复杂度分布 (complexity = 1 - 1/ratio) ---")
    print(f"最小: {comp.min():.4f}")
    print(f"最大: {comp.max():.4f}")
    print(f"平均: {comp.mean():.4f}")
    print(f"中位: {np.median(comp):.4f}")
    print(f"标准差: {comp.std():.4f}")

    print(f"\n--- 百分位数 (复杂度) ---")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(comp, p)
        # 反推 path_ratio: complexity = 1 - 1/ratio => ratio = 1/(1-complexity)
        if value < 1.0:
            ratio = 1.0 / (1.0 - value + 1e-8)
        else:
            ratio = float('inf')
        print(f"  {p:2d}%: {value:.4f} (ratio ≈ {ratio:.2f})")

    print(f"\n--- 分布区间 (复杂度) ---")
    # 复杂度区间及对应的 alpha 参考值
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins) - 1):
        mask = (comp >= bins[i]) & (comp < bins[i+1])
        count = mask.sum()
        pct = count / len(comp) * 100
        bar = '#' * int(pct / 2)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {count:>6,} ({pct:5.1f}%) {bar}")

    # 与 alpha 的对应关系建议
    print(f"\n--- 与模型 alpha 参数的对应 ---")
    print(f"  如果使用 alpha=0.3, 目标 complexity ≈ 0.3")
    print(f"  如果使用 alpha=0.5, 目标 complexity ≈ 0.5")
    print(f"  如果使用 alpha=0.7, 目标 complexity ≈ 0.7")
    print(f"\n  人类数据复杂度中位数: {np.median(comp):.4f}")
    print(f"  建议使用 alpha ≈ {np.median(comp):.2f} 以匹配人类分布")


def analyze_dataset(data_dir: Path, dataset_name: str, file_type: str = "auto") -> tuple:
    """
    分析整个数据集目录

    Args:
        data_dir: 数据集目录
        dataset_name: 数据集名称
        file_type: 文件类型 ("parquet", "csv", "auto")

    Returns:
        (point_counts, complexities): 点数列表和复杂度列表
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
            return [], []
    elif file_type == "parquet":
        files = list(data_dir.glob("**/*.parquet"))
        analyze_func, file_type_name = analyze_parquet_file, "Parquet"
    elif file_type == "csv":
        files = list(data_dir.glob("**/*.csv"))
        analyze_func, file_type_name = analyze_csv_file, "CSV"
    else:
        print(f"未知文件类型: {file_type}")
        return [], []

    if not files:
        print(f"在 {data_dir} 中没有找到 {file_type_name} 文件")
        return [], []

    print(f"\n正在分析 {dataset_name}...")
    print(f"找到 {len(files)} 个 {file_type_name} 文件")

    all_counts = []
    all_complexities = []

    for file_path in tqdm(files, desc="读取文件"):
        counts, complexities = analyze_func(file_path)
        all_counts.extend(counts)
        all_complexities.extend(complexities)

    return all_counts, all_complexities


def main():
    parser = argparse.ArgumentParser(description="分析轨迹长度和复杂度分布")
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
    all_complexities = []
    datasets_analyzed = []

    # 分析SapiMouse数据集
    sapimouse_path = base_dir / args.sapimouse
    if sapimouse_path.exists():
        counts, complexities = analyze_dataset(sapimouse_path, "SapiMouse", file_type="csv")
        if counts:
            print_distribution(counts, "SapiMouse")
            print_complexity_distribution(complexities, "SapiMouse")
            all_counts.extend(counts)
            all_complexities.extend(complexities)
            datasets_analyzed.append("SapiMouse")
    else:
        print(f"SapiMouse目录不存在: {sapimouse_path}")

    # 分析BOUN数据集
    boun_path = base_dir / args.boun
    if boun_path.exists():
        counts, complexities = analyze_dataset(boun_path, "BOUN (处理后)")
        if counts:
            print_distribution(counts, "BOUN (处理后)")
            print_complexity_distribution(complexities, "BOUN (处理后)")
            all_counts.extend(counts)
            all_complexities.extend(complexities)
            datasets_analyzed.append("BOUN")
    else:
        print(f"BOUN目录不存在: {boun_path}")

    # 分析Open Images数据集
    open_images_path = base_dir / args.open_images
    if open_images_path.exists():
        counts, complexities = analyze_dataset(open_images_path, "Open Images V6")
        if counts:
            print_distribution(counts, "Open Images V6")
            print_complexity_distribution(complexities, "Open Images V6")
            all_counts.extend(counts)
            all_complexities.extend(complexities)
            datasets_analyzed.append("Open Images")
    else:
        print(f"Open Images目录不存在: {open_images_path}")

    # 分析自定义数据集
    if args.custom:
        custom_path = base_dir / args.custom
        if custom_path.exists():
            counts, complexities = analyze_dataset(custom_path, f"自定义 ({custom_path.name})")
            if counts:
                print_distribution(counts, f"自定义 ({custom_path.name})")
                print_complexity_distribution(complexities, f"自定义 ({custom_path.name})")
                all_counts.extend(counts)
                all_complexities.extend(complexities)
                datasets_analyzed.append("Custom")
        else:
            print(f"自定义目录不存在: {custom_path}")

    # 合并统计
    if args.all and len(datasets_analyzed) > 1:
        print_distribution(all_counts, f"合并统计 ({', '.join(datasets_analyzed)})")
        print_complexity_distribution(all_complexities, f"合并统计 ({', '.join(datasets_analyzed)})")

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

        if all_complexities:
            median_comp = np.median(all_complexities)
            print(f"\n  - 人类轨迹复杂度中位数: {median_comp:.4f}")
            print(f"  - 推荐 alpha 值以匹配人类分布: {median_comp:.2f}")


if __name__ == "__main__":
    main()
