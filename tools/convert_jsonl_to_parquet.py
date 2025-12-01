"""
将 Localized Narratives JSONL 文件转换为 Parquet 格式
输出 x, y, dt 三列，供 dataset.py 加载

注意：
- 原始 JSONL 中的时间戳是秒
- dt = t[i] - t[i-1]，单位毫秒，dt[0] = 0
"""
import argparse
import math
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as pj
from pathlib import Path
from tqdm import tqdm


def compute_straight_dist(x_list: list, y_list: list) -> float:
    """计算起终点直线距离"""
    if len(x_list) < 2:
        return 0.0
    dx = x_list[-1] - x_list[0]
    dy = y_list[-1] - y_list[0]
    return math.sqrt(dx * dx + dy * dy)


def extract_trajectories(jsonl_file: Path, min_length: int = 10, min_straight_dist: float = 0.01):
    """从单个 JSONL 文件提取轨迹

    Returns:
        (x_data, y_data, dt_data, filtered_count)
    """
    try:
        table = pj.read_json(jsonl_file)
    except Exception as e:
        print(f"  Error reading {jsonl_file.name}: {e}")
        return [], [], [], 0

    if 'traces' not in table.column_names:
        return [], [], [], 0

    traces_col = table.column('traces')
    x_data, y_data, dt_data = [], [], []
    filtered_count = 0

    for i in tqdm(range(len(table)), desc="  Processing", leave=False):
        traces = traces_col[i].as_py()
        if not traces:
            continue

        for trace in traces:
            if not trace or len(trace) < min_length:
                continue

            x_list, y_list, t_list = [], [], []
            for point in trace:
                if 'x' in point and 'y' in point:
                    x_list.append(float(point['x']))
                    y_list.append(float(point['y']))
                    t_list.append(float(point.get('t', 0.0)))

            if len(x_list) >= min_length:
                # 过滤起终点距离过小的轨迹
                if compute_straight_dist(x_list, y_list) < min_straight_dist:
                    filtered_count += 1
                    continue

                # 计算时间差 dt (毫秒)
                # dt[0] = 0, dt[i] = t[i] - t[i-1]
                dt_list = [0.0]  # 第一个点 dt=0
                for j in range(1, len(t_list)):
                    dt = (t_list[j] - t_list[j-1]) * 1000  # 秒 -> 毫秒
                    dt_list.append(max(0.0, dt))  # 确保非负

                x_data.append(x_list)
                y_data.append(y_list)
                dt_data.append(dt_list)

    return x_data, y_data, dt_data, filtered_count


def save_parquet(x_data, y_data, dt_data, output_path: Path):
    """保存为 Parquet 文件"""
    parquet_table = pa.table({
        'x': pa.array(x_data, type=pa.list_(pa.float32())),
        'y': pa.array(y_data, type=pa.list_(pa.float32())),
        'dt': pa.array(dt_data, type=pa.list_(pa.float32())),
    })
    pq.write_table(parquet_table, output_path, compression='snappy')
    return len(x_data)


def convert_each(input_dir: str, output_dir: str = None, min_length: int = 10,
                  min_straight_dist: float = 0.01, delete_jsonl: bool = False):
    """每个 JSONL 文件转换为对应的 Parquet 文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files (one-to-one mode)")
    print(f"min_straight_dist: {min_straight_dist}")
    print("=" * 60)

    total = 0
    total_filtered = 0
    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] {jsonl_file.name}")
        x_data, y_data, dt_data, filtered = extract_trajectories(jsonl_file, min_length, min_straight_dist)
        if x_data:
            parquet_file = output_path / (jsonl_file.stem + ".parquet")
            total += save_parquet(x_data, y_data, dt_data, parquet_file)
        total_filtered += filtered
        if delete_jsonl:
            jsonl_file.unlink()

    print("=" * 60)
    print(f"Total trajectories: {total}")
    print(f"Filtered (straight_dist < {min_straight_dist}): {total_filtered}")


def convert_merged(input_dir: str, output_file: str, min_length: int = 10,
                   min_straight_dist: float = 0.01, delete_jsonl: bool = False):
    """所有 JSONL 文件合并为单个 Parquet 文件"""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files (merge mode)")
    print(f"Output: {output_path}")
    print(f"min_straight_dist: {min_straight_dist}")
    print("=" * 60)

    all_x, all_y, all_dt = [], [], []
    total_filtered = 0
    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] {jsonl_file.name}")
        x_data, y_data, dt_data, filtered = extract_trajectories(jsonl_file, min_length, min_straight_dist)
        all_x.extend(x_data)
        all_y.extend(y_data)
        all_dt.extend(dt_data)
        total_filtered += filtered
        if delete_jsonl:
            jsonl_file.unlink()

    print("=" * 60)

    if not all_x:
        print("No trajectories extracted!")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = save_parquet(all_x, all_y, all_dt, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total trajectories: {total}")
    print(f"Filtered (straight_dist < {min_straight_dist}): {total_filtered}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet")
    parser.add_argument("--input_dir", type=str, default="datasets/open_images_v6")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（一对一模式）")
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--min_straight_dist", type=float, default=0.01,
                        help="最小起终点直线距离（归一化后），过滤太短的轨迹")
    parser.add_argument("--delete_jsonl", action="store_true")
    parser.add_argument("--merge", type=str, nargs='?', const='trajectories.parquet', default=None,
                        help="合并为单个 Parquet 文件（可指定文件名，默认 trajectories.parquet）")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent  # 项目根目录
    input_dir = base_dir / args.input_dir

    if args.merge is not None:
        output_file = input_dir / args.merge
        convert_merged(
            input_dir=str(input_dir),
            output_file=str(output_file),
            min_length=args.min_length,
            min_straight_dist=args.min_straight_dist,
            delete_jsonl=args.delete_jsonl,
        )
    else:
        output_dir = (base_dir / args.output_dir) if args.output_dir else None
        convert_each(
            input_dir=str(input_dir),
            output_dir=str(output_dir) if output_dir else None,
            min_length=args.min_length,
            min_straight_dist=args.min_straight_dist,
            delete_jsonl=args.delete_jsonl,
        )


if __name__ == "__main__":
    main()
