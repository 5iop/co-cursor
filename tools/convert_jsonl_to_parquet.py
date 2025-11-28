"""
将 Localized Narratives JSONL 文件转换为 Parquet 格式
输出 x, y, t 三列，供 dataset.py 加载
"""
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as pj
from pathlib import Path
from tqdm import tqdm


def extract_trajectories(jsonl_file: Path, min_length: int = 10):
    """从单个 JSONL 文件提取轨迹"""
    try:
        table = pj.read_json(jsonl_file)
    except Exception as e:
        print(f"  Error reading {jsonl_file.name}: {e}")
        return [], [], []

    if 'traces' not in table.column_names:
        return [], [], []

    traces_col = table.column('traces')
    x_data, y_data, t_data = [], [], []

    for i in range(len(table)):
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
                # 时间戳归一化：每个片段从 0 开始
                t_start = t_list[0]
                t_list = [t - t_start for t in t_list]

                x_data.append(x_list)
                y_data.append(y_list)
                t_data.append(t_list)

    return x_data, y_data, t_data


def save_parquet(x_data, y_data, t_data, output_path: Path):
    """保存为 Parquet 文件"""
    parquet_table = pa.table({
        'x': pa.array(x_data, type=pa.list_(pa.float32())),
        'y': pa.array(y_data, type=pa.list_(pa.float32())),
        't': pa.array(t_data, type=pa.list_(pa.float32())),
    })
    pq.write_table(parquet_table, output_path, compression='snappy')
    return len(x_data)


def convert_each(input_dir: str, output_dir: str = None, min_length: int = 10, delete_jsonl: bool = False):
    """每个 JSONL 文件转换为对应的 Parquet 文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files (one-to-one mode)")
    print("=" * 60)

    total = 0
    for jsonl_file in tqdm(jsonl_files, desc="Converting"):
        x_data, y_data, t_data = extract_trajectories(jsonl_file, min_length)
        if x_data:
            parquet_file = output_path / (jsonl_file.stem + ".parquet")
            total += save_parquet(x_data, y_data, t_data, parquet_file)
        if delete_jsonl:
            jsonl_file.unlink()

    print("=" * 60)
    print(f"Total trajectories: {total}")


def convert_merged(input_dir: str, output_file: str, min_length: int = 10, delete_jsonl: bool = False):
    """所有 JSONL 文件合并为单个 Parquet 文件"""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files (merge mode)")
    print(f"Output: {output_path}")
    print("=" * 60)

    all_x, all_y, all_t = [], [], []
    for jsonl_file in tqdm(jsonl_files, desc="Extracting"):
        x_data, y_data, t_data = extract_trajectories(jsonl_file, min_length)
        all_x.extend(x_data)
        all_y.extend(y_data)
        all_t.extend(t_data)
        if delete_jsonl:
            jsonl_file.unlink()

    print("=" * 60)

    if not all_x:
        print("No trajectories extracted!")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = save_parquet(all_x, all_y, all_t, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total trajectories: {total}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet")
    parser.add_argument("--input_dir", type=str, default="datasets/open_images_v6")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（一对一模式）")
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--delete_jsonl", action="store_true")
    parser.add_argument("--merge", action="store_true", help="合并为单个 Parquet 文件")
    parser.add_argument("--merge_output", type=str, default="datasets/open_images_v6/trajectories.parquet",
                        help="合并模式输出文件路径")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent  # 项目根目录
    input_dir = base_dir / args.input_dir

    if args.merge:
        output_file = base_dir / args.merge_output
        convert_merged(
            input_dir=str(input_dir),
            output_file=str(output_file),
            min_length=args.min_length,
            delete_jsonl=args.delete_jsonl,
        )
    else:
        output_dir = (base_dir / args.output_dir) if args.output_dir else None
        convert_each(
            input_dir=str(input_dir),
            output_dir=str(output_dir) if output_dir else None,
            min_length=args.min_length,
            delete_jsonl=args.delete_jsonl,
        )


if __name__ == "__main__":
    main()
