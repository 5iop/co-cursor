"""
BOUN数据集预处理脚本
按Release事件分割轨迹，去掉Drag和短轨迹
输出Parquet格式（PyArrow内部并行处理）

列结构：
- trajectory_id: uint32 - 全局轨迹ID
- user_id: string - 用户ID
- test_type: string - 测试类型
- session_id: string - 会话ID
- x: list<float32> - x坐标序列（归一化）
- y: list<float32> - y坐标序列（归一化）
- t: list<float32> - 时间戳序列（秒，相对于片段开始）
"""
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
from typing import List, Tuple, Dict

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

# 屏幕尺寸（用于过滤和归一化）
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


def process_session(csv_path: Path, min_steps: int = 10) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    处理单个session文件，按Released分割轨迹

    Args:
        csv_path: CSV文件路径
        min_steps: 最小步数阈值

    Returns:
        有效轨迹列表，每个元素是 (x, y, t) 三个数组
    """
    try:
        # 使用PyArrow读取CSV
        table = pv.read_csv(
            csv_path,
            read_options=pv.ReadOptions(use_threads=True),
            convert_options=pv.ConvertOptions(
                include_columns=['client_timestamp', 'x', 'y', 'state']
            )
        )

        x_data = table.column('x').to_numpy()
        y_data = table.column('y').to_numpy()
        t_data = table.column('client_timestamp').to_numpy()
        states = table.column('state').to_pylist()

    except Exception as e:
        return []

    if len(x_data) == 0:
        return []

    # 向量化：找到所有Released事件的索引
    states_arr = np.array(states)
    released_indices = np.where(states_arr == 'Released')[0]

    if len(released_indices) == 0:
        return []

    valid_trajectories = []
    start_idx = 0

    for end_idx in released_indices:
        segment_states = states_arr[start_idx:end_idx + 1]

        # 检查是否包含Drag
        if np.any(segment_states == 'Drag'):
            start_idx = end_idx + 1
            continue

        if len(segment_states) < min_steps:
            start_idx = end_idx + 1
            continue

        # 只保留Move事件
        move_mask = segment_states == 'Move'
        move_count = np.sum(move_mask)

        if move_count < min_steps:
            start_idx = end_idx + 1
            continue

        # 提取 Move 事件的数据（使用全局索引）
        global_move_mask = np.zeros(len(states_arr), dtype=bool)
        global_move_mask[start_idx:end_idx + 1] = move_mask

        x_vals = x_data[global_move_mask].astype(np.float32)
        y_vals = y_data[global_move_mask].astype(np.float32)
        t_vals = t_data[global_move_mask].astype(np.float64)

        # 检查坐标是否在有效范围内
        if (np.all(x_vals >= 0) and np.all(x_vals <= SCREEN_WIDTH) and
            np.all(y_vals >= 0) and np.all(y_vals <= SCREEN_HEIGHT)):

            # 归一化坐标
            x_norm = x_vals / SCREEN_WIDTH
            y_norm = y_vals / SCREEN_HEIGHT

            # 转换时间戳：毫秒转秒，相对于片段开始
            t_norm = ((t_vals - t_vals[0]) / 1000.0).astype(np.float32)

            valid_trajectories.append((x_norm, y_norm, t_norm))

        start_idx = end_idx + 1

    return valid_trajectories


def preprocess_boun(
    input_dir: str,
    output_dir: str,
    min_steps: int = 10,
    clean_output: bool = True,
):
    """
    预处理整个BOUN数据集（单线程，PyArrow内部并行）

    Args:
        input_dir: BOUN数据集根目录
        output_dir: 输出目录
        min_steps: 最小步数阈值
        clean_output: 是否清空输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 清空或创建输出目录
    if clean_output and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有CSV文件
    csv_files = list(input_path.glob("users/*/external_tests/*.csv"))
    csv_files += list(input_path.glob("users/*/internal_tests/*.csv"))
    csv_files += list(input_path.glob("users/*/training/*.csv"))

    print(f"Found {len(csv_files)} session files")

    # 直接收集数据到列表
    trajectory_ids = []
    user_ids = []
    test_types = []
    session_ids = []
    x_lists = []
    y_lists = []
    t_lists = []

    total_sessions = 0
    failed_sessions = 0
    total_points = 0
    global_id = 0

    # 单线程处理所有文件
    for csv_path in tqdm(csv_files, desc="Processing"):
        total_sessions += 1

        # 提取用户ID和测试类型
        parts = csv_path.parts
        user_idx = next((i for i, p in enumerate(parts) if re.match(r'^user\d+$', p)), None)

        if user_idx is None:
            failed_sessions += 1
            continue

        user_id = parts[user_idx]
        test_type = parts[user_idx + 1]
        session_id = csv_path.stem

        try:
            trajectories = process_session(csv_path, min_steps)

            for x, y, t in trajectories:
                trajectory_ids.append(global_id)
                user_ids.append(user_id)
                test_types.append(test_type)
                session_ids.append(session_id)
                x_lists.append(x.tolist())
                y_lists.append(y.tolist())
                t_lists.append(t.tolist())

                total_points += len(x)
                global_id += 1

        except Exception as e:
            failed_sessions += 1

    print(f"\nTotal trajectories: {global_id}")
    print("Building Parquet table...")

    # 定义schema
    schema = pa.schema([
        ('trajectory_id', pa.uint32()),
        ('user_id', pa.string()),
        ('test_type', pa.string()),
        ('session_id', pa.string()),
        ('x', pa.list_(pa.float32())),
        ('y', pa.list_(pa.float32())),
        ('t', pa.list_(pa.float32())),
    ])

    # 创建表
    table = pa.table({
        'trajectory_id': pa.array(trajectory_ids, type=pa.uint32()),
        'user_id': pa.array(user_ids, type=pa.string()),
        'test_type': pa.array(test_types, type=pa.string()),
        'session_id': pa.array(session_ids, type=pa.string()),
        'x': pa.array(x_lists, type=pa.list_(pa.float32())),
        'y': pa.array(y_lists, type=pa.list_(pa.float32())),
        't': pa.array(t_lists, type=pa.list_(pa.float32())),
    }, schema=schema)

    # 写入Parquet文件
    parquet_file = output_path / "boun_trajectories.parquet"
    pq.write_table(
        table,
        parquet_file,
        compression='zstd',
        compression_level=3,
    )

    print(f"\nParquet file saved to: {parquet_file}")
    print(f"File size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("BOUN Dataset Preprocessing Complete")
    print("=" * 60)
    print(f"Total sessions processed: {total_sessions}")
    print(f"Failed sessions: {failed_sessions}")
    print(f"Total valid trajectories: {global_id}")
    print(f"Total data points: {total_points}")
    print(f"Average trajectory length: {total_points / max(1, global_id):.1f}")

    # 保存统计信息
    stats_file = output_path / "preprocessing_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"BOUN Dataset Preprocessing Statistics\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Output format: Parquet\n")
        f.write(f"Output file: {parquet_file}\n")
        f.write(f"Min steps threshold: {min_steps}\n")
        f.write(f"Total sessions: {total_sessions}\n")
        f.write(f"Failed sessions: {failed_sessions}\n")
        f.write(f"Total trajectories: {global_id}\n")
        f.write(f"Total points: {total_points}\n")

    print(f"Stats saved to {stats_file}")

    return {
        'total_sessions': total_sessions,
        'failed_sessions': failed_sessions,
        'total_trajectories': global_id,
        'total_points': total_points,
    }


def analyze_dataset(input_dir: str):
    """分析原始数据集的状态分布"""
    input_path = Path(input_dir)

    csv_files = list(input_path.glob("users/*/external_tests/*.csv"))[:10]

    state_counts = {}
    segment_lengths = []

    for csv_file in tqdm(csv_files, desc="Analyzing"):
        try:
            table = pv.read_csv(csv_file)
            states = table.column('state').to_pylist()

            for state in set(states):
                count = states.count(state)
                state_counts[state] = state_counts.get(state, 0) + count

            # 统计Released之间的距离
            released_idx = [i for i, s in enumerate(states) if s == 'Released']
            prev_idx = 0
            for idx in released_idx:
                segment_lengths.append(idx - prev_idx)
                prev_idx = idx + 1

        except Exception as e:
            print(f"Error: {e}")

    print("\n--- State Distribution (sample) ---")
    total = sum(state_counts.values())
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count} ({100*count/total:.1f}%)")

    if segment_lengths:
        print(f"\n--- Segment Lengths (between Released) ---")
        print(f"  Min: {min(segment_lengths)}")
        print(f"  Max: {max(segment_lengths)}")
        print(f"  Mean: {np.mean(segment_lengths):.1f}")
        print(f"  Median: {np.median(segment_lengths):.1f}")
        print(f"  Segments >= 10: {sum(1 for l in segment_lengths if l >= 10)}/{len(segment_lengths)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BOUN mouse dataset to Parquet")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/boun-mouse-dynamics-dataset",
        help="Input BOUN dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/boun-processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=10,
        help="Minimum number of Move events per trajectory"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help="Do not clean output directory before processing"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze the dataset without processing"
    )

    args = parser.parse_args()

    # 项目根目录
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_dir = base_dir / args.output_dir

    if args.analyze:
        analyze_dataset(str(input_dir))
    else:
        preprocess_boun(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            min_steps=args.min_steps,
            clean_output=not args.no_clean,
        )


if __name__ == "__main__":
    main()
