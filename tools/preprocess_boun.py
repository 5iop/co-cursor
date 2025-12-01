"""
BOUN数据集预处理脚本
按Release事件分割轨迹，去掉Drag和短轨迹
输出Parquet格式（PyArrow内部并行处理）

列结构：
- x: list<float32> - x坐标序列（归一化）
- y: list<float32> - y坐标序列（归一化）
- dt: list<float32> - 时间差序列（毫秒，dt[0]=0, dt[i]=t[i]-t[i-1]）
- user_id: string - 用户ID
"""
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

# 屏幕尺寸（用于过滤和归一化）
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


def compute_straight_dist(x_norm: np.ndarray, y_norm: np.ndarray) -> float:
    """计算起终点直线距离（归一化坐标）"""
    if len(x_norm) < 2:
        return 0.0
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    return np.sqrt(dx * dx + dy * dy)


def process_session(csv_path: Path, min_steps: int = 10, min_straight_dist: float = 0.01) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """
    处理单个session文件，按Released分割轨迹

    Args:
        csv_path: CSV文件路径
        min_steps: 最小步数阈值
        min_straight_dist: 最小起终点直线距离（归一化后）

    Returns:
        (有效轨迹列表, 被过滤的轨迹数)
    """
    filtered_count = 0
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
        return [], 0

    if len(x_data) == 0:
        return [], 0

    # 向量化：找到所有Released事件的索引
    states_arr = np.array(states)
    released_indices = np.where(states_arr == 'Released')[0]

    if len(released_indices) == 0:
        return [], 0

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

            # 过滤起终点距离过小的轨迹
            if compute_straight_dist(x_norm, y_norm) < min_straight_dist:
                filtered_count += 1
                start_idx = end_idx + 1
                continue

            # 计算时间差 dt (毫秒)
            # dt[0] = 0, dt[i] = t[i] - t[i-1]
            # BOUN 原始数据已是毫秒
            dt = np.zeros_like(t_vals, dtype=np.float32)
            dt[1:] = np.maximum(0, np.diff(t_vals))  # 确保非负

            valid_trajectories.append((x_norm, y_norm, dt))

        start_idx = end_idx + 1

    return valid_trajectories, filtered_count


def preprocess_boun(
    input_dir: str,
    output_dir: str,
    min_steps: int = 10,
    min_straight_dist: float = 0.01,
    users: List[str] = None,
):
    """
    预处理整个BOUN数据集（单线程，PyArrow内部并行）

    Args:
        input_dir: BOUN数据集根目录
        output_dir: 输出目录
        min_steps: 最小步数阈值
        min_straight_dist: 最小起终点直线距离（归一化后）
        users: 只处理指定的用户ID列表 (如 ["user1", "user2"])，None表示处理所有用户
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有CSV文件
    if users:
        # 只处理指定用户
        csv_files = []
        for user_id in users:
            csv_files += list(input_path.glob(f"users/{user_id}/external_tests/*.csv"))
            csv_files += list(input_path.glob(f"users/{user_id}/internal_tests/*.csv"))
            csv_files += list(input_path.glob(f"users/{user_id}/training/*.csv"))
        print(f"Filtering by users: {users}")
    else:
        csv_files = list(input_path.glob("users/*/external_tests/*.csv"))
        csv_files += list(input_path.glob("users/*/internal_tests/*.csv"))
        csv_files += list(input_path.glob("users/*/training/*.csv"))

    print(f"Found {len(csv_files)} session files")
    print(f"min_straight_dist: {min_straight_dist}")

    # 直接收集数据到列表
    x_lists = []
    y_lists = []
    dt_lists = []
    user_ids = []

    total_sessions = 0
    failed_sessions = 0
    total_filtered = 0
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
            trajectories, filtered = process_session(csv_path, min_steps, min_straight_dist)
            total_filtered += filtered

            for x, y, dt in trajectories:
                x_lists.append(x.tolist())
                y_lists.append(y.tolist())
                dt_lists.append(dt.tolist())
                user_ids.append(user_id)

                total_points += len(x)
                global_id += 1

        except Exception as e:
            failed_sessions += 1

    print(f"\nTotal trajectories: {global_id}")
    print("Building Parquet table...")

    table = pa.table({
        'x': pa.array(x_lists, type=pa.list_(pa.float32())),
        'y': pa.array(y_lists, type=pa.list_(pa.float32())),
        'dt': pa.array(dt_lists, type=pa.list_(pa.float32())),
        'user_id': pa.array(user_ids, type=pa.string()),
    })

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
    print(f"Filtered (straight_dist < {min_straight_dist}): {total_filtered}")
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
        f.write(f"Min straight dist: {min_straight_dist}\n")
        f.write(f"Total sessions: {total_sessions}\n")
        f.write(f"Failed sessions: {failed_sessions}\n")
        f.write(f"Filtered trajectories: {total_filtered}\n")
        f.write(f"Total trajectories: {global_id}\n")
        f.write(f"Total points: {total_points}\n")

    print(f"Stats saved to {stats_file}")

    return {
        'total_sessions': total_sessions,
        'failed_sessions': failed_sessions,
        'total_filtered': total_filtered,
        'total_trajectories': global_id,
        'total_points': total_points,
    }


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
        "--min_straight_dist",
        type=float,
        default=0.01,
        help="Minimum straight-line distance between start and end (normalized)"
    )
    parser.add_argument(
        "--users",
        type=str,
        nargs="+",
        default=None,
        help="Only process specific users (e.g., --users user1 user2 user3)"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_dir = base_dir / args.output_dir

    preprocess_boun(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        min_steps=args.min_steps,
        min_straight_dist=args.min_straight_dist,
        users=args.users,
    )


if __name__ == "__main__":
    main()
