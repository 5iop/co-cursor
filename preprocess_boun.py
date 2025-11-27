"""
BOUN数据集预处理脚本
按Release事件分割轨迹，去掉Drag和短轨迹
支持多线程并行处理
输出JSONL格式（参考Open Images V7）

内存优化：
- 只读取必要的CSV列
- 使用 numpy 数组代替 DataFrame
- 流式写入 JSONL，不在内存中累积
- 使用 float32 减少内存占用
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# 尝试使用更快的 JSON 库
try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

NUM_THREADS = 16

# 屏幕尺寸（用于过滤和归一化）
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


@dataclass
class TrajectoryRecord:
    """单条轨迹记录（JSONL格式）"""
    dataset_id: str
    user_id: str
    test_type: str
    session_id: str
    trajectory_id: int
    traces: List[List[Dict[str, float]]]  # [[{x, y, t}, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "test_type": self.test_type,
            "session_id": self.session_id,
            "trajectory_id": self.trajectory_id,
            "traces": self.traces,
        }


@dataclass
class ProcessResult:
    """单个文件处理结果"""
    csv_path: Path
    user_id: str
    test_type: str
    trajectories: List[pd.DataFrame]
    records: List[TrajectoryRecord]  # JSONL记录
    success: bool
    error_msg: Optional[str] = None


@dataclass
class UserProcessResult:
    """单个用户处理结果（仅统计信息，不存储数据）"""
    user_id: str
    total_sessions: int
    failed_sessions: int
    total_trajectories: int
    total_points: int


def process_session(csv_path: Path, min_steps: int = 100) -> List[np.ndarray]:
    """
    处理单个session文件，按Released分割轨迹（内存优化版本）

    Args:
        csv_path: CSV文件路径
        min_steps: 最小步数阈值

    Returns:
        有效轨迹列表，每个元素是 numpy array (N, 3) 包含 [x, y, timestamp]
    """
    try:
        # 只读取必要的列，减少内存占用
        df = pd.read_csv(csv_path, usecols=['client_timestamp', 'x', 'y', 'state'])
    except Exception as e:
        return []

    if len(df) == 0:
        return []

    # 直接使用 numpy 数组，避免 pandas 开销
    states = df['state'].values
    x_data = df['x'].values
    y_data = df['y'].values
    t_data = df['client_timestamp'].values

    # 释放 DataFrame 内存
    del df

    # 向量化：找到所有Released事件的索引
    released_indices = np.where(states == 'Released')[0]

    if len(released_indices) == 0:
        return []

    valid_trajectories = []
    start_idx = 0

    for end_idx in released_indices:
        segment_states = states[start_idx:end_idx + 1]

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
        global_move_mask = np.zeros(len(states), dtype=bool)
        global_move_mask[start_idx:end_idx + 1] = move_mask

        x_vals = x_data[global_move_mask]
        y_vals = y_data[global_move_mask]
        t_vals = t_data[global_move_mask]

        # 检查坐标是否在有效范围内
        if (np.all(x_vals >= 0) and np.all(x_vals <= SCREEN_WIDTH) and
            np.all(y_vals >= 0) and np.all(y_vals <= SCREEN_HEIGHT)):
            # 存储为紧凑的 numpy 数组 (N, 3)，使用 float32 节省内存
            traj_data = np.column_stack([x_vals, y_vals, t_vals]).astype(np.float32)
            valid_trajectories.append(traj_data)

        start_idx = end_idx + 1

    return valid_trajectories


def trajectory_to_traces(traj_data: np.ndarray) -> List[Dict[str, float]]:
    """
    将numpy数组轨迹转换为traces格式
    坐标归一化到 [0, 1] 范围

    Args:
        traj_data: numpy数组 (N, 3)，包含 [x, y, timestamp]

    Returns:
        traces列表，每个元素是{x, y, t}字典，坐标已归一化
    """
    if len(traj_data) == 0:
        return []

    # 归一化坐标到 [0, 1]
    x_normalized = traj_data[:, 0] / SCREEN_WIDTH
    y_normalized = traj_data[:, 1] / SCREEN_HEIGHT

    # 归一化时间（转换为秒）
    t_vals = traj_data[:, 2]
    t_normalized = (t_vals - t_vals[0]) / 1000.0

    # 构建traces列表
    traces = [
        {"x": round(float(x), 4), "y": round(float(y), 4), "t": round(float(t), 3)}
        for x, y, t in zip(x_normalized, y_normalized, t_normalized)
    ]

    return traces


def process_file_worker(args: Tuple[Path, int]) -> ProcessResult:
    """
    多线程worker函数，处理单个CSV文件（内存优化版本）

    Args:
        args: (csv_path, min_steps) 元组

    Returns:
        ProcessResult 对象
    """
    csv_path, min_steps = args

    # 提取用户ID和测试类型
    parts = csv_path.parts
    user_idx = next((i for i, p in enumerate(parts) if re.match(r'^user\d+$', p)), None)

    if user_idx is None:
        return ProcessResult(
            csv_path=csv_path,
            user_id="unknown",
            test_type="unknown",
            trajectories=[],
            records=[],
            success=False,
            error_msg="Could not extract user ID"
        )

    user_id = parts[user_idx]
    test_type = parts[user_idx + 1]
    session_id = csv_path.stem

    try:
        trajectories = process_session(csv_path, min_steps)

        # 创建JSONL记录
        records = []
        for i, traj_data in enumerate(trajectories):
            traces = trajectory_to_traces(traj_data)
            record = TrajectoryRecord(
                dataset_id="boun",
                user_id=user_id,
                test_type=test_type,
                session_id=session_id,
                trajectory_id=i,
                traces=[traces],
            )
            records.append(record)

        return ProcessResult(
            csv_path=csv_path,
            user_id=user_id,
            test_type=test_type,
            trajectories=[],  # 不再存储原始轨迹数据
            records=records,
            success=True
        )
    except Exception as e:
        return ProcessResult(
            csv_path=csv_path,
            user_id=user_id,
            test_type=test_type,
            trajectories=[],
            records=[],
            success=False,
            error_msg=str(e)
        )


def process_user_worker(args: Tuple[str, List[Path], int, Path]) -> UserProcessResult:
    """
    多线程worker函数，处理单个用户的所有CSV文件（内存优化版本）
    直接写入文件，不在内存中累积数据

    Args:
        args: (user_id, csv_files, min_steps, output_path) 元组

    Returns:
        UserProcessResult 对象（仅统计信息）
    """
    user_id, csv_files, min_steps, output_path = args

    total_sessions = 0
    failed_sessions = 0
    total_trajectories = 0
    total_points = 0

    total_files = len(csv_files)
    last_progress = 0

    # 直接写入文件，避免在内存中累积所有记录
    jsonl_file = output_path / f"{user_id}.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for idx, csv_path in enumerate(csv_files):
            total_sessions += 1

            # 提取测试类型
            parts = csv_path.parts
            user_idx = next((i for i, p in enumerate(parts) if re.match(r'^user\d+$', p)), None)
            test_type = parts[user_idx + 1] if user_idx is not None else "unknown"
            session_id = csv_path.stem

            try:
                trajectories = process_session(csv_path, min_steps)

                # 直接写入JSONL，不存储在内存中
                for i, traj_data in enumerate(trajectories):
                    traces = trajectory_to_traces(traj_data)
                    # 直接构建字典并写入，避免创建 dataclass 对象
                    record_dict = {
                        "dataset_id": "boun",
                        "user_id": user_id,
                        "test_type": test_type,
                        "session_id": session_id,
                        "trajectory_id": i,
                        "traces": [traces],
                    }
                    f.write(json_dumps(record_dict) + '\n')
                    total_trajectories += 1
                    total_points += len(traces)

                # 释放轨迹数据内存
                del trajectories

            except Exception as e:
                failed_sessions += 1

            # 每10%打印一次进度
            progress = int((idx + 1) * 100 / total_files)
            if progress >= last_progress + 10:
                last_progress = progress
                print(f"  {user_id}: {progress}% ({idx + 1}/{total_files} files, {total_trajectories} trajectories)")

    print(f"  {user_id}: Done ({total_sessions} sessions, {total_trajectories} trajectories, {total_points} points)")

    # 显式触发垃圾回收，释放内存
    gc.collect()

    return UserProcessResult(
        user_id=user_id,
        total_sessions=total_sessions,
        failed_sessions=failed_sessions,
        total_trajectories=total_trajectories,
        total_points=total_points
    )


def preprocess_boun(
    input_dir: str,
    output_dir: str,
    min_steps: int = 100,
    clean_output: bool = True,
    num_threads: int = NUM_THREADS,
):
    """
    预处理整个BOUN数据集（多线程版本）

    Args:
        input_dir: BOUN数据集根目录
        output_dir: 输出目录
        min_steps: 最小步数阈值
        clean_output: 是否清空输出目录
        num_threads: 并行线程数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 清空或创建输出目录
    if clean_output and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        'total_sessions': 0,
        'total_trajectories': 0,
        'total_points': 0,
        'skipped_drag': 0,
        'skipped_short': 0,
        'failed_sessions': 0,
        'by_user': {},
    }

    # 查找所有CSV文件
    csv_files = list(input_path.glob("users/*/external_tests/*.csv"))
    csv_files += list(input_path.glob("users/*/internal_tests/*.csv"))
    csv_files += list(input_path.glob("users/*/training/*.csv"))

    print(f"Found {len(csv_files)} session files")

    # 按用户分组CSV文件
    files_by_user: Dict[str, List[Path]] = {}
    for csv_file in csv_files:
        parts = csv_file.parts
        user_idx = next((i for i, p in enumerate(parts) if re.match(r'^user\d+$', p)), None)
        if user_idx is not None:
            user_id = parts[user_idx]
            if user_id not in files_by_user:
                files_by_user[user_id] = []
            files_by_user[user_id].append(csv_file)

    print(f"Found {len(files_by_user)} users")
    print(f"Using {num_threads} threads for parallel processing (one thread per user)")

    # 准备任务参数：每个用户一个任务
    task_args = [
        (user_id, user_files, min_steps, output_path)
        for user_id, user_files in sorted(files_by_user.items())
    ]

    # 使用线程池并行处理（每个用户一个线程）
    user_results: List[UserProcessResult] = []
    print("\nProcessing users...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_user_worker, args): args[0] for args in task_args}

        for future in as_completed(futures):
            result = future.result()
            user_results.append(result)

    # 汇总统计信息
    jsonl_files = []
    for result in user_results:
        stats['total_sessions'] += result.total_sessions
        stats['failed_sessions'] += result.failed_sessions
        stats['total_trajectories'] += result.total_trajectories
        stats['total_points'] += result.total_points
        stats['by_user'][result.user_id] = {
            'trajectories': result.total_trajectories,
            'points': result.total_points
        }
        jsonl_files.append(output_path / f"{result.user_id}.jsonl")

    print(f"\nJSONL files saved to: {output_path}/")
    print(f"Total {len(jsonl_files)} user files created")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("BOUN Dataset Preprocessing Complete (JSONL by User)")
    print("=" * 60)
    print(f"Threads used: {num_threads}")
    print(f"Total sessions processed: {stats['total_sessions']}")
    print(f"Failed sessions: {stats['failed_sessions']}")
    print(f"Total valid trajectories: {stats['total_trajectories']}")
    print(f"Total data points: {stats['total_points']}")
    print(f"Average trajectory length: {stats['total_points'] / max(1, stats['total_trajectories']):.1f}")
    print(f"\nOutput directory: {output_path}")
    print(f"User files: {len(jsonl_files)}")

    print("\n--- By User ---")
    for user_id, user_stats in sorted(stats['by_user'].items()):
        avg_len = user_stats['points'] / max(1, user_stats['trajectories'])
        print(f"  {user_id}: {user_stats['trajectories']} trajectories, avg length {avg_len:.1f}")

    # 保存统计信息
    stats_file = output_path / "preprocessing_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"BOUN Dataset Preprocessing Statistics (JSONL by User)\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Output format: JSONL (one file per user)\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"User files: {len(jsonl_files)}\n")
        f.write(f"Threads used: {num_threads}\n")
        f.write(f"Min steps threshold: {min_steps}\n")
        f.write(f"Total sessions: {stats['total_sessions']}\n")
        f.write(f"Failed sessions: {stats['failed_sessions']}\n")
        f.write(f"Total trajectories: {stats['total_trajectories']}\n")
        f.write(f"Total points: {stats['total_points']}\n")
        f.write(f"\nBy User:\n")
        for user_id, user_stats in sorted(stats['by_user'].items()):
            f.write(f"  {user_id}.jsonl: {user_stats['trajectories']} traj, {user_stats['points']} pts\n")

    print(f"\nStats saved to {stats_file}")

    return stats


def analyze_dataset(input_dir: str):
    """分析原始数据集的状态分布"""
    input_path = Path(input_dir)

    csv_files = list(input_path.glob("users/*/external_tests/*.csv"))[:10]  # 只分析前10个

    state_counts = {}
    segment_lengths = []

    for csv_file in tqdm(csv_files, desc="Analyzing"):
        try:
            df = pd.read_csv(csv_file)
            for state in df['state'].unique():
                state_counts[state] = state_counts.get(state, 0) + len(df[df['state'] == state])

            # 统计Released之间的距离
            released_idx = df[df['state'] == 'Released'].index.tolist()
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
        print(f"  Segments >= 100: {sum(1 for l in segment_lengths if l >= 100)}/{len(segment_lengths)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BOUN mouse dataset")
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
        default=100,
        help="Minimum number of Move events per trajectory"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        dest="no_clean",
        help="Do not clean output directory before processing (default: clean)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=NUM_THREADS,
        help=f"Number of threads for parallel processing (default: {NUM_THREADS})"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze the dataset without processing"
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_dataset(args.input_dir)
    else:
        preprocess_boun(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_steps=args.min_steps,
            clean_output=not args.no_clean,
            num_threads=args.threads,
        )


if __name__ == "__main__":
    main()
