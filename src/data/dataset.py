"""
DMTG数据集加载和预处理模块
用于加载BOUN和SapiMouse鼠标轨迹数据集
支持JSONL格式（Open Images V7风格）
支持多线程加载加速
"""
import os
import numpy as np

# 使用 orjson 加速 JSON 解析（比标准 json 快 3-10 倍）
try:
    import orjson
    def json_loads(s):
        return orjson.loads(s)
    JSONDecodeError = orjson.JSONDecodeError
except ImportError:
    import json
    def json_loads(s):
        return json.loads(s)
    JSONDecodeError = json.JSONDecodeError
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class MouseTrajectoryDataset(Dataset):
    """鼠标轨迹数据集

    使用 Sequence Padding with Masking 方法（与论文一致）：
    - 保留原始轨迹点数，不进行重采样
    - 短于 max_length 的轨迹用 [0, 0] 填充
    - 返回 mask 标记有效位置
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = "sapimouse",  # "sapimouse" or "boun"
        max_length: int = 500,  # 论文中的 N，最大序列长度
        normalize: bool = True,
        screen_size: Tuple[int, int] = (1920, 1080),
        min_trajectory_length: int = 10,
        num_threads: int = 16,
    ):
        """
        Args:
            data_dir: 数据集根目录
            dataset_type: 数据集类型 ("sapimouse" 或 "boun")
            max_length: 最大序列长度 N（论文默认500），超过此长度的轨迹会被截断
            normalize: 是否归一化坐标
            screen_size: 屏幕尺寸 (width, height)
            min_trajectory_length: 最小轨迹长度
            num_threads: 数据加载线程数
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.normalize = normalize
        self.screen_size = screen_size
        self.min_trajectory_length = min_trajectory_length
        self.num_threads = num_threads

        self.trajectories = []  # 存储 (coords, length) 元组
        self._load_data()

    def _load_data(self):
        """加载所有轨迹数据"""
        if self.dataset_type == "sapimouse":
            self._load_sapimouse()
        elif self.dataset_type == "boun":
            self._load_boun()
        elif self.dataset_type == "boun_processed":
            self._load_boun_processed()
        elif self.dataset_type == "boun_jsonl":
            self._load_boun_jsonl()
        elif self.dataset_type == "open_images":
            self._load_open_images()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        print(f"Loaded {len(self.trajectories)} trajectories from {self.dataset_type}")

    def _load_sapimouse(self):
        """加载SapiMouse数据集（多文件并行）"""
        user_dirs = sorted(self.data_dir.glob("user*"))

        # 收集所有CSV文件
        all_csv_files = []
        for user_dir in user_dirs:
            csv_files = list(user_dir.glob("*.csv"))
            all_csv_files.extend(csv_files)

        print(f"Found {len(all_csv_files)} CSV files, loading with {self.num_threads} threads...")

        # 多线程并行处理多个文件
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._process_csv_file_worker, f, "sapimouse") for f in all_csv_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading SapiMouse"):
                result = future.result()
                if result:
                    self.trajectories.extend(result)

    def _load_boun(self):
        """加载BOUN原始数据集（多文件并行）"""
        user_dirs = sorted(self.data_dir.glob("users/user*"))

        # 收集所有CSV文件
        all_csv_files = []
        for user_dir in user_dirs:
            training_dir = user_dir / "training"
            if training_dir.exists():
                csv_files = list(training_dir.glob("*.csv"))
                all_csv_files.extend(csv_files)

        print(f"Found {len(all_csv_files)} CSV files, loading with {self.num_threads} threads...")

        # 多线程并行处理多个文件
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._process_csv_file_worker, f, "boun") for f in all_csv_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading BOUN"):
                result = future.result()
                if result:
                    self.trajectories.extend(result)

    def _load_boun_processed(self):
        """加载预处理后的BOUN数据集（多文件并行）"""
        # 收集所有CSV文件
        all_csv_files = list(self.data_dir.glob("**/*.csv"))

        print(f"Found {len(all_csv_files)} trajectory files, loading with {self.num_threads} threads...")

        # 多线程并行处理多个文件
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._process_single_trajectory_worker, f) for f in all_csv_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading BOUN-processed"):
                result = future.result()
                if result is not None:
                    self.trajectories.append(result)

    def _load_boun_jsonl(self):
        """加载JSONL格式的BOUN数据集（多文件并行）"""
        # 查找JSONL文件
        jsonl_files = list(self.data_dir.glob("*.jsonl"))

        if not jsonl_files:
            print(f"No JSONL files found in {self.data_dir}")
            return

        print(f"Found {len(jsonl_files)} JSONL files, loading with {self.num_threads} threads...")

        # 多线程并行处理多个文件
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._process_jsonl_file_worker, f) for f in jsonl_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading JSONL files"):
                result = future.result()
                if result:
                    self.trajectories.extend(result)

    def _load_open_images(self):
        """加载Open Images Localized Narratives数据集（多文件并行）"""
        # 查找JSONL文件
        jsonl_files = list(self.data_dir.glob("*.jsonl"))

        if not jsonl_files:
            print(f"No JSONL files found in {self.data_dir}")
            return

        print(f"Found {len(jsonl_files)} Open Images JSONL files, loading with {self.num_threads} threads...")

        # 多线程并行处理多个文件（复用JSONL处理逻辑）
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._process_jsonl_file_worker, f) for f in jsonl_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Open Images"):
                result = future.result()
                if result:
                    self.trajectories.extend(result)

    def _process_jsonl_file(self, jsonl_file: Path):
        """处理单个JSONL文件"""
        try:
            # 先统计行数用于进度条
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_lines, desc=f"Loading {jsonl_file.name}"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json_loads(line)
                        self._process_jsonl_record(record)
                    except JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue

        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")

    def _process_jsonl_record(self, record: Dict[str, Any]):
        """处理单条JSONL记录"""
        traces = record.get('traces', [])

        if not traces:
            return

        # traces是一个列表的列表，每个子列表是一条轨迹
        for trace in traces:
            if not trace:
                continue

            # 提取x, y坐标
            coords = []
            for point in trace:
                if 'x' in point and 'y' in point:
                    coords.append([point['x'], point['y']])

            if len(coords) < self.min_trajectory_length:
                continue

            coords = np.array(coords)

            # 检查数据是否已经归一化
            coord_max = coords.max()
            coord_min = coords.min()
            coord_median = np.median(coords)
            is_already_normalized = (coord_max <= 1.5 and coord_min >= -0.5 and
                                    0.0 <= coord_median <= 1.0)

            if self.normalize and not is_already_normalized:
                coords = self._normalize_coords(coords)
            elif is_already_normalized:
                coords = np.clip(coords, 0.0, 1.0)

            # 截断超长轨迹
            if len(coords) > self.max_length:
                coords = coords[:self.max_length]

            # 存储轨迹和原始长度（padding 在 __getitem__ 中进行）
            self.trajectories.append((coords, len(coords)))

    def _process_single_trajectory(self, csv_file: Path):
        """处理预处理后的单条轨迹CSV文件"""
        try:
            df = pd.read_csv(csv_file)

            if 'x' not in df.columns or 'y' not in df.columns:
                return

            coords = df[['x', 'y']].values

            if len(coords) < self.min_trajectory_length:
                return

            if self.normalize:
                coords = self._normalize_coords(coords)

            # 截断超长轨迹
            if len(coords) > self.max_length:
                coords = coords[:self.max_length]

            self.trajectories.append((coords, len(coords)))

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    def _process_csv_file(self, csv_file: Path, dataset_format: str):
        """处理单个CSV文件，提取轨迹"""
        try:
            df = pd.read_csv(csv_file)

            # 根据数据格式获取坐标列
            if dataset_format == "sapimouse":
                # SapiMouse格式: client timestamp,button,state,x,y
                if 'x' not in df.columns or 'y' not in df.columns:
                    return
                coords = df[['x', 'y']].values
                states = df['state'].values if 'state' in df.columns else None
            else:
                # BOUN格式: client_timestamp,x,y,button,state,window
                if 'x' not in df.columns or 'y' not in df.columns:
                    return
                coords = df[['x', 'y']].values
                states = df['state'].values if 'state' in df.columns else None

            # 只保留移动事件
            if states is not None:
                move_mask = (states == 'Move')
                coords = coords[move_mask]

            # 分割成轨迹段（通过检测大的时间间隔或距离跳跃）
            trajectories = self._split_into_trajectories(coords)

            for traj in trajectories:
                if len(traj) >= self.min_trajectory_length:
                    if self.normalize:
                        traj = self._normalize_coords(traj)
                    # 截断超长轨迹
                    if len(traj) > self.max_length:
                        traj = traj[:self.max_length]
                    self.trajectories.append((traj, len(traj)))

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    def _process_csv_file_worker(self, csv_file: Path, dataset_format: str) -> List[Tuple[np.ndarray, int]]:
        """多线程worker：处理单个CSV文件，返回轨迹列表"""
        results = []
        try:
            df = pd.read_csv(csv_file)

            # 根据数据格式获取坐标列
            if dataset_format == "sapimouse":
                if 'x' not in df.columns or 'y' not in df.columns:
                    return results
                coords = df[['x', 'y']].values
                states = df['state'].values if 'state' in df.columns else None
            else:
                if 'x' not in df.columns or 'y' not in df.columns:
                    return results
                coords = df[['x', 'y']].values
                states = df['state'].values if 'state' in df.columns else None

            # 只保留移动事件
            if states is not None:
                move_mask = (states == 'Move')
                coords = coords[move_mask]

            # 分割成轨迹段
            trajectories = self._split_into_trajectories(coords)

            for traj in trajectories:
                if len(traj) >= self.min_trajectory_length:
                    if self.normalize:
                        traj = self._normalize_coords(traj)
                    # 截断超长轨迹
                    if len(traj) > self.max_length:
                        traj = traj[:self.max_length]
                    results.append((traj, len(traj)))

        except Exception as e:
            pass  # 静默处理错误，避免多线程输出混乱

        return results

    def _process_single_trajectory_worker(self, csv_file: Path) -> Optional[Tuple[np.ndarray, int]]:
        """多线程worker：处理单条轨迹CSV文件"""
        try:
            df = pd.read_csv(csv_file)

            if 'x' not in df.columns or 'y' not in df.columns:
                return None

            coords = df[['x', 'y']].values

            if len(coords) < self.min_trajectory_length:
                return None

            if self.normalize:
                coords = self._normalize_coords(coords)

            # 截断超长轨迹
            if len(coords) > self.max_length:
                coords = coords[:self.max_length]

            return (coords, len(coords))

        except Exception:
            return None

    def _process_jsonl_file_worker(self, jsonl_file: Path) -> List[Tuple[np.ndarray, int]]:
        """多线程worker：处理单个JSONL文件，返回轨迹列表"""
        results = []
        try:
            # 获取文件大小用于进度估算
            file_size = jsonl_file.stat().st_size
            file_name = jsonl_file.name

            processed_bytes = 0
            line_count = 0
            last_progress = 0

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_bytes += len(line.encode('utf-8'))
                    line_count += 1

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json_loads(line)
                        trajs = self._process_jsonl_record_worker(record)
                        results.extend(trajs)
                    except JSONDecodeError:
                        continue

                    # 每10%打印一次进度
                    progress = int(processed_bytes * 100 / file_size)
                    if progress >= last_progress + 10:
                        last_progress = progress
                        print(f"  {file_name}: {progress}% ({line_count} records, {len(results)} trajectories)")

            print(f"  {file_name}: Done ({line_count} records, {len(results)} trajectories)")

        except Exception as e:
            print(f"  Error processing {jsonl_file.name}: {e}")

        return results

    def _process_jsonl_record_worker(self, record: Dict[str, Any]) -> List[Tuple[np.ndarray, int]]:
        """处理单条JSONL记录，返回轨迹列表"""
        results = []
        traces = record.get('traces', [])

        if not traces:
            return results

        for trace in traces:
            if not trace:
                continue

            coords = []
            for point in trace:
                if 'x' in point and 'y' in point:
                    coords.append([point['x'], point['y']])

            if len(coords) < self.min_trajectory_length:
                continue

            coords = np.array(coords)

            # 检查数据是否已经归一化
            coord_max = coords.max()
            coord_min = coords.min()
            coord_median = np.median(coords)
            is_already_normalized = (coord_max <= 1.5 and coord_min >= -0.5 and
                                    0.0 <= coord_median <= 1.0)

            if self.normalize and not is_already_normalized:
                coords = self._normalize_coords(coords)
            elif is_already_normalized:
                coords = np.clip(coords, 0.0, 1.0)

            # 截断超长轨迹
            if len(coords) > self.max_length:
                coords = coords[:self.max_length]

            results.append((coords, len(coords)))

        return results

    def _split_into_trajectories(
        self,
        coords: np.ndarray,
        distance_threshold: float = 100.0
    ) -> List[np.ndarray]:
        """将坐标序列分割成独立的轨迹段"""
        if len(coords) < 2:
            return []

        trajectories = []
        current_traj = [coords[0]]

        for i in range(1, len(coords)):
            # 计算与上一个点的距离
            dist = np.linalg.norm(coords[i] - coords[i-1])

            if dist > distance_threshold:
                # 距离跳跃，开始新轨迹
                if len(current_traj) >= self.min_trajectory_length:
                    trajectories.append(np.array(current_traj))
                current_traj = [coords[i]]
            else:
                current_traj.append(coords[i])

        # 添加最后一段轨迹
        if len(current_traj) >= self.min_trajectory_length:
            trajectories.append(np.array(current_traj))

        return trajectories

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """归一化坐标到[0, 1]范围"""
        normalized = coords.copy()
        normalized[:, 0] = normalized[:, 0] / self.screen_size[0]
        normalized[:, 1] = normalized[:, 1] / self.screen_size[1]
        # Clip到有效范围
        normalized = np.clip(normalized, 0, 1)
        return normalized

    def _pad_trajectory(self, coords: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """将轨迹 padding 到 max_length，返回 (padded_coords, mask)"""
        padded = np.zeros((self.max_length, 2), dtype=np.float32)
        mask = np.zeros(self.max_length, dtype=np.float32)

        # 复制有效数据
        padded[:length] = coords
        mask[:length] = 1.0

        return padded, mask

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict:
        coords, length = self.trajectories[idx]

        # Padding 到固定长度
        padded_traj, mask = self._pad_trajectory(coords, length)

        # 起点和终点作为条件
        start_point = coords[0]
        end_point = coords[-1]

        return {
            'trajectory': torch.FloatTensor(padded_traj),  # (max_length, 2)
            'mask': torch.FloatTensor(mask),  # (max_length,)
            'start_point': torch.FloatTensor(start_point),  # (2,)
            'end_point': torch.FloatTensor(end_point),  # (2,)
            'length': torch.LongTensor([length]),  # (1,) 原始轨迹长度
        }


class CombinedMouseDataset(Dataset):
    """组合多个数据集

    使用 Sequence Padding with Masking 方法（与论文一致）
    """

    def __init__(
        self,
        sapimouse_dir: Optional[str] = None,
        boun_dir: Optional[str] = None,
        boun_processed_dir: Optional[str] = None,
        boun_jsonl_dir: Optional[str] = None,
        open_images_dir: Optional[str] = None,
        max_length: int = 500,  # 论文中的 N
        normalize: bool = True,
        max_samples: Optional[int] = None,
        num_threads: int = 16,
    ):
        self.max_length = max_length
        self.datasets = []

        if sapimouse_dir:
            self.datasets.append(
                MouseTrajectoryDataset(
                    sapimouse_dir,
                    dataset_type="sapimouse",
                    max_length=max_length,
                    normalize=normalize,
                    num_threads=num_threads,
                )
            )

        if boun_dir:
            self.datasets.append(
                MouseTrajectoryDataset(
                    boun_dir,
                    dataset_type="boun",
                    max_length=max_length,
                    normalize=normalize,
                    num_threads=num_threads,
                )
            )

        if boun_processed_dir:
            self.datasets.append(
                MouseTrajectoryDataset(
                    boun_processed_dir,
                    dataset_type="boun_processed",
                    max_length=max_length,
                    normalize=normalize,
                    num_threads=num_threads,
                )
            )

        if boun_jsonl_dir:
            self.datasets.append(
                MouseTrajectoryDataset(
                    boun_jsonl_dir,
                    dataset_type="boun_jsonl",
                    max_length=max_length,
                    normalize=normalize,
                    num_threads=num_threads,
                )
            )

        if open_images_dir:
            self.datasets.append(
                MouseTrajectoryDataset(
                    open_images_dir,
                    dataset_type="open_images",
                    max_length=max_length,
                    normalize=normalize,
                    num_threads=num_threads,
                )
            )

        # 合并所有轨迹 (每个轨迹是 (coords, length) 元组)
        self.all_trajectories = []
        for ds in self.datasets:
            self.all_trajectories.extend(ds.trajectories)

        # 如果设置了最大样本数，随机采样
        if max_samples and len(self.all_trajectories) > max_samples:
            indices = np.random.choice(
                len(self.all_trajectories),
                max_samples,
                replace=False
            )
            self.all_trajectories = [self.all_trajectories[i] for i in indices]

        print(f"Combined dataset: {len(self.all_trajectories)} trajectories")

    def _pad_trajectory(self, coords: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """将轨迹 padding 到 max_length，返回 (padded_coords, mask)"""
        padded = np.zeros((self.max_length, 2), dtype=np.float32)
        mask = np.zeros(self.max_length, dtype=np.float32)

        # 复制有效数据
        padded[:length] = coords
        mask[:length] = 1.0

        return padded, mask

    def __len__(self) -> int:
        return len(self.all_trajectories)

    def __getitem__(self, idx: int) -> dict:
        coords, length = self.all_trajectories[idx]

        # Padding 到固定长度
        padded_traj, mask = self._pad_trajectory(coords, length)

        # 起点和终点作为条件
        start_point = coords[0]
        end_point = coords[-1]

        return {
            'trajectory': torch.FloatTensor(padded_traj),  # (max_length, 2)
            'mask': torch.FloatTensor(mask),  # (max_length,)
            'start_point': torch.FloatTensor(start_point),  # (2,)
            'end_point': torch.FloatTensor(end_point),  # (2,)
            'length': torch.LongTensor([length]),  # (1,) 原始轨迹长度
        }


def create_dataloader(
    sapimouse_dir: str = None,
    boun_dir: str = None,
    boun_processed_dir: str = None,
    boun_jsonl_dir: str = None,
    open_images_dir: str = None,
    batch_size: int = 64,
    max_length: int = 500,  # 论文中的 N，最大序列长度
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int = None,
    val_split: float = 0.1,
    return_val: bool = True,
    num_threads: int = 16,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器

    使用 Sequence Padding with Masking 方法（与论文一致）

    Args:
        sapimouse_dir: SapiMouse数据集目录
        boun_dir: BOUN原始数据集目录
        boun_processed_dir: BOUN预处理后数据集目录（CSV格式）
        boun_jsonl_dir: BOUN预处理后数据集目录（JSONL格式）
        open_images_dir: Open Images Localized Narratives数据集目录
        batch_size: 批次大小
        max_length: 最大序列长度 N（论文默认500）
        num_workers: 数据加载工作进程数
        shuffle: 是否打乱数据
        max_samples: 最大样本数
        val_split: 验证集比例 (默认10%)
        return_val: 是否返回验证集加载器
        num_threads: 数据加载线程数（默认16）

    Returns:
        (train_loader, val_loader) 或 train_loader
    """
    dataset = CombinedMouseDataset(
        sapimouse_dir=sapimouse_dir,
        boun_dir=boun_dir,
        boun_processed_dir=boun_processed_dir,
        boun_jsonl_dir=boun_jsonl_dir,
        open_images_dir=open_images_dir,
        max_length=max_length,
        max_samples=max_samples,
        num_threads=num_threads,
    )

    # 分割训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    if return_val and val_size > 0:
        from torch.utils.data import random_split

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定种子保证可复现
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集不需要打乱
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"Train set: {train_size} samples, Val set: {val_size} samples")
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, None


if __name__ == "__main__":
    # 测试数据加载
    import sys

    # 设置数据路径
    base_dir = Path(__file__).parent.parent.parent
    sapimouse_dir = base_dir / "datasets" / "sapimouse"
    boun_dir = base_dir / "datasets" / "boun-mouse-dynamics-dataset"
    boun_jsonl_dir = base_dir / "datasets" / "boun-processed"

    print("Testing SapiMouse dataset...")
    if sapimouse_dir.exists():
        ds = MouseTrajectoryDataset(str(sapimouse_dir), dataset_type="sapimouse")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Trajectory shape: {sample['trajectory'].shape}")
            print(f"  Start point: {sample['start_point']}")
            print(f"  End point: {sample['end_point']}")

    print("\nTesting BOUN dataset...")
    if boun_dir.exists():
        ds = MouseTrajectoryDataset(str(boun_dir), dataset_type="boun")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Trajectory shape: {sample['trajectory'].shape}")

    print("\nTesting BOUN JSONL dataset...")
    if boun_jsonl_dir.exists():
        ds = MouseTrajectoryDataset(str(boun_jsonl_dir), dataset_type="boun_jsonl")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Trajectory shape: {sample['trajectory'].shape}")
            print(f"  Start point: {sample['start_point']}")
            print(f"  End point: {sample['end_point']}")
