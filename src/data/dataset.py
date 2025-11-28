"""
DMTG数据集加载和预处理模块
用于加载BOUN和SapiMouse鼠标轨迹数据集
支持Parquet和JSONL格式

使用 PyArrow 加速数据读取（CSV、Parquet、JSONL）
"""
import os
import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import pyarrow.json as pj
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm


class MouseTrajectoryDataset(Dataset):
    """鼠标轨迹数据集

    使用 Sequence Padding with Masking 方法（与论文一致）：
    - 保留原始轨迹点数，不进行重采样
    - 短于 max_length 的轨迹用 [0, 0] 填充
    - 返回 mask 标记有效位置

    支持两种加载模式：
    - eager (默认): 一次性加载所有数据到内存，访问快但内存占用大
    - lazy: 仅存储文件索引，按需加载数据，适合大数据集
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = "sapimouse",  # "sapimouse", "boun_parquet", "boun_jsonl", "open_images"
        max_length: int = 500,  # 论文中的 N，最大序列长度
        normalize: bool = True,
        screen_size: Tuple[int, int] = (1920, 1080),
        min_trajectory_length: int = 10,
        lazy: bool = False,  # 是否启用懒加载
    ):
        """
        Args:
            data_dir: 数据集根目录
            dataset_type: 数据集类型 ("sapimouse", "boun_parquet", "boun_jsonl", "open_images")
            max_length: 最大序列长度 N（论文默认500），超过此长度的轨迹会被截断
            normalize: 是否归一化坐标（BOUN Parquet数据已预归一化）
            screen_size: 屏幕尺寸 (width, height)
            min_trajectory_length: 最小轨迹长度
            lazy: 是否启用懒加载（仅支持 Parquet 格式）
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.normalize = normalize
        self.screen_size = screen_size
        self.min_trajectory_length = min_trajectory_length
        self.lazy = lazy

        # Eager模式: 存储 (coords, length, timestamps) 元组，timestamps 可为 None
        # Lazy模式: 存储 (file_path, row_index) 元组
        self.trajectories = []

        # Lazy模式缓存
        self._parquet_cache = {}  # file_path -> (x_col, y_col)
        self._cache_max_files = 3  # 最多缓存3个文件

        self._load_data()

    def _load_data(self):
        """加载所有轨迹数据"""
        if self.dataset_type == "sapimouse":
            self._load_sapimouse()
        elif self.dataset_type == "boun_parquet":
            self._load_boun_parquet()
        elif self.dataset_type == "boun_jsonl":
            self._load_jsonl()
        elif self.dataset_type == "open_images_parquet":
            self._load_open_images_parquet()
        elif self.dataset_type == "open_images":
            self._load_jsonl()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        print(f"Loaded {len(self.trajectories)} trajectories from {self.dataset_type}")

    def _load_sapimouse(self):
        """加载SapiMouse数据集（使用PyArrow）"""
        user_dirs = sorted(self.data_dir.glob("user*"))

        # 收集所有CSV文件
        all_csv_files = []
        for user_dir in user_dirs:
            csv_files = list(user_dir.glob("*.csv"))
            all_csv_files.extend(csv_files)

        if not all_csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return

        print(f"Found {len(all_csv_files)} CSV files, loading with PyArrow...")

        for csv_file in tqdm(all_csv_files, desc="Loading SapiMouse"):
            trajs = self._process_csv_file(csv_file)
            self.trajectories.extend(trajs)

    def _load_boun_parquet(self):
        """加载BOUN Parquet数据集（预处理后的格式）"""
        parquet_file = self.data_dir / "boun_trajectories.parquet"

        if not parquet_file.exists():
            print(f"Parquet file not found: {parquet_file}")
            return

        if self.lazy:
            # Lazy模式：只扫描索引
            print(f"Lazy loading BOUN Parquet from {parquet_file}...")
            table = pq.read_table(parquet_file, columns=['x'])  # 只读x列获取长度
            x_col = table.column('x')

            for i in range(len(table)):
                x_list = x_col[i].as_py()
                if len(x_list) >= self.min_trajectory_length:
                    self.trajectories.append((str(parquet_file), i))

            print(f"Found {len(self.trajectories)} valid trajectories (lazy)")
        else:
            # Eager模式：加载全部数据到内存
            print(f"Loading BOUN Parquet from {parquet_file}...")

            # 读取Parquet文件
            table = pq.read_table(parquet_file)

            # 提取列数据
            x_col = table.column('x')
            y_col = table.column('y')

            num_trajectories = len(table)
            print(f"Found {num_trajectories} trajectories in Parquet file")

            for i in tqdm(range(num_trajectories), desc="Loading BOUN"):
                x_list = x_col[i].as_py()
                y_list = y_col[i].as_py()

                if len(x_list) < self.min_trajectory_length:
                    continue

                # 数据已归一化，直接构建坐标数组
                coords = np.column_stack([x_list, y_list]).astype(np.float32)

                # 截断过长的轨迹
                if len(coords) > self.max_length:
                    coords = coords[:self.max_length]

                self.trajectories.append((coords, len(coords)))

    def _load_open_images_parquet(self):
        """加载 Open Images Parquet 数据集（支持多个 Parquet 文件，包含 x, y, t 列）"""
        parquet_files = sorted(self.data_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"No Parquet files found in {self.data_dir}")
            return

        if self.lazy:
            # Lazy模式：只扫描索引，不加载实际数据
            print(f"Lazy loading Open Images Parquet ({len(parquet_files)} files)...")

            for parquet_file in tqdm(parquet_files, desc="Scanning Parquet files"):
                try:
                    # 只读x列获取长度信息
                    table = pq.read_table(parquet_file, columns=['x'])
                except Exception as e:
                    print(f"  Error reading {parquet_file.name}: {e}")
                    continue

                x_col = table.column('x')

                for i in range(len(table)):
                    x_list = x_col[i].as_py()
                    if len(x_list) >= self.min_trajectory_length:
                        self.trajectories.append((str(parquet_file), i))

            print(f"Found {len(self.trajectories)} valid trajectories (lazy)")
        else:
            # Eager模式：加载全部数据到内存
            print(f"Loading Open Images Parquet ({len(parquet_files)} files)...")

            for parquet_file in parquet_files:
                try:
                    table = pq.read_table(parquet_file)
                except Exception as e:
                    print(f"  Error reading {parquet_file.name}: {e}")
                    continue

                x_col = table.column('x')
                y_col = table.column('y')
                t_col = table.column('t') if 't' in table.column_names else None

                # 每个文件单独的进度条
                for i in tqdm(range(len(table)), desc=f"  {parquet_file.name}", leave=False):
                    x_list = x_col[i].as_py()
                    y_list = y_col[i].as_py()

                    if len(x_list) < self.min_trajectory_length:
                        continue

                    # 数据已归一化，直接构建坐标数组
                    coords = np.column_stack([x_list, y_list]).astype(np.float32)
                    timestamps = np.array(t_col[i].as_py(), dtype=np.float32) if t_col else None

                    # 截断过长的轨迹
                    if len(coords) > self.max_length:
                        coords = coords[:self.max_length]
                        if timestamps is not None:
                            timestamps = timestamps[:self.max_length]

                    self.trajectories.append((coords, len(coords), timestamps))

                print(f"  {parquet_file.name}: {len(table)} records loaded")

    def _load_jsonl(self):
        """加载JSONL格式数据集"""
        jsonl_files = list(self.data_dir.glob("*.jsonl"))

        if not jsonl_files:
            print(f"No JSONL files found in {self.data_dir}")
            return

        print(f"Found {len(jsonl_files)} JSONL files...")

        for jsonl_file in jsonl_files:
            trajs = self._process_jsonl_file(jsonl_file)
            self.trajectories.extend(trajs)

    def _process_csv_file(self, csv_file: Path) -> List[Tuple[np.ndarray, int]]:
        """使用PyArrow处理单个CSV文件"""
        results = []
        try:
            # PyArrow读取CSV（比pandas快5-10倍）
            # 只读取需要的列
            table = pv.read_csv(
                csv_file,
                read_options=pv.ReadOptions(use_threads=True),
                convert_options=pv.ConvertOptions(
                    include_columns=['x', 'y', 'state']
                )
            )

            # 转换为numpy数组
            x = table.column('x').to_numpy()
            y = table.column('y').to_numpy()
            state = table.column('state').to_pylist()

            # 只保留Move事件
            move_mask = np.array([s == 'Move' for s in state])
            coords = np.column_stack([x[move_mask], y[move_mask]])

            # 分割成轨迹段
            trajectories = self._split_into_trajectories(coords)

            for traj in trajectories:
                if len(traj) >= self.min_trajectory_length:
                    if self.normalize:
                        traj = self._normalize_coords(traj)
                    if len(traj) > self.max_length:
                        traj = traj[:self.max_length]
                    results.append((traj.astype(np.float32), len(traj)))

        except Exception as e:
            # 静默处理错误
            pass

        return results

    def _process_jsonl_file(self, jsonl_file: Path) -> List[Tuple[np.ndarray, int]]:
        """使用PyArrow处理单个JSONL文件"""
        results = []
        file_name = jsonl_file.name

        print(f"  {file_name}: Loading with PyArrow...")
        table = pj.read_json(jsonl_file)

        # 获取traces列
        if 'traces' not in table.column_names:
            print(f"  {file_name}: No 'traces' column found")
            return results

        traces_col = table.column('traces')
        num_records = len(table)

        for i in range(num_records):
            traces = traces_col[i].as_py()
            if traces:
                trajs = self._process_traces(traces)
                results.extend(trajs)

        print(f"  {file_name}: Done ({num_records} records, {len(results)} trajectories)")
        return results

    def _process_traces(self, traces: list) -> List[Tuple[np.ndarray, int]]:
        """处理traces列表，提取轨迹坐标"""
        results = []

        for trace in traces:
            if not trace:
                continue

            # 快速提取坐标
            coords = np.array([[p['x'], p['y']] for p in trace if 'x' in p and 'y' in p])

            if len(coords) < self.min_trajectory_length:
                continue

            # 检查是否已归一化
            coord_max = coords.max()
            coord_min = coords.min()
            coord_median = np.median(coords)
            is_normalized = (coord_max <= 1.5 and coord_min >= -0.5 and
                           0.0 <= coord_median <= 1.0)

            if self.normalize and not is_normalized:
                coords = self._normalize_coords(coords)
            elif is_normalized:
                coords = np.clip(coords, 0.0, 1.0)

            if len(coords) > self.max_length:
                coords = coords[:self.max_length]

            results.append((coords.astype(np.float32), len(coords)))

        return results

    def _split_into_trajectories(
        self,
        coords: np.ndarray,
        distance_threshold: float = 100.0
    ) -> List[np.ndarray]:
        """将坐标序列分割成独立的轨迹段"""
        if len(coords) < 2:
            return []

        # 计算相邻点距离
        diffs = np.diff(coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        # 找到分割点
        split_indices = np.where(distances > distance_threshold)[0] + 1
        split_indices = np.concatenate([[0], split_indices, [len(coords)]])

        trajectories = []
        for i in range(len(split_indices) - 1):
            start, end = split_indices[i], split_indices[i + 1]
            if end - start >= self.min_trajectory_length:
                trajectories.append(coords[start:end])

        return trajectories

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """归一化坐标到[0, 1]范围"""
        normalized = coords.copy()
        normalized[:, 0] = normalized[:, 0] / self.screen_size[0]
        normalized[:, 1] = normalized[:, 1] / self.screen_size[1]
        return np.clip(normalized, 0, 1)

    def _pad_trajectory(self, coords: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """将轨迹padding到max_length"""
        padded = np.zeros((self.max_length, 2), dtype=np.float32)
        mask = np.zeros(self.max_length, dtype=np.float32)

        padded[:length] = coords
        mask[:length] = 1.0

        return padded, mask

    def __len__(self) -> int:
        return len(self.trajectories)

    def _get_parquet_data(self, file_path: str, row_idx: int) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
        """懒加载模式：从Parquet文件获取指定行的轨迹数据"""
        # 简单LRU缓存
        if file_path not in self._parquet_cache:
            # 缓存满时删除最老的
            if len(self._parquet_cache) >= self._cache_max_files:
                oldest_key = next(iter(self._parquet_cache))
                del self._parquet_cache[oldest_key]

            # 加载整个文件到缓存
            table = pq.read_table(file_path)
            t_col = table.column('t') if 't' in table.column_names else None
            self._parquet_cache[file_path] = (
                table.column('x'),
                table.column('y'),
                t_col,
            )

        x_col, y_col, t_col = self._parquet_cache[file_path]

        x_list = x_col[row_idx].as_py()
        y_list = y_col[row_idx].as_py()

        coords = np.column_stack([x_list, y_list]).astype(np.float32)
        timestamps = np.array(t_col[row_idx].as_py(), dtype=np.float32) if t_col else None

        # 截断过长的轨迹
        if len(coords) > self.max_length:
            coords = coords[:self.max_length]
            if timestamps is not None:
                timestamps = timestamps[:self.max_length]

        return coords, len(coords), timestamps

    def __getitem__(self, idx: int) -> dict:
        if self.lazy:
            # Lazy模式：按需加载
            file_path, row_idx = self.trajectories[idx]
            coords, length, timestamps = self._get_parquet_data(file_path, row_idx)
        else:
            # Eager模式：直接从内存获取
            item = self.trajectories[idx]
            if len(item) == 3:
                coords, length, timestamps = item
            else:
                coords, length = item
                timestamps = None

        padded_traj, mask = self._pad_trajectory(coords, length)

        start_point = coords[0]
        end_point = coords[-1]

        # 时间戳（始终包含以保持batch一致性）
        padded_t = np.zeros(self.max_length, dtype=np.float32)
        if timestamps is not None:
            padded_t[:length] = timestamps[:length]

        result = {
            'trajectory': torch.FloatTensor(padded_traj),
            'mask': torch.FloatTensor(mask),
            'start_point': torch.FloatTensor(start_point),
            'end_point': torch.FloatTensor(end_point),
            'length': torch.LongTensor([length]),
            'timestamps': torch.FloatTensor(padded_t),
        }

        return result


class CombinedMouseDataset(Dataset):
    """组合多个数据集"""

    def __init__(
        self,
        sapimouse_dir: Optional[str] = None,
        boun_dir: Optional[str] = None,  # 自动检测Parquet或JSONL
        open_images_dir: Optional[str] = None,
        max_length: int = 500,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        lazy: bool = False,  # 是否启用懒加载（仅对Parquet格式有效）
    ):
        self.max_length = max_length
        self.lazy = lazy
        self.datasets = []

        if sapimouse_dir:
            # SapiMouse不支持lazy（CSV格式）
            self.datasets.append(
                MouseTrajectoryDataset(
                    sapimouse_dir,
                    dataset_type="sapimouse",
                    max_length=max_length,
                    normalize=normalize,
                    lazy=False,  # CSV格式不支持lazy
                )
            )

        if boun_dir:
            boun_path = Path(boun_dir)
            # 优先使用Parquet格式
            if (boun_path / "boun_trajectories.parquet").exists():
                self.datasets.append(
                    MouseTrajectoryDataset(
                        boun_dir,
                        dataset_type="boun_parquet",
                        max_length=max_length,
                        normalize=normalize,
                        lazy=lazy,
                    )
                )
            elif list(boun_path.glob("*.jsonl")):
                # 回退到JSONL格式（不支持lazy）
                self.datasets.append(
                    MouseTrajectoryDataset(
                        boun_dir,
                        dataset_type="boun_jsonl",
                        max_length=max_length,
                        normalize=normalize,
                        lazy=False,
                    )
                )
            else:
                print(f"Warning: No BOUN data found in {boun_dir}")

        if open_images_dir:
            open_images_path = Path(open_images_dir)
            # 优先使用 Parquet 格式（支持多个文件）
            parquet_files = list(open_images_path.glob("*.parquet"))
            if parquet_files:
                self.datasets.append(
                    MouseTrajectoryDataset(
                        open_images_dir,
                        dataset_type="open_images_parquet",
                        max_length=max_length,
                        normalize=normalize,
                        lazy=lazy,
                    )
                )
            elif list(open_images_path.glob("*.jsonl")):
                # 回退到 JSONL 格式（不支持lazy）
                self.datasets.append(
                    MouseTrajectoryDataset(
                        open_images_dir,
                        dataset_type="open_images",
                        max_length=max_length,
                        normalize=normalize,
                        lazy=False,
                    )
                )
            else:
                print(f"Warning: No Open Images data found in {open_images_dir}")

        # 构建索引映射 (dataset_idx, local_idx) 或直接合并轨迹
        self._dataset_offsets = []  # 每个dataset的起始偏移
        self._total_size = 0

        for ds in self.datasets:
            self._dataset_offsets.append(self._total_size)
            self._total_size += len(ds)

        # 如果不是lazy模式，合并所有轨迹到内存（与之前行为一致）
        if not lazy:
            self.all_trajectories = []
            for ds in self.datasets:
                self.all_trajectories.extend(ds.trajectories)
            self._total_size = len(self.all_trajectories)
        else:
            self.all_trajectories = None  # lazy模式不合并

        # 随机采样（仅非lazy模式）
        if max_samples and not lazy and len(self.all_trajectories) > max_samples:
            indices = np.random.choice(
                len(self.all_trajectories),
                max_samples,
                replace=False
            )
            self.all_trajectories = [self.all_trajectories[i] for i in indices]
            self._total_size = len(self.all_trajectories)

        mode_str = "lazy" if lazy else "eager"
        print(f"Combined dataset: {self._total_size} trajectories ({mode_str} mode)")

    def _pad_trajectory(self, coords: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """将轨迹padding到max_length"""
        padded = np.zeros((self.max_length, 2), dtype=np.float32)
        mask = np.zeros(self.max_length, dtype=np.float32)

        padded[:length] = coords
        mask[:length] = 1.0

        return padded, mask

    def _find_dataset_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """将全局索引转换为(dataset_idx, local_idx)"""
        for i in range(len(self._dataset_offsets) - 1, -1, -1):
            if global_idx >= self._dataset_offsets[i]:
                return i, global_idx - self._dataset_offsets[i]
        return 0, global_idx

    def __len__(self) -> int:
        return self._total_size

    def __getitem__(self, idx: int) -> dict:
        if self.lazy:
            # Lazy模式：通过子dataset获取数据
            ds_idx, local_idx = self._find_dataset_and_local_idx(idx)
            return self.datasets[ds_idx][local_idx]
        else:
            # Eager模式：直接从合并的轨迹列表获取
            item = self.all_trajectories[idx]
            if len(item) == 3:
                coords, length, timestamps = item
            else:
                coords, length = item
                timestamps = None

            padded_traj, mask = self._pad_trajectory(coords, length)

            start_point = coords[0]
            end_point = coords[-1]

            # 时间戳（始终包含以保持batch一致性）
            padded_t = np.zeros(self.max_length, dtype=np.float32)
            if timestamps is not None:
                padded_t[:length] = timestamps[:length]

            result = {
                'trajectory': torch.FloatTensor(padded_traj),
                'mask': torch.FloatTensor(mask),
                'start_point': torch.FloatTensor(start_point),
                'end_point': torch.FloatTensor(end_point),
                'length': torch.LongTensor([length]),
                'timestamps': torch.FloatTensor(padded_t),
            }

            return result


def create_dataloader(
    sapimouse_dir: str = None,
    boun_dir: str = None,
    open_images_dir: str = None,
    batch_size: int = 64,
    max_length: int = 500,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int = None,
    val_split: float = 0.1,
    return_val: bool = True,
    lazy: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器

    Args:
        sapimouse_dir: SapiMouse数据集目录
        boun_dir: BOUN预处理后数据集目录（自动检测Parquet或JSONL格式）
        open_images_dir: Open Images数据集目录
        batch_size: 批次大小
        max_length: 最大序列长度N
        num_workers: DataLoader工作进程数
        shuffle: 是否打乱
        max_samples: 最大样本数
        val_split: 验证集比例
        return_val: 是否返回验证集
        lazy: 是否启用懒加载（适合大数据集，节省内存）

    Returns:
        (train_loader, val_loader)
    """
    dataset = CombinedMouseDataset(
        sapimouse_dir=sapimouse_dir,
        boun_dir=boun_dir,
        open_images_dir=open_images_dir,
        max_length=max_length,
        max_samples=max_samples,
        lazy=lazy,
    )

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    if return_val and val_size > 0:
        from torch.utils.data import random_split

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
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
            shuffle=False,
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
    base_dir = Path(__file__).parent.parent.parent
    sapimouse_dir = base_dir / "datasets" / "sapimouse"
    boun_dir = base_dir / "datasets" / "boun-processed"

    print("Testing SapiMouse dataset...")
    if sapimouse_dir.exists():
        ds = MouseTrajectoryDataset(str(sapimouse_dir), dataset_type="sapimouse")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Trajectory shape: {sample['trajectory'].shape}")
            print(f"  Start point: {sample['start_point']}")
            print(f"  End point: {sample['end_point']}")

    print("\nTesting BOUN dataset (auto-detect format)...")
    if boun_dir.exists():
        parquet_file = boun_dir / "boun_trajectories.parquet"
        if parquet_file.exists():
            print("  Found Parquet format, loading...")
            ds = MouseTrajectoryDataset(str(boun_dir), dataset_type="boun_parquet")
        else:
            print("  Falling back to JSONL format...")
            ds = MouseTrajectoryDataset(str(boun_dir), dataset_type="boun_jsonl")

        if len(ds) > 0:
            sample = ds[0]
            print(f"  Trajectory shape: {sample['trajectory'].shape}")
            print(f"  Start point: {sample['start_point']}")
            print(f"  End point: {sample['end_point']}")
