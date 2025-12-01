"""
DMTG数据集加载和预处理模块
用于加载BOUN和Open Images鼠标轨迹数据集
支持Parquet格式，输出 (x, y, dt) 三维轨迹

dt 归一化: log(dt + 1) / DT_LOG_SCALE
使用 PyArrow 加速数据读取
"""
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

# dt 归一化参数
# log(1001) ≈ 6.91，假设大部分 dt < 1000ms
# 归一化后 dt ∈ [0, 1] 左右
DT_LOG_SCALE = np.log(1001.0)


def normalize_dt(dt: np.ndarray) -> np.ndarray:
    """dt 归一化: log(dt + 1) / scale"""
    return np.log(dt + 1.0) / DT_LOG_SCALE


def denormalize_dt(dt_norm: np.ndarray) -> np.ndarray:
    """dt 反归一化: exp(dt_norm * scale) - 1"""
    return np.exp(dt_norm * DT_LOG_SCALE) - 1.0


class MouseTrajectoryDataset(Dataset):
    """鼠标轨迹数据集

    使用 Sequence Padding with Masking 方法（与论文一致）：
    - 保留原始轨迹点数，不进行重采样
    - 短于 max_length 的轨迹用 [0, 0, 0] 填充
    - 返回 mask 标记有效位置

    支持两种加载模式：
    - eager (默认): 一次性加载所有数据到内存，访问快但内存占用大
    - lazy: 仅存储文件索引，按需加载数据，适合大数据集
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = "boun_parquet",  # "boun_parquet", "open_images_parquet"
        max_length: int = 500,  # 论文中的 N，最大序列长度
        min_trajectory_length: int = 10,
        min_straight_dist: float = 0.0,  # 最小起终点直线距离（默认关闭，数据已预处理）
        lazy: bool = False,  # 是否启用懒加载
    ):
        """
        Args:
            data_dir: 数据集根目录
            dataset_type: 数据集类型 ("boun_parquet", "open_images_parquet")
            max_length: 最大序列长度 N（论文默认500），超过此长度的轨迹会被截断
            min_trajectory_length: 最小轨迹长度
            min_straight_dist: 最小起终点直线距离，过滤掉起终点太近的轨迹（避免 path_ratio 爆炸）
            lazy: 是否启用懒加载（仅支持 Parquet 格式）
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.min_trajectory_length = min_trajectory_length
        self.min_straight_dist = min_straight_dist
        self.lazy = lazy

        # Eager模式: 存储 (coords, length) 元组，coords 为 (N, 3) 包含 x, y, dt
        # Lazy模式: 存储 (file_path, row_index) 元组
        self.trajectories = []

        # Lazy模式缓存
        self._parquet_cache = {}  # file_path -> (x_col, y_col, dt_col)
        self._cache_max_files = 3  # 最多缓存3个文件

        self._filtered_count = 0  # 记录被过滤的轨迹数量
        self._load_data()

    def _compute_straight_dist(self, coords: np.ndarray) -> float:
        """计算起终点直线距离（仅使用 x, y）"""
        if len(coords) < 2:
            return 0.0
        # coords 为 (N, 3)，只取前两列 x, y
        return np.linalg.norm(coords[-1, :2] - coords[0, :2])

    def _load_data(self):
        """加载所有轨迹数据"""
        if self.dataset_type == "boun_parquet":
            self._load_boun_parquet()
        elif self.dataset_type == "open_images_parquet":
            self._load_open_images_parquet()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        print(f"Loaded {len(self.trajectories)} trajectories from {self.dataset_type}")

    def _load_boun_parquet(self):
        """加载BOUN Parquet数据集（预处理后的格式）"""
        parquet_file = self.data_dir / "boun_trajectories.parquet"

        if not parquet_file.exists():
            print(f"Parquet file not found: {parquet_file}")
            return

        if self.lazy:
            # Lazy模式：直接记录索引，不做过滤（数据已预处理）
            print(f"Lazy loading BOUN Parquet from {parquet_file}...")
            table = pq.read_table(parquet_file, columns=['x'])  # 只读x获取行数
            num_rows = len(table)
            for i in range(num_rows):
                self.trajectories.append((str(parquet_file), i))
            print(f"Found {num_rows} trajectories (lazy)")
        else:
            # Eager模式：批量加载全部数据到内存
            print(f"Loading BOUN Parquet from {parquet_file}...")

            # 读取Parquet文件
            table = pq.read_table(parquet_file)
            num_trajectories = len(table)
            print(f"Found {num_trajectories} trajectories in Parquet file")

            # 批量转换为numpy数组（比逐行转换快很多）
            x_arrays = table.column('x').to_pandas()
            y_arrays = table.column('y').to_pandas()
            dt_arrays = table.column('dt').to_pandas() if 'dt' in table.column_names else None

            for i in tqdm(range(num_trajectories), desc="Loading BOUN"):
                x_list = np.array(x_arrays[i], dtype=np.float32)
                y_list = np.array(y_arrays[i], dtype=np.float32)

                # 构建 (N, 3) 坐标数组: [x, y, dt_norm]
                if dt_arrays is not None:
                    dt_arr = np.array(dt_arrays[i], dtype=np.float32)
                    dt_norm = normalize_dt(dt_arr)
                    coords = np.column_stack([x_list, y_list, dt_norm])
                else:
                    coords = np.column_stack([x_list, y_list, np.zeros(len(x_list), dtype=np.float32)])

                # 截断过长的轨迹
                if len(coords) > self.max_length:
                    coords = coords[:self.max_length]

                self.trajectories.append((coords, len(coords)))

    def _load_open_images_parquet(self):
        """加载 Open Images Parquet 数据集（支持多个 Parquet 文件，包含 x, y, dt 列）"""
        parquet_files = sorted(self.data_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"No Parquet files found in {self.data_dir}")
            return

        if self.lazy:
            # Lazy模式：直接记录索引，不做过滤（数据已预处理）
            print(f"Lazy loading Open Images Parquet ({len(parquet_files)} files)...")

            for parquet_file in tqdm(parquet_files, desc="Scanning Parquet files"):
                try:
                    table = pq.read_table(parquet_file, columns=['x'])  # 只读x获取行数
                except Exception as e:
                    print(f"  Error reading {parquet_file.name}: {e}")
                    continue

                num_rows = len(table)
                for i in range(num_rows):
                    self.trajectories.append((str(parquet_file), i))

            print(f"Found {len(self.trajectories)} trajectories (lazy)")
        else:
            # Eager模式：批量加载全部数据到内存
            print(f"Loading Open Images Parquet ({len(parquet_files)} files)...")

            for parquet_file in parquet_files:
                try:
                    table = pq.read_table(parquet_file)
                except Exception as e:
                    print(f"  Error reading {parquet_file.name}: {e}")
                    continue

                num_rows = len(table)

                # 批量转换为numpy数组（比逐行转换快很多）
                x_arrays = table.column('x').to_pandas()
                y_arrays = table.column('y').to_pandas()
                dt_arrays = table.column('dt').to_pandas() if 'dt' in table.column_names else None

                for i in tqdm(range(num_rows), desc=f"  {parquet_file.name}", leave=False):
                    x_list = np.array(x_arrays[i], dtype=np.float32)
                    y_list = np.array(y_arrays[i], dtype=np.float32)

                    # 构建 (N, 3) 坐标数组: [x, y, dt_norm]
                    if dt_arrays is not None:
                        dt_arr = np.array(dt_arrays[i], dtype=np.float32)
                        dt_norm = normalize_dt(dt_arr)
                        coords = np.column_stack([x_list, y_list, dt_norm])
                    else:
                        coords = np.column_stack([x_list, y_list, np.zeros(len(x_list), dtype=np.float32)])

                    # 截断过长的轨迹
                    if len(coords) > self.max_length:
                        coords = coords[:self.max_length]

                    self.trajectories.append((coords, len(coords)))

                print(f"  {parquet_file.name}: {num_rows} records loaded")

    def _pad_trajectory(self, coords: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """将轨迹padding到max_length，coords 为 (N, 3)"""
        padded = np.zeros((self.max_length, 3), dtype=np.float32)
        mask = np.zeros(self.max_length, dtype=np.float32)

        padded[:length] = coords
        mask[:length] = 1.0

        return padded, mask

    def __len__(self) -> int:
        return len(self.trajectories)

    def _get_parquet_data(self, file_path: str, row_idx: int) -> Tuple[np.ndarray, int]:
        """懒加载模式：从Parquet文件获取指定行的轨迹数据"""
        # 简单LRU缓存
        if file_path not in self._parquet_cache:
            # 缓存满时删除最老的
            if len(self._parquet_cache) >= self._cache_max_files:
                oldest_key = next(iter(self._parquet_cache))
                del self._parquet_cache[oldest_key]

            # 加载整个文件到缓存
            table = pq.read_table(file_path)
            dt_col = table.column('dt') if 'dt' in table.column_names else None
            self._parquet_cache[file_path] = (
                table.column('x'),
                table.column('y'),
                dt_col,
            )

        x_col, y_col, dt_col = self._parquet_cache[file_path]

        x_list = x_col[row_idx].as_py()
        y_list = y_col[row_idx].as_py()

        # 构建 (N, 3) 坐标数组: [x, y, dt_norm]
        if dt_col is not None:
            dt_arr = np.array(dt_col[row_idx].as_py(), dtype=np.float32)
            dt_norm = normalize_dt(dt_arr)
            coords = np.column_stack([x_list, y_list, dt_norm]).astype(np.float32)
        else:
            coords = np.column_stack([x_list, y_list, np.zeros(len(x_list))]).astype(np.float32)

        # 截断过长的轨迹
        if len(coords) > self.max_length:
            coords = coords[:self.max_length]

        return coords, len(coords)

    def __getitem__(self, idx: int) -> dict:
        if self.lazy:
            # Lazy模式：按需加载
            file_path, row_idx = self.trajectories[idx]
            coords, length = self._get_parquet_data(file_path, row_idx)
        else:
            # Eager模式：直接从内存获取
            coords, length = self.trajectories[idx]

        padded_traj, mask = self._pad_trajectory(coords, length)

        # 起点终点仅使用 x, y
        start_point = coords[0, :2]
        end_point = coords[-1, :2]

        result = {
            'trajectory': torch.FloatTensor(padded_traj),  # (max_length, 3)
            'mask': torch.FloatTensor(mask),
            'start_point': torch.FloatTensor(start_point),  # (2,)
            'end_point': torch.FloatTensor(end_point),  # (2,)
            'length': torch.LongTensor([length]),
        }

        return result


class CombinedMouseDataset(Dataset):
    """组合多个数据集"""

    def __init__(
        self,
        boun_dir: Optional[str] = None,
        open_images_dir: Optional[str] = None,
        max_length: int = 500,
        max_samples: Optional[int] = None,
        min_straight_dist: float = 0.0,
        lazy: bool = False,
    ):
        self.max_length = max_length
        self.lazy = lazy
        self.datasets = []
        self.dataset_names = []

        if boun_dir:
            boun_path = Path(boun_dir)
            if (boun_path / "boun_trajectories.parquet").exists():
                self.datasets.append(
                    MouseTrajectoryDataset(
                        boun_dir,
                        dataset_type="boun_parquet",
                        max_length=max_length,
                        min_straight_dist=min_straight_dist,
                        lazy=lazy,
                    )
                )
                self.dataset_names.append("boun")
            else:
                print(f"Warning: No BOUN data found in {boun_dir}")

        if open_images_dir:
            open_images_path = Path(open_images_dir)
            parquet_files = list(open_images_path.glob("*.parquet"))
            if parquet_files:
                self.datasets.append(
                    MouseTrajectoryDataset(
                        open_images_dir,
                        dataset_type="open_images_parquet",
                        max_length=max_length,
                        min_straight_dist=min_straight_dist,
                        lazy=lazy,
                    )
                )
                self.dataset_names.append("open_images")
            else:
                print(f"Warning: No Open Images data found in {open_images_dir}")

        # 构建索引映射
        self._dataset_offsets = []
        self._total_size = 0

        for ds in self.datasets:
            self._dataset_offsets.append(self._total_size)
            self._total_size += len(ds)

        # 如果不是lazy模式，合并所有轨迹到内存
        if not lazy:
            self.all_trajectories = []
            for ds in self.datasets:
                self.all_trajectories.extend(ds.trajectories)
            self._total_size = len(self.all_trajectories)
        else:
            self.all_trajectories = None

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
        """将轨迹padding到max_length，coords 为 (N, 3)"""
        padded = np.zeros((self.max_length, 3), dtype=np.float32)
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
            result = self.datasets[ds_idx][local_idx]
            result['source'] = self.dataset_names[ds_idx]
            return result
        else:
            # Eager模式：直接从合并的轨迹列表获取
            coords, length = self.all_trajectories[idx]

            padded_traj, mask = self._pad_trajectory(coords, length)

            start_point = coords[0, :2]
            end_point = coords[-1, :2]

            result = {
                'trajectory': torch.FloatTensor(padded_traj),
                'mask': torch.FloatTensor(mask),
                'start_point': torch.FloatTensor(start_point),
                'end_point': torch.FloatTensor(end_point),
                'length': torch.LongTensor([length]),
            }

            return result


def create_dataloader(
    boun_dir: str = None,
    open_images_dir: str = None,
    batch_size: int = 64,
    max_length: int = 500,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int = None,
    val_split: float = 0.1,
    return_val: bool = True,
    min_straight_dist: float = 0.0,
    lazy: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器

    Args:
        boun_dir: BOUN预处理后数据集目录
        open_images_dir: Open Images数据集目录
        batch_size: 批次大小
        max_length: 最大序列长度N
        num_workers: DataLoader工作进程数
        shuffle: 是否打乱
        max_samples: 最大样本数
        val_split: 验证集比例
        return_val: 是否返回验证集
        min_straight_dist: 最小起终点直线距离
        lazy: 是否启用懒加载

    Returns:
        (train_loader, val_loader)
    """
    dataset = CombinedMouseDataset(
        boun_dir=boun_dir,
        open_images_dir=open_images_dir,
        max_length=max_length,
        max_samples=max_samples,
        min_straight_dist=min_straight_dist,
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
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
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
            persistent_workers=num_workers > 0,
        )
        return train_loader, None


if __name__ == "__main__":
    # 测试数据加载
    base_dir = Path(__file__).parent.parent.parent
    boun_dir = base_dir / "datasets" / "boun-processed"

    print(f"DT_LOG_SCALE = {DT_LOG_SCALE:.4f}")
    print(f"dt=0ms -> {normalize_dt(np.array([0.0]))[0]:.4f}")
    print(f"dt=10ms -> {normalize_dt(np.array([10.0]))[0]:.4f}")
    print(f"dt=100ms -> {normalize_dt(np.array([100.0]))[0]:.4f}")
    print(f"dt=1000ms -> {normalize_dt(np.array([1000.0]))[0]:.4f}")
    print()

    print("Testing BOUN dataset...")
    if boun_dir.exists():
        parquet_file = boun_dir / "boun_trajectories.parquet"
        if parquet_file.exists():
            print("  Found Parquet format, loading...")
            ds = MouseTrajectoryDataset(str(boun_dir), dataset_type="boun_parquet")

            if len(ds) > 0:
                sample = ds[0]
                print(f"  Trajectory shape: {sample['trajectory'].shape}")  # (500, 3)
                print(f"  Start point: {sample['start_point']}")  # (2,)
                print(f"  End point: {sample['end_point']}")  # (2,)
                print(f"  Length: {sample['length'].item()}")
                # 显示 dt_norm 统计
                traj = sample['trajectory'].numpy()
                length = sample['length'].item()
                dt_norm = traj[:length, 2]
                print(f"  dt_norm range: [{dt_norm.min():.4f}, {dt_norm.max():.4f}]")
                print(f"  dt_norm mean: {dt_norm.mean():.4f}")
