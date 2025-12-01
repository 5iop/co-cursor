"""
数据集综合分析脚本
分析轨迹长度、复杂度和时间间隔分布
支持 Parquet/CSV/JSONL 格式，自动检测

用法:
    python tools/analyze_datasets.py --boun datasets/boun_trajectories.parquet
    python tools/analyze_datasets.py --sapimouse datasets/sapimouse --boun datasets/boun
    python tools/analyze_datasets.py --all --plot
"""
import argparse
import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pyarrow.json as pj
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class AnalysisResult:
    """分析结果"""
    point_counts: List[int] = field(default_factory=list)
    complexities: List[float] = field(default_factory=list)
    time_intervals: List[float] = field(default_factory=list)
    total_durations: List[float] = field(default_factory=list)

    def extend(self, other: 'AnalysisResult'):
        self.point_counts.extend(other.point_counts)
        self.complexities.extend(other.complexities)
        self.time_intervals.extend(other.time_intervals)
        self.total_durations.extend(other.total_durations)

    @property
    def count(self) -> int:
        return len(self.point_counts)


class DatasetAnalyzer:
    """数据集分析器"""

    DISTANCE_THRESHOLD = 100.0  # SapiMouse 轨迹分割阈值

    def __init__(self, min_length: int = 10):
        self.min_length = min_length

    def analyze(self, path: str, dataset_type: str = "auto") -> AnalysisResult:
        """
        分析数据集

        Args:
            path: 文件或目录路径（支持 glob）
            dataset_type: 数据集类型 (sapimouse/boun/open-images/auto)
        """
        files = self._collect_files(path)
        if not files:
            return AnalysisResult()

        result = AnalysisResult()

        # 处理文件
        for f in files:
            file_result = self._analyze_file(f, dataset_type)
            result.extend(file_result)

        return result

    def _collect_files(self, path: str) -> List[Path]:
        """收集文件列表"""
        import glob as glob_module

        path_obj = Path(path)
        base_dir = Path(__file__).parent.parent

        # glob 模式
        if '*' in path or '?' in path:
            # 使用 glob 模块处理，支持绝对路径
            files = [Path(f) for f in glob_module.glob(path, recursive=True)]
            if not files:
                # 尝试相对于项目根目录
                files = list(base_dir.glob(path))
            if files:
                print(f"找到 {len(files)} 个文件 (glob)")
            return files

        # 绝对路径或相对路径
        if not path_obj.exists():
            path_obj = base_dir / path

        if not path_obj.exists():
            print(f"路径不存在: {path}")
            return []

        if path_obj.is_file():
            print(f"找到文件: {path_obj.name}")
            return [path_obj]

        # 目录: 递归查找
        files = []
        for ext in ['*.parquet', '*.csv', '*.jsonl']:
            files.extend(path_obj.glob(f"**/{ext}"))

        if files:
            print(f"找到 {len(files)} 个文件")
        return files

    def _analyze_file(self, file_path: Path, dataset_type: str) -> AnalysisResult:
        """根据扩展名分析文件"""
        ext = file_path.suffix.lower()

        if ext == '.parquet':
            return self._analyze_parquet(file_path, dataset_type)
        elif ext == '.csv':
            return self._analyze_csv(file_path, dataset_type)
        elif ext == '.jsonl':
            return self._analyze_jsonl(file_path)

        return AnalysisResult()

    def _analyze_parquet(self, file_path: Path, dataset_type: str) -> AnalysisResult:
        """分析 Parquet 文件"""
        result = AnalysisResult()

        try:
            schema = pq.read_schema(file_path)
            cols = schema.names
            has_dt = 'dt' in cols
            read_cols = ['x', 'y'] + (['dt'] if has_dt else [])
            table = pq.read_table(file_path, columns=read_cols)

            x_col, y_col = table.column('x'), table.column('y')
            dt_col = table.column('dt') if has_dt else None

            for i in tqdm(range(len(table)), desc=f"读取 {file_path.name}", leave=False):
                x_list, y_list = x_col[i].as_py(), y_col[i].as_py()
                if not x_list or len(x_list) < 2:
                    continue

                result.point_counts.append(len(x_list))
                coords = np.column_stack([x_list, y_list]).astype(np.float32)
                result.complexities.append(self._compute_complexity(coords))

                if dt_col:
                    dt_list = dt_col[i].as_py()
                    if dt_list and len(dt_list) >= 2:
                        dt = np.array(dt_list, dtype=np.float64)
                        dt = dt[dt >= 0]  # 过滤负值
                        result.time_intervals.extend(dt.tolist())
                        # 总时长 = sum(dt)
                        total_dur = np.sum(dt)
                        if total_dur > 0:
                            result.total_durations.append(total_dur)

        except Exception as e:
            print(f"读取错误 {file_path.name}: {e}")

        return result

    def _analyze_csv(self, file_path: Path, dataset_type: str) -> AnalysisResult:
        """分析 CSV 文件 (SapiMouse / BOUN raw)"""
        result = AnalysisResult()
        print(f"读取 {file_path.name}...", end="\r")

        try:
            table = pv.read_csv(file_path, read_options=pv.ReadOptions(use_threads=True))
            cols = table.column_names

            # 检测格式
            is_sapimouse = 'state' in cols and 'client timestamp' in cols
            is_boun_raw = 'client_timestamp' in cols and 'state' in cols

            if is_sapimouse:
                return self._analyze_sapimouse_csv(table)
            elif is_boun_raw:
                return self._analyze_boun_csv(table)
            else:
                # 通用 CSV: 假设有 x, y 列
                if 'x' in cols and 'y' in cols:
                    x = table.column('x').to_numpy()
                    y = table.column('y').to_numpy()
                    if len(x) >= 2:
                        coords = np.column_stack([x, y]).astype(np.float32)
                        result.point_counts.append(len(x))
                        result.complexities.append(self._compute_complexity(coords))

        except Exception:
            pass

        return result

    def _analyze_sapimouse_csv(self, table) -> AnalysisResult:
        """分析 SapiMouse CSV"""
        result = AnalysisResult()

        x = table.column('x').to_numpy()
        y = table.column('y').to_numpy()
        states = table.column('state').to_pylist()
        timestamps = table.column('client timestamp').to_numpy()

        # 过滤 Move 事件
        move_mask = np.array([s == 'Move' for s in states])
        x, y = x[move_mask], y[move_mask]
        timestamps = timestamps[move_mask].astype(np.float64)

        if len(x) < 2:
            return result

        coords = np.column_stack([x, y]).astype(np.float32)

        # 按距离分割
        for traj_coords, traj_ts in self._split_by_distance(coords, timestamps):
            if len(traj_coords) < self.min_length:
                continue

            result.point_counts.append(len(traj_coords))
            result.complexities.append(self._compute_complexity(traj_coords))

            if traj_ts is not None and len(traj_ts) >= 2:
                dt = self._compute_dt(traj_ts)
                result.time_intervals.extend(dt.tolist())
                dur = traj_ts[-1] - traj_ts[0]
                if dur > 0:
                    result.total_durations.append(dur)

        return result

    def _analyze_boun_csv(self, table) -> AnalysisResult:
        """分析 BOUN 原始 CSV"""
        result = AnalysisResult()

        x = table.column('x').to_numpy()
        y = table.column('y').to_numpy()
        states = table.column('state').to_pylist()
        timestamps = table.column('client_timestamp').to_numpy()

        # 过滤 Move 事件
        move_mask = np.array([s == 'Move' for s in states])
        x, y = x[move_mask], y[move_mask]
        timestamps = timestamps[move_mask].astype(np.float64)

        if len(x) < self.min_length:
            return result

        coords = np.column_stack([x, y]).astype(np.float32)
        result.point_counts.append(len(x))
        result.complexities.append(self._compute_complexity(coords))

        if len(timestamps) >= 2:
            dt = self._compute_dt(timestamps)
            result.time_intervals.extend(dt.tolist())
            dur = timestamps[-1] - timestamps[0]
            if dur > 0:
                result.total_durations.append(dur)

        return result

    def _analyze_jsonl(self, file_path: Path) -> AnalysisResult:
        """分析 JSONL 文件 (Open Images Localized Narratives)"""
        result = AnalysisResult()

        try:
            print(f"加载 {file_path.name}...")
            table = pj.read_json(file_path)

            if 'traces' not in table.column_names:
                print(f"  无 traces 列")
                return result

            traces_col = table.column('traces')
            print(f"  共 {len(table)} 行")

            for i in tqdm(range(len(table)), desc=f"处理 {file_path.name}", leave=False):
                traces = traces_col[i].as_py()
                if not traces:
                    continue

                for trace in traces:
                    if not trace or len(trace) < self.min_length:
                        continue

                    x_list, y_list, t_list = [], [], []
                    for point in trace:
                        if 'x' in point and 'y' in point:
                            x_list.append(float(point['x']))
                            y_list.append(float(point['y']))
                            t_list.append(float(point.get('t', 0.0)))

                    if len(x_list) < self.min_length:
                        continue

                    result.point_counts.append(len(x_list))
                    coords = np.column_stack([x_list, y_list]).astype(np.float32)
                    result.complexities.append(self._compute_complexity(coords))

                    if t_list and len(t_list) >= 2:
                        ts = np.array(t_list, dtype=np.float64)
                        dt = self._compute_dt(ts)
                        result.time_intervals.extend(dt.tolist())
                        dur = ts[-1] - ts[0]
                        if dur > 0:
                            result.total_durations.append(dur)

        except Exception as e:
            print(f"读取错误 {file_path.name}: {e}")

        return result

    def _split_by_distance(self, coords: np.ndarray, timestamps: Optional[np.ndarray]):
        """按距离分割轨迹"""
        diffs = np.diff(coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        split_idx = np.where(distances > self.DISTANCE_THRESHOLD)[0] + 1
        split_idx = np.concatenate([[0], split_idx, [len(coords)]])

        for i in range(len(split_idx) - 1):
            start, end = split_idx[i], split_idx[i + 1]
            ts = timestamps[start:end] if timestamps is not None else None
            yield coords[start:end], ts

    def _compute_complexity(self, coords: np.ndarray) -> float:
        """计算 path_ratio"""
        if len(coords) < 2:
            return 1.0
        path_len = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
        straight = np.linalg.norm(coords[-1] - coords[0])
        if straight < 1e-8:
            return max(path_len, 1.0)
        return max(path_len / straight, 1.0)

    def _compute_dt(self, timestamps: np.ndarray) -> np.ndarray:
        """计算时间间隔（原始值）"""
        if len(timestamps) < 2:
            return np.array([])
        dt = np.diff(timestamps)
        return dt[dt >= 0]


def print_statistics(result: AnalysisResult, name: str):
    """打印统计信息"""
    if not result.point_counts:
        print(f"\n{name}: 无有效数据")
        return

    counts = np.array(result.point_counts)
    pcts = [5, 25, 50, 75, 95]

    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"{'='*60}")
    print(f"轨迹数: {len(counts):,}")

    print(f"\n--- 轨迹长度 ---")
    print(f"  范围: {counts.min()} - {counts.max()}, 均值: {counts.mean():.1f}, 中位: {np.median(counts):.1f}")
    print(f"  百分位: {', '.join(f'P{p}={np.percentile(counts, p):.0f}' for p in pcts)}")

    if result.complexities:
        comp = np.array(result.complexities)
        print(f"\n--- 复杂度 (α) ---")
        print(f"  范围: {comp.min():.2f} - {comp.max():.2f}, 均值: {comp.mean():.2f}, 中位: {np.median(comp):.2f}")

    if result.time_intervals:
        dt = np.array(result.time_intervals)
        print(f"\n--- 时间间隔 (dt) ---")
        print(f"  样本: {len(dt):,}, 范围: {dt.min():.4f} - {dt.max():.4f}")
        print(f"  均值: {dt.mean():.4f}, 中位: {np.median(dt):.4f}, 标准差: {dt.std():.4f}")
        print(f"  百分位: {', '.join(f'P{p}={np.percentile(dt, p):.4f}' for p in pcts)}")

    if result.total_durations:
        dur = np.array(result.total_durations)
        print(f"\n--- 总时长 ---")
        print(f"  范围: {dur.min():.4f} - {dur.max():.4f}, 均值: {dur.mean():.4f}, 中位: {np.median(dur):.4f}")


def plot_distributions(result: AnalysisResult, output_path: Path, name: str):
    """生成图表"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Dataset: {name}', fontsize=14)

    # 点数
    if result.point_counts:
        counts = np.array(result.point_counts)
        ax = axes[0, 0]
        ax.hist(counts, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.median(counts), color='r', linestyle='--', label=f'Median: {np.median(counts):.0f}')
        ax.set_xlabel('Points')
        ax.set_title('Trajectory Length')
        ax.legend()
        ax.set_xlim(0, np.percentile(counts, 99))

    # 复杂度
    if result.complexities:
        comp = np.array(result.complexities)
        ax = axes[0, 1]
        ax.hist(comp, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(np.median(comp), color='r', linestyle='--', label=f'Median: {np.median(comp):.2f}')
        ax.set_xlabel('Path Ratio (α)')
        ax.set_title('Complexity')
        ax.legend()
        ax.set_xlim(1, np.percentile(comp, 99))

    # dt
    ax = axes[1, 0]
    if result.time_intervals:
        dt = np.array(result.time_intervals)
        ax.hist(dt, bins=100, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.median(dt), color='r', linestyle='--', label=f'Median: {np.median(dt):.4f}')
        ax.set_xlabel('Time Interval (dt)')
        ax.set_title('dt Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # 时长
    ax = axes[1, 1]
    if result.total_durations:
        dur = np.array(result.total_durations)
        ax.hist(dur, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax.axvline(np.median(dur), color='r', linestyle='--', label=f'Median: {np.median(dur):.4f}')
        ax.set_xlabel('Duration')
        ax.set_title('Trajectory Duration')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="数据集分析工具")
    parser.add_argument("--sapimouse", type=str, help="SapiMouse 数据集路径")
    parser.add_argument("--boun", type=str, help="BOUN 数据集路径")
    parser.add_argument("--open-images", type=str, help="Open Images 数据集路径")
    parser.add_argument("--all", action="store_true", help="分析所有默认数据集")
    parser.add_argument("--min-length", type=int, default=10, help="最小轨迹长度")
    parser.add_argument("--plot", action="store_true", help="生成图表")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="图表输出目录")

    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent

    # 默认路径
    defaults = {
        'sapimouse': 'datasets/sapimouse',
        'boun': 'datasets/boun-processed',
        'open_images': 'datasets/open_images_v6',
    }

    analyzer = DatasetAnalyzer(min_length=args.min_length)
    combined = AnalysisResult()
    analyzed = []

    # 收集要分析的数据集
    datasets = []
    if args.sapimouse:
        datasets.append(('SapiMouse', args.sapimouse))
    if args.boun:
        datasets.append(('BOUN', args.boun))
    if args.open_images:
        datasets.append(('Open Images', args.open_images))

    if args.all:
        if not args.sapimouse:
            datasets.append(('SapiMouse', defaults['sapimouse']))
        if not args.boun:
            datasets.append(('BOUN', defaults['boun']))
        if not args.open_images:
            datasets.append(('Open Images', defaults['open_images']))

    if not datasets:
        parser.print_help()
        print("\n请指定至少一个数据集路径，或使用 --all")
        return

    # 分析各数据集
    for name, path in datasets:
        full_path = base_dir / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            print(f"\n{name} 路径不存在: {full_path}")
            continue

        print(f"\n分析 {name}: {full_path}")
        result = analyzer.analyze(str(full_path), dataset_type=name.lower())

        if result.count > 0:
            print_statistics(result, name)
            if args.plot:
                out_dir = base_dir / args.output_dir
                plot_distributions(result, out_dir / f"{name.lower().replace(' ', '_')}_analysis.png", name)
            combined.extend(result)
            analyzed.append(name)

    # 合并统计
    if len(analyzed) > 1 and combined.count > 0:
        print_statistics(combined, f"Combined ({', '.join(analyzed)})")
        if args.plot:
            out_dir = base_dir / args.output_dir
            plot_distributions(combined, out_dir / "combined_analysis.png", "Combined")


if __name__ == "__main__":
    main()
