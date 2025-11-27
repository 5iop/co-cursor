"""
鼠标轨迹数据可视化工具
支持CSV和JSONL格式（preprocess_boun.py处理后的数据）
优化大文件加载，支持归一化数据自动检测
"""
import sys
import tkinter as tk

# 使用 orjson 加速 JSON 解析
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
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import os

# 大文件阈值（MB）
LARGE_FILE_THRESHOLD_MB = 50
# 大文件默认加载的最大轨迹数
DEFAULT_MAX_TRAJECTORIES = 1000


class TrajectoryVisualizer:
    """轨迹可视化器"""

    def __init__(self, root):
        self.root = root
        self.root.title("鼠标轨迹可视化工具")
        self.root.geometry("1200x800")

        # 屏幕尺寸（用于归一化）
        self.screen_size = (1920, 1080)
        self.seq_length = 50
        self.distance_threshold = 100.0
        self.min_trajectory_length = 10
        self.max_trajectories = DEFAULT_MAX_TRAJECTORIES

        # 当前加载的轨迹
        self.trajectories = []
        self.current_file = None
        self.total_available = 0  # 文件中的总轨迹数

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 文件选择按钮
        ttk.Button(control_frame, text="选择文件(CSV/JSONL)", command=self._select_file).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="选择文件夹", command=self._select_folder).pack(fill=tk.X, pady=5)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 加载设置
        ttk.Label(control_frame, text="最大加载轨迹数:").pack(anchor=tk.W)
        self.max_traj_var = tk.StringVar(value=str(DEFAULT_MAX_TRAJECTORIES))
        ttk.Entry(control_frame, textvariable=self.max_traj_var, width=10).pack(fill=tk.X, pady=2)

        # 参数设置
        ttk.Label(control_frame, text="屏幕宽度:").pack(anchor=tk.W)
        self.screen_w_var = tk.StringVar(value="1920")
        ttk.Entry(control_frame, textvariable=self.screen_w_var, width=10).pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="屏幕高度:").pack(anchor=tk.W)
        self.screen_h_var = tk.StringVar(value="1080")
        ttk.Entry(control_frame, textvariable=self.screen_h_var, width=10).pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="重采样长度:").pack(anchor=tk.W)
        self.seq_len_var = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.seq_len_var, width=10).pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="分割阈值(像素):").pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.threshold_var, width=10).pack(fill=tk.X, pady=2)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 显示选项
        self.show_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示采样点", variable=self.show_points_var,
                       command=self._refresh_plot).pack(anchor=tk.W)

        self.show_endpoints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示起点/终点", variable=self.show_endpoints_var,
                       command=self._refresh_plot).pack(anchor=tk.W)

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="归一化显示", variable=self.normalize_var,
                       command=self._refresh_plot).pack(anchor=tk.W)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 轨迹选择
        ttk.Label(control_frame, text="轨迹选择:").pack(anchor=tk.W)
        self.traj_listbox = tk.Listbox(control_frame, height=10, selectmode=tk.EXTENDED)
        self.traj_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.traj_listbox.bind('<<ListboxSelect>>', self._on_select_trajectory)

        # 轨迹信息标签
        self.traj_info_var = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=self.traj_info_var, wraplength=180).pack(anchor=tk.W)

        ttk.Button(control_frame, text="全选", command=self._select_all).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="清除选择", command=self._clear_selection).pack(fill=tk.X, pady=2)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 导出按钮
        ttk.Button(control_frame, text="保存图片", command=self._save_image).pack(fill=tk.X, pady=5)

        # 右侧绘图区域
        plot_frame = ttk.LabelFrame(main_frame, text="轨迹显示", padding="5")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib图形
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="请选择CSV/JSONL文件或文件夹")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress_bar.pack_forget()  # 默认隐藏

        # 绑定拖拽事件
        self._setup_drag_drop()

    def _setup_drag_drop(self):
        """设置拖拽功能"""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            if hasattr(self.root, 'drop_target_register'):
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self._on_drop)
                self.status_var.set("请选择文件或拖拽CSV/JSONL文件到窗口")
        except ImportError:
            self.status_var.set("请选择CSV/JSONL文件（安装tkinterdnd2可支持拖拽）")

    def _on_drop(self, event):
        """处理拖拽事件"""
        files = self.root.tk.splitlist(event.data)
        supported_files = [f for f in files if f.lower().endswith(('.csv', '.jsonl'))]
        if supported_files:
            self._load_files(supported_files)

    def _select_file(self):
        """选择数据文件"""
        files = filedialog.askopenfilenames(
            title="选择数据文件",
            filetypes=[
                ("数据文件", "*.csv *.jsonl"),
                ("CSV files", "*.csv"),
                ("JSONL files", "*.jsonl"),
                ("All files", "*.*")
            ]
        )
        if files:
            self._load_files(list(files))

    def _select_folder(self):
        """选择文件夹"""
        folder = filedialog.askdirectory(title="选择包含数据文件的文件夹")
        if folder:
            folder_path = Path(folder)
            data_files = list(folder_path.glob("**/*.csv")) + list(folder_path.glob("**/*.jsonl"))
            if data_files:
                self._load_files([str(f) for f in data_files[:50]])
            else:
                messagebox.showwarning("警告", "文件夹中没有找到CSV或JSONL文件")

    def _show_progress(self, show=True):
        """显示/隐藏进度条"""
        if show:
            self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.progress_bar.pack_forget()
        self.root.update()

    def _update_progress(self, value, status_text=None):
        """更新进度"""
        self.progress_var.set(value)
        if status_text:
            self.status_var.set(status_text)
        self.root.update()

    def _load_files(self, files: List[str]):
        """加载数据文件（CSV或JSONL）"""
        self.trajectories = []
        self.traj_listbox.delete(0, tk.END)
        self.total_available = 0

        # 更新参数
        try:
            self.screen_size = (int(self.screen_w_var.get()), int(self.screen_h_var.get()))
            self.seq_length = int(self.seq_len_var.get())
            self.distance_threshold = float(self.threshold_var.get())
            self.max_trajectories = int(self.max_traj_var.get())
        except ValueError:
            messagebox.showerror("错误", "参数格式错误")
            return

        self._show_progress(True)
        total_trajs = 0
        skipped_trajs = 0

        for file_idx, file_path in enumerate(files):
            file_progress = (file_idx / len(files)) * 100
            self._update_progress(file_progress, f"正在加载: {Path(file_path).name}")

            # 检查文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            is_large_file = file_size_mb > LARGE_FILE_THRESHOLD_MB

            if is_large_file:
                self._update_progress(file_progress, f"正在加载大文件: {Path(file_path).name} ({file_size_mb:.1f}MB)")

            # 根据扩展名选择处理方法
            if file_path.lower().endswith('.jsonl'):
                trajs, file_total = self._process_jsonl(file_path, is_large_file)
                self.total_available += file_total
            else:
                trajs = self._process_csv(file_path)
                self.total_available += len(trajs)

            # 限制加载数量
            remaining = self.max_trajectories - total_trajs
            if remaining <= 0:
                skipped_trajs += len(trajs)
                continue

            for i, traj in enumerate(trajs[:remaining]):
                self.trajectories.append({
                    'file': Path(file_path).name,
                    'index': i,
                    'data': traj['data'],
                    'is_normalized': traj['is_normalized'],
                    'metadata': traj.get('metadata', {}),
                })
                meta = traj.get('metadata', {})
                label = f"{Path(file_path).stem}_{i} ({len(traj['data'])}pts)"
                if meta.get('user_id'):
                    label = f"{meta['user_id']}_{meta.get('trajectory_id', i)} ({len(traj['data'])}pts)"
                self.traj_listbox.insert(tk.END, label)
                total_trajs += 1

            if len(trajs) > remaining:
                skipped_trajs += len(trajs) - remaining

        self._show_progress(False)

        status = f"已加载 {total_trajs} 条轨迹"
        if self.total_available > total_trajs:
            status += f" (共 {self.total_available} 条可用，限制 {self.max_trajectories})"
        self.status_var.set(status)

        # 更新轨迹信息
        self.traj_info_var.set(f"加载: {total_trajs}/{self.total_available}")

        # 默认选中前几条
        if total_trajs > 0:
            for i in range(min(5, total_trajs)):
                self.traj_listbox.selection_set(i)
            self._refresh_plot()

    def _process_csv(self, file_path: str) -> List[dict]:
        """处理单个CSV文件"""
        try:
            df = pd.read_csv(file_path)

            if 'x' not in df.columns or 'y' not in df.columns:
                return []

            coords = df[['x', 'y']].values

            # 过滤移动事件
            if 'state' in df.columns:
                move_mask = (df['state'] == 'Move')
                coords = coords[move_mask]

            # 检测是否已归一化（使用更宽松的阈值）
            coord_max = coords.max()
            coord_min = coords.min()
            coord_median = np.median(coords)
            is_normalized = (coord_max <= 1.5 and coord_min >= -0.5 and
                            0.0 <= coord_median <= 1.0)

            # 分割轨迹
            raw_trajectories = self._split_trajectories(coords, is_normalized)

            results = []
            for raw_traj in raw_trajectories:
                if len(raw_traj) >= self.min_trajectory_length:
                    resampled = self._resample(raw_traj, self.seq_length)
                    # 如果是归一化数据，裁剪到[0, 1]范围
                    if is_normalized:
                        resampled = np.clip(resampled, 0.0, 1.0)
                    results.append({
                        'data': resampled,
                        'is_normalized': is_normalized,
                    })

            return results

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _process_jsonl(self, file_path: str, is_large_file: bool = False) -> Tuple[List[dict], int]:
        """
        处理JSONL文件（优化大文件处理）

        Returns:
            (轨迹列表, 文件中总轨迹数)
        """
        results = []
        total_count = 0

        try:
            file_size = os.path.getsize(file_path)
            processed_bytes = 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    processed_bytes += len(line.encode('utf-8'))

                    # 大文件时更新进度
                    if is_large_file and line_num % 1000 == 0:
                        progress = (processed_bytes / file_size) * 100
                        self._update_progress(progress,
                            f"处理 {Path(file_path).name}: {progress:.0f}% ({len(results)} 轨迹)")

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json_loads(line)
                    except JSONDecodeError:
                        continue

                    traces = record.get('traces', [])
                    for trace in traces:
                        if not trace:
                            continue

                        coords = []
                        for point in trace:
                            if 'x' in point and 'y' in point:
                                coords.append([point['x'], point['y']])

                        if len(coords) < self.min_trajectory_length:
                            continue

                        total_count += 1

                        # 如果已达到最大数量，只计数不加载
                        if len(results) >= self.max_trajectories:
                            continue

                        raw_traj = np.array(coords, dtype=np.float32)

                        # 检测是否已归一化（JSONL通常已归一化）
                        # 使用更宽松的阈值，因为有些数据可能略微超出[0,1]范围
                        # 如果大部分值在[-0.1, 1.1]范围内，且中位数在[0, 1]，则认为是归一化数据
                        coord_max = raw_traj.max()
                        coord_min = raw_traj.min()
                        coord_median = np.median(raw_traj)
                        is_normalized = (coord_max <= 1.5 and coord_min >= -0.5 and
                                        0.0 <= coord_median <= 1.0)

                        resampled = self._resample(raw_traj, self.seq_length)

                        # 如果是归一化数据，裁剪到[0, 1]范围
                        if is_normalized:
                            resampled = np.clip(resampled, 0.0, 1.0)

                        results.append({
                            'data': resampled,
                            'is_normalized': is_normalized,
                            'metadata': {
                                'user_id': record.get('user_id', ''),
                                'session_id': record.get('session_id', ''),
                                'trajectory_id': record.get('trajectory_id', 0),
                                'test_type': record.get('test_type', ''),
                            }
                        })

            return results, total_count

        except Exception as e:
            print(f"Error processing JSONL {file_path}: {e}")
            return [], 0

    def _split_trajectories(self, coords: np.ndarray, is_normalized: bool = False) -> List[np.ndarray]:
        """分割轨迹"""
        if len(coords) < 2:
            return []

        # 归一化数据使用更小的阈值
        threshold = 0.05 if is_normalized else self.distance_threshold

        trajectories = []
        current_traj = [coords[0]]

        for i in range(1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[i-1])
            if dist > threshold:
                if len(current_traj) >= self.min_trajectory_length:
                    trajectories.append(np.array(current_traj))
                current_traj = [coords[i]]
            else:
                current_traj.append(coords[i])

        if len(current_traj) >= self.min_trajectory_length:
            trajectories.append(np.array(current_traj))

        return trajectories

    def _resample(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
        """重采样轨迹"""
        n_points = len(trajectory)
        if n_points == target_length:
            return trajectory.astype(np.float32)

        indices = np.linspace(0, n_points - 1, target_length)
        resampled = np.zeros((target_length, 2), dtype=np.float32)

        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(lower + 1, n_points - 1)
            alpha = idx - lower
            resampled[i] = (1 - alpha) * trajectory[lower] + alpha * trajectory[upper]

        return resampled

    def _normalize(self, coords: np.ndarray) -> np.ndarray:
        """归一化坐标（像素坐标 -> [0,1]）"""
        normalized = coords.copy()
        normalized[:, 0] = normalized[:, 0] / self.screen_size[0]
        normalized[:, 1] = normalized[:, 1] / self.screen_size[1]
        normalized = np.clip(normalized, 0, 1)
        return normalized

    def _denormalize(self, coords: np.ndarray) -> np.ndarray:
        """反归一化坐标（[0,1] -> 像素坐标）"""
        denormalized = coords.copy()
        denormalized[:, 0] = denormalized[:, 0] * self.screen_size[0]
        denormalized[:, 1] = denormalized[:, 1] * self.screen_size[1]
        return denormalized

    def _on_select_trajectory(self, event):
        """轨迹选择事件"""
        self._refresh_plot()

    def _select_all(self):
        """全选轨迹"""
        self.traj_listbox.selection_set(0, tk.END)
        self._refresh_plot()

    def _clear_selection(self):
        """清除选择"""
        self.traj_listbox.selection_clear(0, tk.END)
        self._refresh_plot()

    def _refresh_plot(self):
        """刷新绘图"""
        self.ax.clear()

        selected_indices = list(self.traj_listbox.curselection())
        if not selected_indices:
            self.ax.set_title("请选择轨迹")
            self.canvas.draw()
            return

        use_normalized = self.normalize_var.get()
        show_points = self.show_points_var.get()
        show_endpoints = self.show_endpoints_var.get()

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, idx in enumerate(selected_indices):
            traj_data = self.trajectories[idx]
            data = traj_data['data']
            is_normalized = traj_data['is_normalized']

            # 根据显示设置和数据状态转换坐标
            if use_normalized:
                if is_normalized:
                    traj = data  # 已归一化，直接使用
                else:
                    traj = self._normalize(data)  # 需要归一化
            else:
                if is_normalized:
                    traj = self._denormalize(data)  # 反归一化为像素坐标
                else:
                    traj = data  # 已是像素坐标

            color = colors[i % 10]

            # 绘制轨迹线
            self.ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, alpha=0.7)

            # 绘制采样点
            if show_points:
                self.ax.scatter(traj[:, 0], traj[:, 1], c=[color], s=10, alpha=0.5)

            # 绘制起点和终点
            if show_endpoints:
                self.ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=5, edgecolors='white')
                self.ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='o', zorder=5, edgecolors='white')

        # 设置坐标轴
        if use_normalized:
            self.ax.set_xlim(-0.05, 1.05)
            self.ax.set_ylim(-0.05, 1.05)
            self.ax.set_xlabel("X (归一化)")
            self.ax.set_ylabel("Y (归一化)")
        else:
            self.ax.set_xlim(-50, self.screen_size[0] + 50)
            self.ax.set_ylim(-50, self.screen_size[1] + 50)
            self.ax.set_xlabel("X (像素)")
            self.ax.set_ylabel("Y (像素)")

        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"轨迹显示 ({len(selected_indices)} 条)")

        # 反转Y轴（屏幕坐标系Y向下）
        self.ax.invert_yaxis()

        self.canvas.draw()

    def _save_image(self):
        """保存图片"""
        file_path = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=150, bbox_inches='tight')
            self.status_var.set(f"图片已保存到 {file_path}")


def main():
    # 尝试使用支持拖拽的Tk
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        print("提示: 安装 tkinterdnd2 可支持拖拽功能")
        print("  pip install tkinterdnd2")
        root = tk.Tk()

    app = TrajectoryVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
