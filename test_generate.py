"""
DMTG 轨迹生成测试脚本
测试训练好的模型生成人类风格鼠标轨迹
"""
import sys
import argparse

# 解析参数以决定是否使用 Agg 后端（必须在 import matplotlib.pyplot 之前）
_temp_parser = argparse.ArgumentParser(add_help=False)
_temp_parser.add_argument("--no_display", action="store_true")
_temp_args, _ = _temp_parser.parse_known_args()

if _temp_args.no_display:
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端，避免 X11 依赖

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import platform

from src.models.alpha_ddim import create_alpha_ddim, EntropyController
from src.data.dataset import denormalize_dt
from src.utils.notify import send_image_result

# 设置中文字体
def setup_chinese_font():
    """配置 matplotlib 支持中文"""
    system = platform.system()
    if system == 'Windows':
        # Windows 系统使用微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    elif system == 'Darwin':
        # macOS 使用苹方
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'DejaVu Sans']
    else:
        # Linux 使用文泉驿
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

setup_chinese_font()


def show_figure_with_scroll(fig, title="DMTG 轨迹生成结果"):
    """
    在带滚动条的窗口中显示 matplotlib 图表
    """
    # 延迟导入 tkinter（只在需要显示时）
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # 创建主窗口
    root = tk.Tk()
    root.title(title)

    # 获取屏幕尺寸
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 设置窗口大小（屏幕的80%）
    window_width = int(screen_width * 0.85)
    window_height = int(screen_height * 0.85)

    # 居中显示
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 创建画布和滚动条
    canvas = tk.Canvas(main_frame)
    v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)

    # 配置画布
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    # 布局
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 创建内部框架
    inner_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

    # 将 matplotlib 图表嵌入 tkinter
    fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack()

    # 更新滚动区域
    def update_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner_frame.bind("<Configure>", update_scroll_region)

    # 鼠标滚轮支持
    def on_mousewheel(event):
        # Windows
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_shift_mousewheel(event):
        # 水平滚动
        canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)
    canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

    # 关闭窗口时的处理
    def on_closing():
        plt.close(fig)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 运行主循环
    root.mainloop()


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 获取模型配置
    config = checkpoint.get('model_config', {})
    seq_length = config.get('seq_length', 500)
    timesteps = config.get('timesteps', 1000)
    input_dim = config.get('input_dim', 3)
    base_channels = config.get('base_channels', 96)
    enable_length_prediction = config.get('enable_length_prediction', False)

    # 创建与训练时相同配置的模型
    from src.models.unet import TrajectoryUNet
    unet = TrajectoryUNet(
        seq_length=seq_length,
        input_dim=input_dim,
        base_channels=base_channels,
        enable_length_prediction=enable_length_prediction,
    )
    model = AlphaDDIM(
        model=unet,
        timesteps=timesteps,
        seq_length=seq_length,
        input_dim=input_dim,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch}")
    print(f"  Config: base_channels={base_channels}, length_pred={enable_length_prediction}")

    return model


def compute_trajectory_metrics(trajectory: np.ndarray) -> dict:
    """计算轨迹的各种指标

    Args:
        trajectory: (N, 2) 或 (N, 3) 轨迹，支持 [x, y] 或 [x, y, dt]
    """
    # 只使用 x, y 计算几何指标
    traj_xy = trajectory[:, :2]

    # 路径长度
    segments = np.diff(traj_xy, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    path_length = np.sum(segment_lengths)

    # 直线距离
    straight_dist = np.linalg.norm(traj_xy[-1] - traj_xy[0])

    # 路径比率 (复杂度)
    path_ratio = path_length / (straight_dist + 1e-8)

    # 曲率变化 (平滑度)
    if len(segments) > 1:
        # 计算方向变化
        directions = segments / (np.linalg.norm(segments, axis=1, keepdims=True) + 1e-8)
        dot_products = np.sum(directions[:-1] * directions[1:], axis=1)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(dot_products)
        curvature = np.mean(angles)
    else:
        curvature = 0

    # 速度变化
    speed_std = np.std(segment_lengths)

    result = {
        'path_length': path_length,
        'straight_dist': straight_dist,
        'path_ratio': path_ratio,
        'curvature': curvature,
        'speed_std': speed_std,
    }

    # 如果有 dt 列，计算时间相关指标
    if trajectory.shape[1] >= 3:
        dt_norm = trajectory[:, 2]  # 归一化后的时间差
        dt_ms = denormalize_dt(dt_norm)  # 反归一化得到真实毫秒值

        # 归一化值 (内部使用)
        result['total_dt_norm'] = np.sum(dt_norm)
        result['mean_dt_norm'] = np.mean(dt_norm[1:]) if len(dt_norm) > 1 else 0

        # 真实毫秒值 (人类可读)
        result['total_time_ms'] = np.sum(dt_ms)  # 总时间 (ms)
        result['mean_dt_ms'] = np.mean(dt_ms[1:]) if len(dt_ms) > 1 else 0  # 平均帧间隔 (ms)

    return result


def compute_acceleration_directions(trajectory: np.ndarray) -> np.ndarray:
    """
    计算轨迹的加速度方向 (论文 Figure 6)

    加速度 = 速度变化 = 二阶差分
    返回加速度方向角度 (弧度，相对于 x 轴正方向)

    Args:
        trajectory: 轨迹坐标 (N, 2) 或 (N, 3)，只使用 x, y

    Returns:
        加速度方向角度数组 (N-2,)，范围 [-π, π]
    """
    # 只使用 x, y
    traj_xy = trajectory[:, :2]

    if len(traj_xy) < 3:
        return np.array([])

    # 速度 = 一阶差分
    velocity = np.diff(traj_xy, axis=0)  # (N-1, 2)

    # 加速度 = 二阶差分 = 速度的一阶差分
    acceleration = np.diff(velocity, axis=0)  # (N-2, 2)

    # 计算加速度方向角度
    angles = np.arctan2(acceleration[:, 1], acceleration[:, 0])  # (N-2,)

    return angles


def classify_trajectory_direction(trajectory: np.ndarray) -> str:
    """
    判断轨迹是上行还是下行

    Args:
        trajectory: 轨迹坐标 (N, 2) 或 (N, 3)

    Returns:
        "up" 或 "down"
    """
    dy = trajectory[-1, 1] - trajectory[0, 1]  # y 坐标差
    return "up" if dy > 0 else "down"


def plot_acceleration_distribution(results: list, save_path: str = None, no_display: bool = False):
    """
    绘制加速度方向分布图 (论文 Figure 6 风格)

    对比上行和下行轨迹的加速度方向分布
    """
    # 分类轨迹
    up_angles = []
    down_angles = []

    for r in results:
        traj = r['trajectory']
        direction = classify_trajectory_direction(traj)
        angles = compute_acceleration_directions(traj)

        if len(angles) > 0:
            if direction == "up":
                up_angles.extend(angles)
            else:
                down_angles.extend(angles)

    up_angles = np.array(up_angles)
    down_angles = np.array(down_angles)

    if len(up_angles) == 0 and len(down_angles) == 0:
        print("警告: 没有足够的数据计算加速度分布")
        return None

    # 创建极坐标直方图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection='polar'))

    # 设置 bin 数量
    n_bins = 36  # 每10度一个bin

    # 上行轨迹
    ax = axes[0]
    if len(up_angles) > 0:
        counts, bin_edges = np.histogram(up_angles, bins=n_bins, range=(-np.pi, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 2 * np.pi / n_bins
        # 归一化
        counts_norm = counts / counts.max() if counts.max() > 0 else counts
        ax.bar(bin_centers, counts_norm, width=width, alpha=0.7, color='blue', edgecolor='darkblue')
    ax.set_title('上行轨迹 (Upward)\n加速度方向分布', fontsize=12, pad=15)
    ax.set_theta_zero_location('E')  # 0度在右侧
    ax.set_theta_direction(-1)  # 顺时针

    # 下行轨迹
    ax = axes[1]
    if len(down_angles) > 0:
        counts, bin_edges = np.histogram(down_angles, bins=n_bins, range=(-np.pi, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 2 * np.pi / n_bins
        counts_norm = counts / counts.max() if counts.max() > 0 else counts
        ax.bar(bin_centers, counts_norm, width=width, alpha=0.7, color='red', edgecolor='darkred')
    ax.set_title('下行轨迹 (Downward)\n加速度方向分布', fontsize=12, pad=15)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)

    # 添加统计信息
    fig.suptitle(f'加速度方向分布 (论文 Fig.6)\n上行: {len(up_angles)} 点, 下行: {len(down_angles)} 点',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存加速度分布图: {save_path}")

    if no_display:
        plt.close(fig)
    else:
        plt.show()

    return fig


def generate_trajectories(
    model,
    start_points: list,
    end_points: list,
    alphas: list,
    device: str = "cuda",
    num_inference_steps: int = 50,
):
    """批量生成轨迹"""
    results = []

    for (start, end) in zip(start_points, end_points):
        for alpha in alphas:
            condition = torch.tensor(
                [[start[0], start[1], end[0], end[1]]],
                device=device
            )

            with torch.no_grad():
                traj = model.sample(
                    batch_size=1,
                    condition=condition,
                    alpha=alpha,
                    num_inference_steps=num_inference_steps,
                    device=device,
                )

            traj_np = traj[0].cpu().numpy()
            metrics = compute_trajectory_metrics(traj_np)

            results.append({
                'start': start,
                'end': end,
                'alpha': alpha,
                'trajectory': traj_np,
                'metrics': metrics,
            })

    return results


def plot_trajectories_by_alpha(results: list, save_path: str = None):
    """按 alpha 值分组绘制轨迹"""
    alphas = sorted(set(r['alpha'] for r in results))
    n_alphas = len(alphas)

    fig, axes = plt.subplots(1, n_alphas, figsize=(5 * n_alphas, 5))
    if n_alphas == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        alpha_results = [r for r in results if r['alpha'] == alpha]

        for r in alpha_results:
            traj = r['trajectory']
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=80, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=80, zorder=5)

        # 计算平均指标
        avg_ratio = np.mean([r['metrics']['path_ratio'] for r in alpha_results])
        avg_curv = np.mean([r['metrics']['curvature'] for r in alpha_results])

        ax.set_title(f'α={alpha}\nratio={avg_ratio:.2f}, curv={avg_curv:.3f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_trajectories_grid(results: list, save_path: str = None):
    """绘制轨迹网格图"""
    # 按起点-终点对分组
    pairs = list(set((tuple(r['start']), tuple(r['end'])) for r in results))
    alphas = sorted(set(r['alpha'] for r in results))

    n_pairs = len(pairs)
    n_alphas = len(alphas)

    fig, axes = plt.subplots(n_pairs, n_alphas, figsize=(4 * n_alphas, 4 * n_pairs))

    if n_pairs == 1:
        axes = [axes]
    if n_alphas == 1:
        axes = [[ax] for ax in axes]

    for i, (start, end) in enumerate(pairs):
        for j, alpha in enumerate(alphas):
            ax = axes[i][j] if n_pairs > 1 else axes[j]

            # 找到对应的结果
            matching = [r for r in results
                       if tuple(r['start']) == start
                       and tuple(r['end']) == end
                       and r['alpha'] == alpha]

            for r in matching:
                traj = r['trajectory']
                ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
                ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, zorder=5)
                ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, zorder=5)

                # 绘制直线参考
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       'k--', alpha=0.3, linewidth=1)

            if matching:
                ratio = matching[0]['metrics']['path_ratio']
                ax.set_title(f'α={alpha}, ratio={ratio:.2f}')

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_metrics_comparison(results: list, save_path: str = None):
    """绘制指标对比图"""
    alphas = sorted(set(r['alpha'] for r in results))

    # 按 alpha 分组计算指标
    metrics_by_alpha = {}
    for alpha in alphas:
        alpha_results = [r for r in results if r['alpha'] == alpha]
        metrics_by_alpha[alpha] = {
            'path_ratio': [r['metrics']['path_ratio'] for r in alpha_results],
            'curvature': [r['metrics']['curvature'] for r in alpha_results],
            'speed_std': [r['metrics']['speed_std'] for r in alpha_results],
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 路径比率
    ax = axes[0]
    data = [metrics_by_alpha[a]['path_ratio'] for a in alphas]
    bp = ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Path Ratio')
    ax.set_title('Path Ratio vs Alpha')
    ax.grid(True, alpha=0.3)

    # 曲率
    ax = axes[1]
    data = [metrics_by_alpha[a]['curvature'] for a in alphas]
    bp = ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Curvature (rad)')
    ax.set_title('Curvature vs Alpha')
    ax.grid(True, alpha=0.3)

    # 速度变化
    ax = axes[2]
    data = [metrics_by_alpha[a]['speed_std'] for a in alphas]
    bp = ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Speed Std')
    ax.set_title('Speed Variation vs Alpha')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_combined_results(results: list, save_path: str = None, effective_length: int = None):
    """
    绘制综合结果图：轨迹 + 指标，合并为一张大图
    """
    alphas = sorted(set(r['alpha'] for r in results))
    n_alphas = len(alphas)

    # 创建大图: 上面是轨迹，下面是指标
    fig = plt.figure(figsize=(5 * n_alphas, 10))

    # 上半部分：按 alpha 分组的轨迹
    for i, alpha in enumerate(alphas):
        ax = fig.add_subplot(2, n_alphas, i + 1)
        alpha_results = [r for r in results if r['alpha'] == alpha]

        for r in alpha_results:
            traj = r['trajectory']
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=80, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=80, zorder=5)

        # 计算平均指标
        avg_ratio = np.mean([r['metrics']['path_ratio'] for r in alpha_results])
        avg_curv = np.mean([r['metrics']['curvature'] for r in alpha_results])

        m_str = f", m={effective_length}" if effective_length else ""
        ax.set_title(f'α={alpha}{m_str}\n路径比={avg_ratio:.2f}, 曲率={avg_curv:.3f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # 下半部分：指标对比 (3个子图)
    metrics_by_alpha = {}
    for alpha in alphas:
        alpha_results = [r for r in results if r['alpha'] == alpha]
        metrics_by_alpha[alpha] = {
            'path_ratio': [r['metrics']['path_ratio'] for r in alpha_results],
            'curvature': [r['metrics']['curvature'] for r in alpha_results],
            'speed_std': [r['metrics']['speed_std'] for r in alpha_results],
        }

    # 路径比率
    ax = fig.add_subplot(2, 3, 4)
    data = [metrics_by_alpha[a]['path_ratio'] for a in alphas]
    ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('路径比率')
    ax.set_title('路径比率 vs Alpha')
    ax.grid(True, alpha=0.3)

    # 曲率
    ax = fig.add_subplot(2, 3, 5)
    data = [metrics_by_alpha[a]['curvature'] for a in alphas]
    ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('曲率 (rad)')
    ax.set_title('曲率 vs Alpha')
    ax.grid(True, alpha=0.3)

    # 速度变化
    ax = fig.add_subplot(2, 3, 6)
    data = [metrics_by_alpha[a]['speed_std'] for a in alphas]
    ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('速度标准差')
    ax.set_title('速度变化 vs Alpha')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")

    # 使用带滚动条的窗口显示
    m_str = f" (m={effective_length})" if effective_length else ""
    show_figure_with_scroll(fig, title=f"DMTG 轨迹生成结果{m_str}")


def plot_combined_results_no_display(results: list, save_path: str = None, effective_length: int = None, label: str = None):
    """
    绘制综合结果图（不显示窗口，仅保存文件）
    用于后台执行
    """
    alphas = sorted(set(r['alpha'] for r in results))
    n_alphas = len(alphas)

    # 创建大图: 上面是轨迹，下面是指标
    fig = plt.figure(figsize=(5 * n_alphas, 10))

    # 添加总标题（如果有 label）
    if label:
        fig.suptitle(f'实验: {label}', fontsize=16, fontweight='bold')

    # 上半部分：按 alpha 分组的轨迹
    for i, alpha in enumerate(alphas):
        ax = fig.add_subplot(2, n_alphas, i + 1)
        alpha_results = [r for r in results if r['alpha'] == alpha]

        for r in alpha_results:
            traj = r['trajectory']
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=80, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=80, zorder=5)

        avg_ratio = np.mean([r['metrics']['path_ratio'] for r in alpha_results])
        avg_curv = np.mean([r['metrics']['curvature'] for r in alpha_results])

        m_str = f", m={effective_length}" if effective_length else ""
        ax.set_title(f'α={alpha}{m_str}\n路径比={avg_ratio:.2f}, 曲率={avg_curv:.3f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # 下半部分：指标对比
    metrics_by_alpha = {}
    for alpha in alphas:
        alpha_results = [r for r in results if r['alpha'] == alpha]
        metrics_by_alpha[alpha] = {
            'path_ratio': [r['metrics']['path_ratio'] for r in alpha_results],
            'curvature': [r['metrics']['curvature'] for r in alpha_results],
            'speed_std': [r['metrics']['speed_std'] for r in alpha_results],
        }

    ax = fig.add_subplot(2, 3, 4)
    data = [metrics_by_alpha[a]['path_ratio'] for a in alphas]
    ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('路径比率')
    ax.set_title('路径比率 vs Alpha')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    data = [metrics_by_alpha[a]['curvature'] for a in alphas]
    ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('曲率 (rad)')
    ax.set_title('曲率 vs Alpha')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
    data = [metrics_by_alpha[a]['speed_std'] for a in alphas]
    ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('速度标准差')
    ax.set_title('速度变化 vs Alpha')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if label:
        plt.subplots_adjust(top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")

    plt.close(fig)


def print_metrics_table(results: list):
    """打印指标表格"""
    # 检查是否有时间数据
    has_time = any('total_time_ms' in r['metrics'] for r in results)

    print("\n" + "=" * 90)
    print("轨迹指标统计")
    print("=" * 90)

    if has_time:
        print(f"{'Alpha':<8} {'Path Ratio':<12} {'Curvature':<12} {'Speed Std':<12} {'Total(ms)':<12} {'Mean dt(ms)':<12}")
    else:
        print(f"{'Alpha':<8} {'Path Ratio':<12} {'Curvature':<12} {'Speed Std':<12}")
    print("-" * 90)

    alphas = sorted(set(r['alpha'] for r in results))

    for alpha in alphas:
        alpha_results = [r for r in results if r['alpha'] == alpha]

        ratios = [r['metrics']['path_ratio'] for r in alpha_results]
        curvs = [r['metrics']['curvature'] for r in alpha_results]
        speeds = [r['metrics']['speed_std'] for r in alpha_results]

        if has_time:
            total_times = [r['metrics'].get('total_time_ms', 0) for r in alpha_results]
            mean_dts = [r['metrics'].get('mean_dt_ms', 0) for r in alpha_results]
            print(f"{alpha:<8} {np.mean(ratios):<12.3f} {np.mean(curvs):<12.3f} {np.mean(speeds):<12.4f} {np.mean(total_times):<12.1f} {np.mean(mean_dts):<12.2f}")
        else:
            print(f"{alpha:<8} {np.mean(ratios):<12.3f} {np.mean(curvs):<12.3f} {np.mean(speeds):<12.4f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Test DMTG trajectory generation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples per configuration")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for plots")
    parser.add_argument("--effective_length", "-m", type=int, default=None,
                       help="Effective trajectory length (node count m)")
    parser.add_argument("--label", "-l", type=str, default=None,
                       help="Label for this experiment (appended to output filename)")
    parser.add_argument("--no_display", action="store_true",
                       help="Don't display the plot (useful for background execution)")
    parser.add_argument("--acceleration_dist", action="store_true",
                       help="Generate acceleration direction distribution plot (Fig.6)")
    parser.add_argument("--webhook", type=str, default=None,
                       help="Webhook URL to send result images (e.g. https://ntfy.jangit.me/notify/notifytg)")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")

    # 加载模型
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # 定义测试配置
    # 不同的起点-终点对
    test_pairs = [
        ((0.1, 0.1), (0.9, 0.9)),  # 对角线
        ((0.1, 0.5), (0.9, 0.5)),  # 水平
        ((0.5, 0.1), (0.5, 0.9)),  # 垂直
        ((0.2, 0.8), (0.8, 0.2)),  # 反对角线
    ]

    # 不同的 alpha 值 (path_ratio ∈ [1, +∞), α=1 为直线)
    alphas = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    # 生成轨迹
    print(f"\nGenerating trajectories...")
    print(f"  - {len(test_pairs)} start-end pairs")
    print(f"  - {len(alphas)} alpha values: {alphas}")
    print(f"  - {args.num_samples} samples each")
    if args.effective_length is not None:
        print(f"  - effective_length (m): {args.effective_length}")
    else:
        print(f"  - effective_length (m): 自动预测 (auto_length)")

    all_results = []

    for start, end in test_pairs:
        for alpha in alphas:
            for _ in range(args.num_samples):
                condition = torch.tensor(
                    [[start[0], start[1], end[0], end[1]]],
                    device=args.device
                )

                with torch.no_grad():
                    if args.effective_length is not None:
                        # 用户指定了 -m 参数，使用指定长度
                        traj = model.sample(
                            batch_size=1,
                            condition=condition,
                            alpha=alpha,
                            num_inference_steps=50,
                            device=args.device,
                            effective_length=args.effective_length,
                        )
                        m = args.effective_length
                    else:
                        # 未指定 -m，使用自动长度预测
                        traj, pred_lengths = model.sample_with_auto_length(
                            batch_size=1,
                            condition=condition,
                            alpha=alpha,
                            num_inference_steps=50,
                            device=args.device,
                        )
                        m = pred_lengths[0].item()

                traj_np = traj[0].cpu().numpy()

                # 只取有效部分
                m = int(m)  # 确保是整数
                traj_valid = traj_np[:m]

                metrics = compute_trajectory_metrics(traj_valid)

                all_results.append({
                    'start': start,
                    'end': end,
                    'alpha': alpha,
                    'trajectory': traj_valid,  # 只存有效部分
                    'metrics': metrics,
                    'effective_length': m,
                    'auto_length': args.effective_length is None,  # 是否使用自动长度
                })

    print(f"Generated {len(all_results)} trajectories")

    # 打印指标
    print_metrics_table(all_results)

    # 生成时间戳和文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.label:
        filename = f"generate_{args.label}.png"
    else:
        filename = f"generate_{timestamp}.png"

    # 绘制图表
    print("\n正在生成图表...")

    # 综合图：轨迹 + 指标合并显示
    # 如果使用自动长度，传入 "auto"；否则传入指定的长度
    eff_len_display = "auto" if args.effective_length is None else args.effective_length
    save_path = output_dir / filename

    if args.no_display:
        # 不显示窗口，只保存文件
        plot_combined_results_no_display(
            all_results,
            save_path=save_path,
            effective_length=eff_len_display,
            label=args.label,
        )
    else:
        plot_combined_results(
            all_results,
            save_path=save_path,
            effective_length=eff_len_display,
        )

    print(f"\n图表已保存到 {save_path}")

    # 发送 webhook 通知
    if args.webhook:
        print(f"\n正在发送通知到 webhook...")
        success = send_image_result(
            title=f"DMTG Generate - {args.label or 'test'}",
            image_path=str(save_path),
            description=f"Alphas: {alphas}\nSamples: {len(all_results)}",
            webhook_url=args.webhook,
        )
        if not success:
            print("通知发送失败")

    # 加速度方向分布图 (论文 Fig.6)
    if args.acceleration_dist:
        print("\n正在生成加速度方向分布图...")
        if args.label:
            accel_filename = f"acceleration_dist_{args.label}.png"
        else:
            accel_filename = f"acceleration_dist_{timestamp}.png"
        accel_save_path = output_dir / accel_filename
        plot_acceleration_distribution(
            all_results,
            save_path=accel_save_path,
            no_display=args.no_display
        )
        # 发送加速度分布图到 webhook
        if args.webhook:
            send_image_result(
                title=f"DMTG Acceleration - {args.label or 'test'}",
                image_path=str(accel_save_path),
                description="Acceleration direction distribution (Fig.6)",
                webhook_url=args.webhook,
            )

    print("完成!")


if __name__ == "__main__":
    main()
