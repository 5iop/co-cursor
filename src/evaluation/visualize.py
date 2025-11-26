"""
轨迹可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class TrajectoryVisualizer:
    """轨迹可视化工具"""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        style: str = 'default'
    ):
        self.figsize = figsize
        self.dpi = dpi
        if style != 'default':
            plt.style.use(style)

    def plot_single_trajectory(
        self,
        trajectory: np.ndarray,
        title: str = "Mouse Trajectory",
        show_points: bool = True,
        show_velocity: bool = False,
        color: str = 'blue',
        save_path: str = None,
    ) -> plt.Figure:
        """
        绘制单条轨迹

        Args:
            trajectory: 轨迹数组 (seq_len, 2)
            title: 图表标题
            show_points: 是否显示轨迹点
            show_velocity: 是否显示速度向量
            color: 轨迹颜色
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 绘制轨迹线
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2, alpha=0.8)

        if show_points:
            # 起点（绿色）
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      color='green', s=100, zorder=5, label='Start')
            # 终点（红色）
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color='red', s=100, zorder=5, label='End')
            # 中间点
            ax.scatter(trajectory[1:-1, 0], trajectory[1:-1, 1],
                      color=color, s=20, alpha=0.5)

        if show_velocity:
            velocity = np.diff(trajectory, axis=0)
            # 缩放速度向量以便可视化
            scale = np.max(np.abs(velocity)) / 20
            for i in range(0, len(trajectory) - 1, 5):
                ax.arrow(trajectory[i, 0], trajectory[i, 1],
                        velocity[i, 0] / scale, velocity[i, 1] / scale,
                        head_width=5, head_length=3, fc='gray', ec='gray', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 屏幕坐标系y轴向下

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_multiple_trajectories(
        self,
        trajectories: List[np.ndarray],
        labels: List[str] = None,
        title: str = "Multiple Trajectories",
        colors: List[str] = None,
        save_path: str = None,
    ) -> plt.Figure:
        """
        绘制多条轨迹对比

        Args:
            trajectories: 轨迹列表
            labels: 标签列表
            title: 图表标题
            colors: 颜色列表
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        if labels is None:
            labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]

        for traj, label, color in zip(trajectories, labels, colors):
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.7, label=label)
            ax.scatter(traj[0, 0], traj[0, 1], color=color, s=80, marker='o')
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=80, marker='x')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_trajectory_with_speed(
        self,
        trajectory: np.ndarray,
        title: str = "Trajectory with Speed",
        save_path: str = None,
    ) -> plt.Figure:
        """
        绘制带速度颜色映射的轨迹

        颜色表示速度：蓝色=慢，红色=快
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 计算速度
        velocity = np.diff(trajectory, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        # 创建线段集合
        points = trajectory.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 使用速度作为颜色
        norm = plt.Normalize(speed.min(), speed.max())
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(speed)
        lc.set_linewidth(3)

        ax.add_collection(lc)
        ax.autoscale()

        # 添加颜色条
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Speed')

        # 标记起点和终点
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                  color='green', s=100, zorder=5, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                  color='red', s=100, zorder=5, label='End')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_feature_comparison(
        self,
        generated_features: Dict[str, np.ndarray],
        real_features: Dict[str, np.ndarray],
        feature_names: List[str] = None,
        title: str = "Feature Distribution Comparison",
        save_path: str = None,
    ) -> plt.Figure:
        """
        绘制特征分布对比

        Args:
            generated_features: 生成轨迹的特征
            real_features: 真实轨迹的特征
            feature_names: 要比较的特征名称
            title: 图表标题
            save_path: 保存路径
        """
        if feature_names is None:
            feature_names = ['speed', 'acceleration', 'curvature']

        n_features = len(feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4), dpi=self.dpi)

        if n_features == 1:
            axes = [axes]

        for ax, name in zip(axes, feature_names):
            if name in generated_features and name in real_features:
                gen_data = generated_features[name]
                real_data = real_features[name]

                # 绘制直方图
                ax.hist(real_data, bins=30, alpha=0.5, label='Real', density=True, color='blue')
                ax.hist(gen_data, bins=30, alpha=0.5, label='Generated', density=True, color='orange')

                ax.set_xlabel(name)
                ax.set_ylabel('Density')
                ax.set_title(f'{name.capitalize()} Distribution')
                ax.legend()

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_velocity_profile(
        self,
        trajectory: np.ndarray,
        title: str = "Velocity Profile",
        save_path: str = None,
    ) -> plt.Figure:
        """
        绘制速度曲线

        人类轨迹通常呈现：加速 -> 匀速 -> 减速 的模式
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)

        velocity = np.diff(trajectory, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        acceleration = np.diff(velocity, axis=0)
        accel_magnitude = np.linalg.norm(acceleration, axis=1)

        t = np.arange(len(speed))

        # 1. 速度曲线
        axes[0, 0].plot(t, speed, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Speed')
        axes[0, 0].set_title('Speed over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 加速度曲线
        t_accel = np.arange(len(accel_magnitude))
        axes[0, 1].plot(t_accel, accel_magnitude, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Acceleration')
        axes[0, 1].set_title('Acceleration over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. X和Y速度分量
        axes[1, 0].plot(t, velocity[:, 0], 'g-', label='Vx', linewidth=2)
        axes[1, 0].plot(t, velocity[:, 1], 'm-', label='Vy', linewidth=2)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Velocity Component')
        axes[1, 0].set_title('Velocity Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 速度相位图
        axes[1, 1].plot(velocity[:, 0], velocity[:, 1], 'b-', alpha=0.5)
        axes[1, 1].scatter(velocity[0, 0], velocity[0, 1], color='green', s=100, label='Start')
        axes[1, 1].scatter(velocity[-1, 0], velocity[-1, 1], color='red', s=100, label='End')
        axes[1, 1].set_xlabel('Vx')
        axes[1, 1].set_ylabel('Vy')
        axes[1, 1].set_title('Velocity Phase Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_alpha_comparison(
        self,
        trajectories: List[np.ndarray],
        alphas: List[float],
        title: str = "Effect of Alpha Parameter",
        save_path: str = None,
    ) -> plt.Figure:
        """
        绘制不同alpha值生成的轨迹对比

        展示alpha参数对轨迹复杂度的影响
        """
        n_trajectories = len(trajectories)
        fig, axes = plt.subplots(1, n_trajectories, figsize=(5 * n_trajectories, 5), dpi=self.dpi)

        if n_trajectories == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_trajectories))

        for ax, traj, alpha, color in zip(axes, trajectories, alphas, colors):
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2)
            ax.scatter(traj[0, 0], traj[0, 1], color='green', s=80, marker='o')
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=80, marker='x')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'α = {alpha:.2f}')
            ax.set_aspect('equal')
            ax.invert_yaxis()

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig


def create_comparison_report(
    generated_trajectories: np.ndarray,
    real_trajectories: np.ndarray,
    output_dir: str = "results",
    prefix: str = "comparison"
):
    """
    生成完整的对比报告

    Args:
        generated_trajectories: 生成的轨迹 (n, seq_len, 2)
        real_trajectories: 真实轨迹 (n, seq_len, 2)
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = TrajectoryVisualizer()

    # 1. 单条轨迹对比
    fig = visualizer.plot_multiple_trajectories(
        [generated_trajectories[0], real_trajectories[0]],
        labels=['Generated', 'Real'],
        title='Single Trajectory Comparison',
        colors=['blue', 'orange']
    )
    fig.savefig(output_path / f'{prefix}_single_comparison.png')
    plt.close(fig)

    # 2. 速度曲线对比
    fig = visualizer.plot_velocity_profile(
        generated_trajectories[0],
        title='Generated Trajectory Velocity Profile'
    )
    fig.savefig(output_path / f'{prefix}_gen_velocity.png')
    plt.close(fig)

    fig = visualizer.plot_velocity_profile(
        real_trajectories[0],
        title='Real Trajectory Velocity Profile'
    )
    fig.savefig(output_path / f'{prefix}_real_velocity.png')
    plt.close(fig)

    # 3. 带速度的轨迹
    fig = visualizer.plot_trajectory_with_speed(
        generated_trajectories[0],
        title='Generated Trajectory with Speed'
    )
    fig.savefig(output_path / f'{prefix}_gen_speed_trajectory.png')
    plt.close(fig)

    print(f"Comparison report saved to {output_path}")


if __name__ == "__main__":
    # 测试可视化
    np.random.seed(42)

    # 生成测试轨迹
    t = np.linspace(0, 1, 50)
    trajectory = np.stack([
        100 + 400 * t + 20 * np.sin(5 * np.pi * t),
        100 + 300 * t + 15 * np.cos(3 * np.pi * t)
    ], axis=-1)

    visualizer = TrajectoryVisualizer()

    # 测试单条轨迹绘制
    print("Plotting single trajectory...")
    fig = visualizer.plot_single_trajectory(
        trajectory,
        title="Test Trajectory",
        show_velocity=True
    )
    plt.show()

    # 测试速度曲线
    print("Plotting velocity profile...")
    fig = visualizer.plot_velocity_profile(trajectory)
    plt.show()

    # 测试带速度的轨迹
    print("Plotting trajectory with speed...")
    fig = visualizer.plot_trajectory_with_speed(trajectory)
    plt.show()
