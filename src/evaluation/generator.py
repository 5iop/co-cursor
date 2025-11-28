"""
轨迹生成器模块
用于生成人类风格的鼠标轨迹
"""
import torch
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.alpha_ddim import AlphaDDIM, create_alpha_ddim


class TrajectoryGenerator:
    """
    鼠标轨迹生成器

    支持：
    1. 给定起点终点生成轨迹
    2. 控制轨迹复杂度（通过alpha参数）
    3. 批量生成
    4. 后处理（去归一化、平滑等）
    """

    def __init__(
        self,
        model: AlphaDDIM = None,
        checkpoint_path: str = None,
        device: str = "cuda",
        seq_length: int = 500,
        screen_size: Tuple[int, int] = (1920, 1080),
    ):
        """
        Args:
            model: 预训练的AlphaDDIM模型
            checkpoint_path: 检查点路径（如果没有传入model）
            device: 运行设备
            seq_length: 生成轨迹的长度
            screen_size: 屏幕尺寸（用于去归一化）
        """
        self.device = device
        self.seq_length = seq_length
        self.screen_size = screen_size

        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_model(checkpoint_path)
        else:
            # 创建默认模型（未训练）
            self.model = create_alpha_ddim(
                seq_length=seq_length,
                device=device
            )

        self.model.eval()

    def _load_model(self, checkpoint_path: str) -> AlphaDDIM:
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 从检查点读取模型配置（如果有），否则使用默认值
        model_config = checkpoint.get('model_config', {})
        seq_length = model_config.get('seq_length', self.seq_length)
        timesteps = model_config.get('timesteps', 1000)
        base_channels = model_config.get('base_channels', 64)

        # 更新实例的seq_length以匹配加载的模型
        self.seq_length = seq_length

        model = create_alpha_ddim(
            seq_length=seq_length,
            timesteps=timesteps,
            base_channels=base_channels,
            device=self.device
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @torch.no_grad()
    def generate(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        alpha: float = 0.5,
        num_inference_steps: int = 50,
        normalized_input: bool = False,
        return_normalized: bool = False,
    ) -> np.ndarray:
        """
        生成单条轨迹 (论文 α-DDIM)

        Args:
            start_point: 起点坐标 (x, y)
            end_point: 终点坐标 (x, y)
            alpha: 目标复杂度 (论文推荐 0.3-0.8)
            num_inference_steps: 最大推理步数
            normalized_input: 输入坐标是否已归一化
            return_normalized: 是否返回归一化的坐标

        Returns:
            轨迹数组 (seq_length, 2)
        """
        # 归一化输入
        if not normalized_input:
            start = (
                start_point[0] / self.screen_size[0],
                start_point[1] / self.screen_size[1]
            )
            end = (
                end_point[0] / self.screen_size[0],
                end_point[1] / self.screen_size[1]
            )
        else:
            start = start_point
            end = end_point

        # 构建条件
        condition = torch.FloatTensor([
            start[0], start[1], end[0], end[1]
        ]).unsqueeze(0).to(self.device)

        # 生成 - α-DDIM 采样 (论文 Eq.4-9)
        # α 直接作为目标复杂度，内置熵控制保证停止
        trajectory = self.model.sample(
            batch_size=1,
            condition=condition,
            num_inference_steps=num_inference_steps,
            alpha=alpha,
            device=self.device,
        )

        trajectory = trajectory[0].cpu().numpy()

        # 去归一化
        if not return_normalized:
            trajectory[:, 0] *= self.screen_size[0]
            trajectory[:, 1] *= self.screen_size[1]

        return trajectory

    @torch.no_grad()
    def generate_batch(
        self,
        start_points: np.ndarray,
        end_points: np.ndarray,
        alpha: float = 0.5,
        num_inference_steps: int = 50,
        normalized_input: bool = False,
        return_normalized: bool = False,
    ) -> np.ndarray:
        """
        批量生成轨迹 (论文 α-DDIM)

        Args:
            start_points: 起点数组 (batch, 2)
            end_points: 终点数组 (batch, 2)
            alpha: 目标复杂度 (论文推荐 0.3-0.8)
            num_inference_steps: 最大推理步数
            normalized_input: 输入坐标是否已归一化
            return_normalized: 是否返回归一化的坐标

        Returns:
            轨迹数组 (batch, seq_length, 2)
        """
        batch_size = len(start_points)

        # 归一化输入
        if not normalized_input:
            start_points = start_points.copy()
            end_points = end_points.copy()
            start_points[:, 0] /= self.screen_size[0]
            start_points[:, 1] /= self.screen_size[1]
            end_points[:, 0] /= self.screen_size[0]
            end_points[:, 1] /= self.screen_size[1]

        # 构建条件
        condition = torch.FloatTensor(
            np.concatenate([start_points, end_points], axis=1)
        ).to(self.device)

        # 生成 - α-DDIM 采样 (论文 Eq.4-9)
        trajectories = self.model.sample(
            batch_size=batch_size,
            condition=condition,
            num_inference_steps=num_inference_steps,
            alpha=alpha,
            device=self.device,
        )

        trajectories = trajectories.cpu().numpy()

        # 去归一化
        if not return_normalized:
            trajectories[:, :, 0] *= self.screen_size[0]
            trajectories[:, :, 1] *= self.screen_size[1]

        return trajectories

    def generate_with_complexity_range(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_samples: int = 5,
        alpha_range: Tuple[float, float] = (0.3, 0.8),  # 论文推荐范围
    ) -> List[np.ndarray]:
        """
        生成不同复杂度的轨迹 (论文推荐 α ∈ [0.3, 0.8])

        Args:
            start_point: 起点
            end_point: 终点
            num_samples: 每个alpha值生成的样本数
            alpha_range: alpha参数范围

        Returns:
            轨迹列表
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_samples)
        trajectories = []

        for alpha in alphas:
            traj = self.generate(
                start_point, end_point,
                alpha=float(alpha)
            )
            trajectories.append(traj)

        return trajectories

    def smooth_trajectory(
        self,
        trajectory: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        平滑轨迹（移动平均）

        Args:
            trajectory: 输入轨迹 (seq_len, 2)
            window_size: 窗口大小

        Returns:
            平滑后的轨迹
        """
        smoothed = trajectory.copy()
        half_window = window_size // 2

        for i in range(half_window, len(trajectory) - half_window):
            smoothed[i] = np.mean(
                trajectory[i - half_window:i + half_window + 1],
                axis=0
            )

        return smoothed

    def interpolate_trajectory(
        self,
        trajectory: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        插值轨迹到目标长度

        Args:
            trajectory: 输入轨迹
            target_length: 目标长度

        Returns:
            插值后的轨迹
        """
        current_length = len(trajectory)
        if current_length == target_length:
            return trajectory

        indices = np.linspace(0, current_length - 1, target_length)
        interpolated = np.zeros((target_length, 2))

        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(lower + 1, current_length - 1)
            alpha = idx - lower
            interpolated[i] = (1 - alpha) * trajectory[lower] + alpha * trajectory[upper]

        return interpolated

    def add_timing(
        self,
        trajectory: np.ndarray,
        total_duration_ms: float = 500,
        human_like: bool = True
    ) -> np.ndarray:
        """
        为轨迹添加时间信息

        Args:
            trajectory: 轨迹坐标 (seq_len, 2)
            total_duration_ms: 总时长（毫秒）
            human_like: 是否使用人类风格的时间分布

        Returns:
            带时间戳的轨迹 (seq_len, 3) - [x, y, timestamp]
        """
        seq_len = len(trajectory)
        timed_trajectory = np.zeros((seq_len, 3))
        timed_trajectory[:, :2] = trajectory

        if human_like:
            # 人类风格：开始和结束较慢，中间较快
            t = np.linspace(0, 1, seq_len)
            # 使用S曲线
            timing = (1 - np.cos(np.pi * t)) / 2
            timestamps = timing * total_duration_ms
        else:
            # 均匀分布
            timestamps = np.linspace(0, total_duration_ms, seq_len)

        timed_trajectory[:, 2] = timestamps
        return timed_trajectory


class BezierGenerator:
    """
    贝塞尔曲线轨迹生成器（作为对比基线）
    """

    @staticmethod
    def generate(
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_points: int = 50,
        num_control_points: int = 2,
        randomness: float = 0.3
    ) -> np.ndarray:
        """
        使用贝塞尔曲线生成轨迹

        Args:
            start_point: 起点
            end_point: 终点
            num_points: 轨迹点数
            num_control_points: 控制点数量
            randomness: 控制点随机性
        """
        start = np.array(start_point)
        end = np.array(end_point)

        # 生成控制点
        control_points = [start]
        for i in range(num_control_points):
            t = (i + 1) / (num_control_points + 1)
            base = start + t * (end - start)
            # 添加随机偏移
            offset = np.random.randn(2) * randomness * np.linalg.norm(end - start)
            control_points.append(base + offset)
        control_points.append(end)

        control_points = np.array(control_points)

        # 生成贝塞尔曲线
        t_values = np.linspace(0, 1, num_points)
        trajectory = np.zeros((num_points, 2))

        for i, t in enumerate(t_values):
            trajectory[i] = BezierGenerator._de_casteljau(control_points, t)

        return trajectory

    @staticmethod
    def _de_casteljau(points: np.ndarray, t: float) -> np.ndarray:
        """De Casteljau算法计算贝塞尔曲线上的点"""
        n = len(points)
        if n == 1:
            return points[0]

        new_points = np.zeros((n - 1, 2))
        for i in range(n - 1):
            new_points[i] = (1 - t) * points[i] + t * points[i + 1]

        return BezierGenerator._de_casteljau(new_points, t)


class LinearGenerator:
    """
    线性轨迹生成器（最简单的基线）
    """

    @staticmethod
    def generate(
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_points: int = 50,
        noise_level: float = 0.0
    ) -> np.ndarray:
        """生成线性轨迹"""
        start = np.array(start_point)
        end = np.array(end_point)

        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        trajectory = start + t * (end - start)

        if noise_level > 0:
            noise = np.random.randn(num_points, 2) * noise_level
            trajectory += noise

        return trajectory


if __name__ == "__main__":
    # 测试生成器
    print("Testing generators...")

    # 测试线性生成器
    linear_gen = LinearGenerator()
    linear_traj = linear_gen.generate((100, 100), (500, 400))
    print(f"Linear trajectory shape: {linear_traj.shape}")

    # 测试贝塞尔生成器
    bezier_gen = BezierGenerator()
    bezier_traj = bezier_gen.generate((100, 100), (500, 400))
    print(f"Bezier trajectory shape: {bezier_traj.shape}")

    # 测试DMTG生成器（未训练模型）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dmtg_gen = TrajectoryGenerator(device=device)

    print("\nGenerating with DMTG (untrained model)...")
    dmtg_traj = dmtg_gen.generate(
        start_point=(100, 100),
        end_point=(500, 400),
        alpha=0.5
    )
    print(f"DMTG trajectory shape: {dmtg_traj.shape}")

    # 测试不同复杂度
    print("\nGenerating trajectories with different complexity levels...")
    trajectories = dmtg_gen.generate_with_complexity_range(
        (100, 100), (500, 400),
        num_samples=3,
        alpha_range=(0.1, 0.9)
    )
    for i, traj in enumerate(trajectories):
        print(f"  Alpha={0.1 + i * 0.4:.1f}: trajectory length = {len(traj)}")

    # 测试添加时间
    timed_traj = dmtg_gen.add_timing(dmtg_traj, total_duration_ms=500)
    print(f"\nTimed trajectory shape: {timed_traj.shape}")
    print(f"Time range: {timed_traj[0, 2]:.1f}ms - {timed_traj[-1, 2]:.1f}ms")
