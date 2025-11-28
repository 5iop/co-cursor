"""
α-DDIM: 熵控制的DDIM扩散模型
论文核心创新：通过α参数控制生成轨迹的熵/随机性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from .unet import TrajectoryUNet


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """余弦噪声调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """线性噪声调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


class AlphaDDIM(nn.Module):
    """
    α-DDIM扩散模型

    核心特点：
    1. 通过α参数控制生成轨迹的熵（复杂度）
    2. 条件生成：给定起点和终点生成轨迹
    3. 使用DDIM加速采样
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        seq_length: int = 500,
        input_dim: int = 2,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.seq_length = seq_length
        self.input_dim = input_dim

        # 设置噪声调度
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        # 计算扩散参数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 注册为buffer（不参与梯度计算）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 前向扩散所需参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # 后验分布参数
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散：q(x_t | x_0)
        给定原始数据x_0，采样t时刻的噪声数据
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """从噪声预测原始数据x_0"""
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    @torch.no_grad()
    def ddim_sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: torch.Tensor,
        alpha: float = 0.5,  # 熵控制参数 (论文推荐 0.3-0.8)
        eta: float = 0.0,  # DDIM随机性参数
    ) -> torch.Tensor:
        """
        α-DDIM单步采样 (论文 Eq.4-6)

        核心创新: 双协方差混合
        Σ = α·Σ_d + (1-α)·Σ_n

        - Σ_d: 方向协方差 (沿起点到终点方向)
        - Σ_n: 各向同性噪声协方差
        - α: 控制复杂度，α大→更多方向性变化，α小→更接近直线

        Args:
            x_t: 当前时刻的噪声数据
            t: 当前时间步
            t_prev: 上一个时间步
            condition: 条件（起点+终点）
            alpha: 熵控制参数 (0.3-0.8, 越大越复杂)
            eta: DDIM随机性参数
        """
        batch_size = x_t.shape[0]
        alpha_tensor = torch.full((batch_size,), alpha, device=x_t.device, dtype=x_t.dtype)

        # 预测噪声 - 论文Eq.10: ε_θ(x_t, t, c, α)
        predicted_noise = self.model(x_t, t, condition, alpha_tensor)

        # 获取扩散参数
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev_raw = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        # 修复: 最后一步 (t_prev=0) 应该使用 alpha=1.0 (完全去噪状态)
        # 否则 sqrt(1 - alpha_t_prev) 不为0，会残留噪声
        alpha_t_prev = torch.where(
            t_prev.view(-1, *([1] * (len(x_t.shape) - 1))) == 0,
            torch.ones_like(alpha_t_prev_raw),
            alpha_t_prev_raw
        )

        # 预测x_0
        pred_x0 = self.predict_x0_from_noise(x_t, t, predicted_noise)

        # ========== 论文 Eq.4-6: 双协方差混合 ==========
        # 计算方向向量 (起点到终点)
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]    # (batch, 2)
        direction = end_point - start_point  # (batch, 2)
        direction_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-8
        direction_unit = direction / direction_norm  # 单位方向向量

        # 论文 Eq.4-5: kc 缩放因子
        # 协方差应随端点距离缩放: Σ_X = kc * ||d|| * (d_unit ⊗ d_unit)
        # 归一化坐标下最大距离为 sqrt(2)，将 kc 归一化到 [0, 1]
        kc = direction_norm / (1.414 + 1e-8)  # sqrt(2) ≈ 1.414

        # DDIM基础噪声方差
        sigma_base = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t + 1e-8) * (1 - alpha_t / (alpha_t_prev + 1e-8))
        )

        # 生成两种噪声分量
        # 1. 方向性噪声 (沿轨迹方向的随机扰动)
        # 论文 Eq.4: Σ_d = kc * ||d|| * direction ⊗ direction
        noise_along = torch.randn(batch_size, self.seq_length, 1, device=x_t.device)
        direction_unit_expanded = direction_unit.unsqueeze(1)  # (batch, 1, 2)
        kc_expanded = kc.unsqueeze(1)  # (batch, 1, 1)
        directional_noise = kc_expanded * noise_along * direction_unit_expanded  # (batch, seq, 2)

        # 2. 各向同性噪声 (垂直于轨迹方向)
        isotropic_noise = torch.randn_like(x_t)

        # 论文 Eq.5: Σ = α·Σ_d + (1-α)·Σ_n
        # α 大: 更多方向性变化 (复杂曲线)
        # α 小: 更多各向同性噪声 (接近直线但有抖动)
        mixed_noise = alpha * directional_noise + (1 - alpha) * isotropic_noise

        # 应用混合噪声
        sigma_t = sigma_base * (0.5 + 0.5 * alpha)  # α 大时噪声更强

        # DDIM更新公式
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_prev - sigma_t ** 2, min=0)) * predicted_noise

        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * mixed_noise

        return x_t_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: torch.Tensor,
        num_inference_steps: int = 50,
        alpha: float = 0.5,
        eta: float = 0.5,  # 默认启用随机性以支持双协方差混合
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        α-DDIM 采样 (论文 Eq.4-9)

        核心机制:
        1. 使用 α 控制双协方差混合 (Eq.4-6)
        2. 保证在达到目标复杂度时停止 (Eq.8-9)
        3. α 直接作为目标复杂度

        Args:
            batch_size: 批次大小
            condition: 条件张量 (batch, 4) - [start_x, start_y, end_x, end_y]
            num_inference_steps: 最大推理步数
            alpha: 目标复杂度 (论文推荐 0.3-0.8)
            eta: DDIM随机性参数
            device: 设备

        Returns:
            生成的轨迹 (batch, seq_length, 2)
        """
        self.model.eval()

        # 约束 α 到论文推荐范围
        alpha = max(0.3, min(0.8, alpha))

        # 论文 Eq.2: 掩码初始化
        x = self._initialize_with_condition(batch_size, condition, device)

        # 设置采样时间步
        # 防止 step_size 为 0（当 num_inference_steps > self.timesteps 时）
        step_size = max(1, self.timesteps // num_inference_steps)
        timesteps = list(range(0, self.timesteps, step_size))[::-1]

        # 论文 Eq.8-9: 熵控制器 (保证达到目标复杂度)
        entropy_controller = EntropyController(
            target_complexity=alpha,
            complexity_tolerance=0.05,  # 严格容差
            min_steps=max(5, num_inference_steps // 4)
        )

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0

            t_prev_tensor = torch.full(
                (batch_size,), t_prev, device=device, dtype=torch.long
            )

            # 论文 Eq.4-6: 双协方差混合采样
            x = self.ddim_sample_step(x, t_tensor, t_prev_tensor, condition, alpha, eta)

            # 论文 Eq.2: 边界约束
            x = self._enforce_boundary_inpainting(x, condition, t, self.timesteps)

            # 论文 Eq.8-9: 检查是否达到目标复杂度 (保证停止)
            if i >= entropy_controller.min_steps:
                should_stop, current_complexity = entropy_controller.should_stop(
                    x, i, len(timesteps)
                )
                if should_stop:
                    break

        # 最终边界精确修正
        x = self._apply_boundary_conditions(x, condition)

        return x

    def _initialize_with_condition(
        self,
        batch_size: int,
        condition: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """
        按论文 Eq.(2) 初始化: X_R = M ⊙ X_c + (1-M) ⊙ ε

        论文中的掩码初始化:
        - M: 掩码矩阵，边界点为1，中间点为0
        - X_c: 条件点 (起点和终点)
        - ε: 高斯噪声

        最终结果: 边界点固定为条件值，中间点为纯噪声
        """
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]    # (batch, 2)

        # 创建掩码 M (论文 Eq.2)
        # M[0] = 1 (起点), M[-1] = 1 (终点), 其他 = 0
        mask = torch.zeros(batch_size, self.seq_length, 1, device=device)
        mask[:, 0, :] = 1.0   # 起点掩码
        mask[:, -1, :] = 1.0  # 终点掩码

        # 创建条件值 X_c
        # 只有起点和终点有值，其他位置为0 (会被噪声覆盖)
        x_c = torch.zeros(batch_size, self.seq_length, self.input_dim, device=device)
        x_c[:, 0, :] = start_point   # 起点
        x_c[:, -1, :] = end_point    # 终点

        # 生成高斯噪声 ε
        noise = torch.randn(batch_size, self.seq_length, self.input_dim, device=device)

        # 论文 Eq.(2): X_R = M ⊙ X_c + (1-M) ⊙ ε
        x = mask * x_c + (1 - mask) * noise

        return x

    def _enforce_boundary_inpainting(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        t: int,
        total_timesteps: int,
    ) -> torch.Tensor:
        """
        论文 Eq.(2) 风格的边界约束 (Inpainting)

        每步采样后强制边界点为条件值:
        x' = M ⊙ X_c + (1-M) ⊙ x

        M: 掩码 (边界点=1)
        X_c: 条件值 (起点/终点)
        """
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]    # (batch, 2)

        # 创建掩码 M
        mask = torch.zeros_like(x)
        mask[:, 0, :] = 1.0   # 起点
        mask[:, -1, :] = 1.0  # 终点

        # 创建条件值 X_c
        x_c = torch.zeros_like(x)
        x_c[:, 0, :] = start_point
        x_c[:, -1, :] = end_point

        # 应用掩码: x' = M ⊙ X_c + (1-M) ⊙ x
        x = mask * x_c + (1 - mask) * x

        return x

    @torch.no_grad()
    def sample_with_complexity_control(
        self,
        batch_size: int,
        condition: torch.Tensor,
        target_complexity: float,
        max_attempts: int = 3,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, dict]:
        """
        使用复杂度控制生成轨迹 (带重试机制)

        如果第一次生成的复杂度不满足目标，会重试

        Args:
            batch_size: 批次大小
            condition: 条件张量
            target_complexity: 目标复杂度 (论文推荐 0.3-0.8)
            max_attempts: 最大尝试次数
            device: 设备

        Returns:
            (trajectories, info_dict)
        """
        # 约束到论文推荐范围
        target_complexity = max(0.3, min(0.8, target_complexity))

        entropy_controller = EntropyController(
            target_complexity=target_complexity,
            complexity_tolerance=0.1,
        )

        best_trajectories = None
        best_complexity_diff = float('inf')
        info = {'attempts': 0, 'final_complexity': 0}

        for attempt in range(max_attempts):
            # α 直接作为目标复杂度 (论文设计)
            # 每次尝试略微调整
            alpha = target_complexity * (1 + 0.05 * (attempt - 1))
            alpha = max(0.3, min(0.8, alpha))

            # 使用新的 sample 函数 (内置熵控制)
            trajectories = self.sample(
                batch_size=batch_size,
                condition=condition,
                alpha=alpha,
                device=device,
            )

            # 计算实际复杂度
            complexity = entropy_controller.compute_complexity(trajectories)
            mean_complexity = complexity.mean().item()
            complexity_diff = abs(mean_complexity - target_complexity)

            info['attempts'] = attempt + 1
            info['final_complexity'] = mean_complexity

            if complexity_diff < best_complexity_diff:
                best_complexity_diff = complexity_diff
                best_trajectories = trajectories

            # 检查是否满足目标
            if complexity_diff <= entropy_controller.complexity_tolerance:
                break

        return best_trajectories, info

    def _apply_boundary_conditions(
        self,
        trajectory: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        应用边界条件：确保轨迹的起点和终点与条件匹配
        使用平滑插值避免突变
        """
        batch_size = trajectory.shape[0]
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]  # (batch, 2)

        # 创建平滑权重
        weights = torch.linspace(0, 1, self.seq_length, device=trajectory.device)
        weights = weights.view(1, -1, 1).expand(batch_size, -1, 2)

        # 计算当前起点和终点的偏移
        current_start = trajectory[:, 0:1, :]
        current_end = trajectory[:, -1:, :]

        # 插值修正
        start_correction = start_point.unsqueeze(1) - current_start
        end_correction = end_point.unsqueeze(1) - current_end

        # 应用平滑修正
        correction = start_correction * (1 - weights) + end_correction * weights
        trajectory = trajectory + correction

        return trajectory

    def get_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor = None,
        alpha: torch.Tensor = None,  # 复杂度参数
    ) -> dict:
        """
        计算训练损失

        Args:
            x_0: 原始轨迹 (batch, seq_len, 2)
            condition: 条件 (batch, 4) - 起点+终点
            t: 时间步 (batch,)
            alpha: 复杂度参数 (batch,) - 从轨迹计算得到

        Returns:
            包含各项损失的字典
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        # 如果没有提供alpha，从轨迹计算
        if alpha is None:
            alpha = self.compute_trajectory_alpha(x_0)

        # 前向扩散
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # 预测噪声 - 论文Eq.10: ε_θ(x_t, t, c, α)
        predicted_noise = self.model(x_t, t, condition, alpha)

        # DDIM损失（MSE）
        ddim_loss = F.mse_loss(predicted_noise, noise)

        return {
            'ddim_loss': ddim_loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'alpha': alpha,
        }

    def compute_trajectory_alpha(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        从轨迹计算复杂度参数α

        基于路径长度比率 (论文Eq.13中的复杂度定义)
        α = (path_length / straight_distance - 1) / max_ratio

        Args:
            trajectory: (batch, seq_len, 2)

        Returns:
            alpha: (batch,) 范围 [0, 1]
        """
        # 计算路径长度
        segments = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        path_length = torch.norm(segments, dim=-1).sum(dim=-1)  # (batch,)

        # 计算直线距离
        straight_dist = torch.norm(
            trajectory[:, -1, :] - trajectory[:, 0, :], dim=-1
        ) + 1e-8  # (batch,)

        # 路径长度比率
        ratio = path_length / straight_dist

        # 归一化到 [0, 1]
        # ratio=1 表示直线 -> alpha=0
        # ratio=3 表示复杂曲线 -> alpha=1
        alpha = torch.clamp((ratio - 1.0) / 2.0, 0.0, 1.0)

        return alpha

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: tuple
    ) -> torch.Tensor:
        """从一维张量中提取值，并调整形状以便广播"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EntropyController:
    """
    熵控制器 (论文 Eq.8-9)

    论文使用 MST (最小生成树) 长度近似轨迹熵:
    H(X) ≈ (d/m) * Σ log(m * L_i) + log(c_d) + C_E

    简化实现: 使用 MST 比率 (路径长度/直线距离) 作为复杂度度量
    - ratio ≈ 1: 接近直线，低复杂度
    - ratio > 1: 曲线越复杂，比率越大

    当生成的轨迹达到目标复杂度时，提前停止采样。
    """

    def __init__(
        self,
        target_complexity: float = 0.5,
        complexity_tolerance: float = 0.1,
        min_steps: int = 10,
    ):
        """
        Args:
            target_complexity: 目标复杂度 (0-1)
            complexity_tolerance: 复杂度容差
            min_steps: 最小采样步数
        """
        self.target_complexity = target_complexity
        self.complexity_tolerance = complexity_tolerance
        self.min_steps = min_steps

    def compute_mst_complexity(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算基于 MST 的复杂度 (论文 Eq.8-9)

        使用路径长度比率近似 MST 熵:
        complexity = (path_length / straight_distance - 1) / 2

        Returns:
            complexity: (batch,) 范围 [0, 1]
        """
        # trajectory: (batch, seq_len, 2)

        # 计算路径长度 (相邻点距离之和，近似 MST)
        segments = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        path_length = torch.norm(segments, dim=-1).sum(dim=-1)  # (batch,)

        # 计算起点到终点的直线距离
        straight_distance = torch.norm(
            trajectory[:, -1, :] - trajectory[:, 0, :],
            dim=-1
        ) + 1e-8  # (batch,)

        # MST 比率
        mst_ratio = path_length / straight_distance

        # 归一化到 [0, 1]
        # ratio=1 (直线) -> complexity=0
        # ratio=3 (复杂曲线) -> complexity=1
        complexity = torch.clamp((mst_ratio - 1.0) / 2.0, 0.0, 1.0)

        return complexity

    def compute_complexity(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹复杂度 (仅使用 MST，符合论文 Eq.8-9)
        """
        return self.compute_mst_complexity(trajectory)

    def should_stop(
        self,
        trajectory: torch.Tensor,
        current_step: int,
        total_steps: int,
    ) -> Tuple[bool, torch.Tensor]:
        """
        判断是否应该提前停止采样

        Returns:
            (should_stop, complexity)
        """
        if current_step < self.min_steps:
            return False, torch.zeros(trajectory.shape[0], device=trajectory.device)

        complexity = self.compute_complexity(trajectory)
        mean_complexity = complexity.mean().item()

        # 检查是否达到目标复杂度
        lower_bound = self.target_complexity - self.complexity_tolerance
        upper_bound = self.target_complexity + self.complexity_tolerance

        if lower_bound <= mean_complexity <= upper_bound:
            return True, complexity

        return False, complexity

    def get_alpha_from_complexity(self, target_complexity: float) -> float:
        """
        根据目标复杂度获取alpha值
        """
        # 非线性映射：低复杂度需要低alpha，高复杂度需要高alpha
        return 0.1 + 0.9 * (target_complexity ** 0.5)


def create_alpha_ddim(
    seq_length: int = 500,
    timesteps: int = 1000,
    base_channels: int = 64,
    device: str = "cuda",
) -> AlphaDDIM:
    """创建α-DDIM模型"""
    unet = TrajectoryUNet(
        seq_length=seq_length,
        input_dim=2,
        base_channels=base_channels,
    )

    model = AlphaDDIM(
        model=unet,
        timesteps=timesteps,
        seq_length=seq_length,
        input_dim=2,
    )

    return model.to(device)


if __name__ == "__main__":
    # 测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = create_alpha_ddim(seq_length=500, device=device)

    # 测试前向传播
    batch_size = 4
    x_0 = torch.randn(batch_size, 500, 2, device=device)
    condition = torch.randn(batch_size, 4, device=device)

    losses = model.get_loss(x_0, condition)
    print(f"DDIM Loss: {losses['ddim_loss'].item():.4f}")

    # 测试采样
    print("\nTesting sampling...")
    samples = model.sample(
        batch_size=2,
        condition=condition[:2],
        num_inference_steps=20,
        alpha=0.5,
        device=device
    )
    print(f"Generated trajectory shape: {samples.shape}")
