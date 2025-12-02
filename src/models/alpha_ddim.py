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
        input_dim: int = 3,  # x, y, dt
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
        alpha: float = 1.5,  # 论文方案A: path_ratio ∈ [1, +∞)
        eta: float = 0.0,  # DDIM随机性参数
    ) -> torch.Tensor:
        """
        α-DDIM单步采样 (论文 Eq.4-6)

        核心创新: 双协方差混合
        Σ = (1-α')·Σ_d + α'·Σ_n

        其中 α' = 1 - 1/α 将 Paper A 的 α ∈ [1, +∞) 转换为 [0, 1):
        - α = 1 (直线) → α' = 0 → 100% 方向噪声
        - α → ∞ (复杂) → α' → 1 → 100% 各向同性噪声

        论文 Eq.4: Σ_d = (k_c·||d||)² · (d̂ ⊗ d̂)  -- 方向协方差 (秩1矩阵)
        论文 Eq.5: z_d = σ_d · ε · d̂ 其中 σ_d = k_c·||d||  -- 方向噪声采样
        论文 Eq.6: Σ = (1-α')·Σ_d + α'·Σ_n    -- 混合协方差 (Σ_n = σ²_n·I 各向同性)

        采样: z ~ N(0, Σ) ≈ √(1-α')·z_d + √α'·z_n
        其中 z_d ~ N(0, Σ_d), z_n ~ N(0, Σ_n)

        Args:
            x_t: 当前时刻的噪声数据 (batch, seq_len, input_dim)
            t: 当前时间步
            t_prev: 上一个时间步
            condition: 条件（起点+终点）(batch, 4)
            alpha: Paper A path_ratio (α ≥ 1, 越大越复杂)
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
        direction_unit = direction / direction_norm  # 单位方向向量 d̂

        # 论文 Eq.4: k_c 是常数缩放因子 (论文推荐值 ≈ 1/6)
        # 注意: k_c 应该是常数，而非距离的函数
        kc = 1.0 / 6.0

        # DDIM 噪声方差 (标准 DDIM 公式)
        sigma_squared = eta ** 2 * (
            (1 - alpha_t_prev) / (1 - alpha_t + 1e-8) * (1 - alpha_t / (alpha_t_prev + 1e-8))
        )
        sigma_base = torch.sqrt(torch.clamp(sigma_squared, min=0))

        # ========== 从混合协方差采样 ==========
        # 论文 Eq.4: z_d ~ N(0, Σ_d) 其中 Σ_d = (k_c·||d||)²·(d̂⊗d̂)
        # 方向协方差是秩1矩阵，标准差为 σ_d = k_c·||d||
        # 采样为: z_d = σ_d · ε_1 · d̂，其中 ε_1 ~ N(0, 1) 是标量
        # 注意: 方向噪声仅适用于 x, y 维度，dt 维度使用各向同性噪声
        noise_scalar = torch.randn(batch_size, self.seq_length, 1, device=x_t.device)
        direction_unit_expanded = direction_unit.unsqueeze(1)  # (batch, 1, 2)
        # 标准差 σ_d = k_c · ||d|| (从协方差 Σ_d = σ_d² · (d̂⊗d̂) 得到)
        sigma_d = (kc * direction_norm).unsqueeze(1)  # (batch, 1, 1)
        z_d_xy = sigma_d * noise_scalar * direction_unit_expanded  # (batch, seq, 2) 仅 x, y

        # 各向同性噪声 (x, y 维度): z_n_xy ~ N(0, Σ_n) 其中 Σ_n = (k_c·||d||)²·I
        # 标准差 σ_n = k_c·||d|| (与方向噪声使用相同的缩放因子)
        sigma_epsilon = (kc * direction_norm).unsqueeze(1)  # (batch, 1, 1)
        z_n_xy = sigma_epsilon * torch.randn(batch_size, self.seq_length, 2, device=x_t.device)

        # 各向同性噪声 (dt 维度): 使用与 x,y 相同的缩放因子，保持尺度一致
        # dt 已归一化到 [0,1]，使用相同的 σ_epsilon 避免尺度不匹配
        z_n_dt = sigma_epsilon * torch.randn(batch_size, self.seq_length, self.input_dim - 2, device=x_t.device) if self.input_dim > 2 else None

        # 论文 Eq.6: 混合协方差采样
        # 将 α ∈ [1, +∞) 转换为混合系数 a ∈ [0, 1)
        # a = 1 - 1/α: α=1 → a=0 (全方向), α→∞ → a→1 (全各向同性)
        # 注意: 这与 StyleEmb 的 α'=1/(α+1) 不同，两者用途不同
        mixing_coef = 1.0 - 1.0 / max(alpha, 1.0)
        mixing_coef = min(mixing_coef, 0.99)  # 防止完全各向同性

        # z ~ N(0, (1-a)Σ_d + a·Σ_ε) ≈ √(1-a)·z_d + √a·z_n
        sqrt_dir = np.sqrt(1 - mixing_coef)    # 方向噪声权重
        sqrt_iso = np.sqrt(mixing_coef)        # 各向同性噪声权重

        # 组合噪声: x, y 维度使用混合噪声，dt 维度仅使用各向同性噪声
        z_mixed = torch.zeros_like(x_t)  # (batch, seq, input_dim)
        z_mixed[:, :, :2] = sqrt_dir * z_d_xy + sqrt_iso * z_n_xy  # x, y: 混合噪声 (距离缩放)
        if self.input_dim > 2 and z_n_dt is not None:
            z_mixed[:, :, 2:] = z_n_dt  # dt: 标准高斯噪声 (σ=1，无距离缩放)

        # 应用总噪声: σ_t 来自 DDIM，z_mixed 来自混合协方差
        sigma_t = sigma_base

        # DDIM更新公式: x_{t-1} = √α_{t-1}·x̂_0 + √(1-α_{t-1}-σ²_t)·ε_θ + σ_t·z
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_prev - sigma_t ** 2, min=0)) * predicted_noise
        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * z_mixed

        return x_t_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: torch.Tensor,
        num_inference_steps: int = 50,
        alpha: float = 1.5,
        eta: float = 0.5,  # 默认启用随机性以支持双协方差混合
        device: str = "cuda",
        effective_length=None,  # 有效轨迹长度 m，支持 int 或 Tensor (batch,)
        use_entropy_stopping: bool = True,  # 是否使用熵控制早停
        auto_length: bool = False,  # 是否自动预测轨迹长度
    ) -> torch.Tensor:
        """
        α-DDIM 采样 (论文 Eq.1-9)

        核心机制:
        1. 论文 Eq.1-3: 初始化 X_R，支持可控轨迹长度 m 和零填充
        2. 论文 Eq.4-6: 使用 α 控制双协方差混合
        3. 论文 Eq.8-9: 基于 MST 熵的早停机制

        Args:
            batch_size: 批次大小
            condition: 条件张量 (batch, 4) - [start_x, start_y, end_x, end_y]
            num_inference_steps: 最大推理步数
            alpha: 论文方案A path_ratio (α ≥ 1, α=1 直线, α→∞ 复杂)
            eta: DDIM随机性参数
            device: 设备
            effective_length: 有效轨迹长度，支持:
                - None: 使用 seq_length
                - int: 所有样本使用相同长度
                - Tensor (batch,): 每个样本使用不同长度 (per-sample)
            use_entropy_stopping: 是否使用 MST 熵控制早停
            auto_length: 是否自动预测轨迹长度 (需要模型启用 enable_length_prediction)
                        如果为 True 且 effective_length 为 None，则自动预测长度

        Returns:
            生成的轨迹 (batch, seq_length, input_dim)
            如果 effective_length < seq_length，后面的点为零填充
        """
        self.model.eval()

        # 论文方案A: α = path_ratio ∈ [1, +∞)
        # α = 1: 直线轨迹（最简单）
        # α → ∞: 复杂曲线轨迹
        alpha_clamped = max(1.0, alpha)  # 确保 α >= 1

        # 自动长度预测 (Shared Encoder 模式)
        # 如果 auto_length=True 且 effective_length 未指定，则使用模型预测长度
        predicted_lengths = None
        if auto_length and effective_length is None:
            if hasattr(self.model, 'length_head') and self.model.length_head is not None:
                # 构建 alpha tensor
                alpha_tensor = torch.full((batch_size,), alpha_clamped, device=device, dtype=torch.float32)
                # 创建初始输入 (纯高斯噪声，对应 t=T) 用于 Shared Encoder 长度预测
                x_init = torch.randn(batch_size, self.seq_length, self.input_dim, device=device)
                # 预测长度
                log_length = self.model.predict_length(x_init, condition, alpha_tensor)
                predicted_lengths = self.model.decode_length(log_length, max_length=self.seq_length)
                # 使用 per-sample 长度
                effective_length = predicted_lengths
            else:
                # 模型不支持长度预测，回退到使用全长
                pass

        # 处理 effective_length 的不同类型
        is_per_sample = isinstance(effective_length, torch.Tensor)

        # 统一处理长度：确保最小长度为 2，避免数值不稳定
        if is_per_sample:
            effective_length = effective_length.long().to(device).clamp(min=2)
        elif effective_length is not None:
            effective_length = max(2, effective_length)

        # 论文 Eq.2-3: 掩码初始化，支持距离缩放噪声和可控长度
        x = self._initialize_with_condition(
            batch_size, condition, device, effective_length
        )

        # 设置采样时间步
        step_size = max(1, self.timesteps // num_inference_steps)
        timesteps = list(range(0, self.timesteps, step_size))[::-1]

        # 论文 Eq.8-9: 熵控制器 (使用完整 MST 熵公式)
        # 将 Paper A 的 α ∈ [1, +∞) 转换为 complexity ∈ [0, 1)
        # complexity = 1 - 1/α: α=1 → 0, α=2 → 0.5, α→∞ → 1
        target_complexity = 1.0 - 1.0 / alpha_clamped
        entropy_controller = EntropyController(
            target_complexity=target_complexity,
            complexity_tolerance=0.05,  # 保持原值，防止 loss 爆炸
            min_steps=max(5, num_inference_steps // 4),
            use_full_entropy=True,  # 使用完整 MST 熵公式
        )

        # 预计算 per-sample padding mask (如果是 per-sample 模式)
        if is_per_sample:
            # effective_length 已经在上面被转换为 long tensor 并 clamp 过
            seq_indices = torch.arange(self.seq_length, device=device).unsqueeze(0)  # (1, seq_length)
            per_sample_padding_mask = (seq_indices < effective_length.unsqueeze(1)).float().unsqueeze(2)  # (batch, seq_length, 1)

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
            x = self.ddim_sample_step(
                x, t_tensor, t_prev_tensor, condition, alpha_clamped, eta
            )

            # 论文 Eq.2: 边界约束 (inpainting)
            x = self._enforce_boundary_inpainting(x, condition, t, self.timesteps, effective_length)

            # 保持零填充
            if is_per_sample:
                # Per-sample padding
                x = x * per_sample_padding_mask
            elif effective_length is not None and effective_length < self.seq_length:
                # 统一 int padding
                padding_mask = torch.zeros(batch_size, self.seq_length, 1, device=device)
                padding_mask[:, :effective_length, :] = 1.0
                x = x * padding_mask

            # 论文 Eq.8-9: 检查是否达到目标复杂度
            if use_entropy_stopping and i >= entropy_controller.min_steps:
                # 只检查有效部分的复杂度 (per-sample 时使用最小长度)
                if is_per_sample:
                    min_length = int(effective_length.min().item())
                    x_valid = x[:, :min_length, :]
                elif effective_length is not None and effective_length < self.seq_length:
                    x_valid = x[:, :effective_length, :]
                else:
                    x_valid = x

                should_stop, current_complexity = entropy_controller.should_stop(
                    x_valid, i, len(timesteps)
                )
                if should_stop:
                    break

        # 最终边界精确修正
        x = self._apply_boundary_conditions(x, condition, effective_length)

        return x

    def _initialize_with_condition(
        self,
        batch_size: int,
        condition: torch.Tensor,
        device: str,
        effective_length=None,  # 有效轨迹长度 m，支持 int 或 Tensor (batch,)
    ) -> torch.Tensor:
        """
        按论文 Eq.(2-3) 初始化:
        X_R = {p_0} ∥ {p_c + ε_i}^m ∥ {p_m} ∥ {0}^(N-m)

        论文 Eq.2: ε ~ N(0, Σ_ε)，其中 Σ_ε = (k_c · ||p_m - p_0||)²
        论文 Eq.3: 中间点初始化为 p_c + ε，其中 p_c 是起终点的中点

        k_c = 1/6 (使99%的点在屏幕内)

        Args:
            effective_length: 有效轨迹长度，支持:
                - None: 使用 seq_length
                - int: 所有样本使用相同长度
                - Tensor (batch,): 每个样本使用不同长度 (per-sample)
        """
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]    # (batch, 2)

        # 计算起终点距离 (论文 Eq.2)
        distance = torch.norm(end_point - start_point, dim=-1, keepdim=True)  # (batch, 1)

        # 计算起终点中点 (论文 Eq.3)
        midpoint = (start_point + end_point) / 2  # (batch, 2)

        # 处理 effective_length 的不同类型
        is_per_sample = isinstance(effective_length, torch.Tensor)

        if is_per_sample:
            # Per-sample 长度: Tensor (batch,)
            lengths = effective_length.long().to(device)
        elif effective_length is not None:
            lengths = None
        else:
            lengths = None

        # 论文 Eq.2: σ_ε = k_c · ||d||，其中 k_c = 1/6
        # 不除以 √(m-1)，按论文原始公式
        k_c = 1.0 / 6.0
        sigma_epsilon = k_c * distance  # (batch, 1)
        sigma_epsilon = sigma_epsilon.unsqueeze(2)  # (batch, 1, 1) for broadcasting

        # 创建掩码 M (论文 Eq.2)
        # x, y 维度: 起点和终点
        # dt 维度: 仅起点 (dt[0] = 0)
        mask_xy = torch.zeros(batch_size, self.seq_length, 2, device=device)
        mask_xy[:, 0, :] = 1.0   # 起点掩码 (x, y)

        mask_dt = torch.zeros(batch_size, self.seq_length, self.input_dim - 2, device=device) if self.input_dim > 2 else None
        if mask_dt is not None:
            mask_dt[:, 0, :] = 1.0  # 起点掩码 (dt[0] = 0)

        # 创建条件值 X_c
        x_c = torch.zeros(batch_size, self.seq_length, self.input_dim, device=device)
        x_c[:, 0, :2] = start_point   # 起点 (x, y)
        # dt[0] = 0 已经是默认值，无需显式设置

        if is_per_sample:
            # Per-sample: 使用 scatter_ 设置每个样本的终点位置
            end_indices = (lengths - 1).clamp(0, self.seq_length - 1)  # (batch,)
            # 为 mask_xy 设置终点
            end_indices_xy = end_indices.view(batch_size, 1, 1).expand(-1, 1, 2)  # (batch, 1, 2)
            mask_xy.scatter_(1, end_indices_xy, 1.0)
            # 为 x_c 设置终点 (仅 x, y 维度)
            x_c[:, :, :2].scatter_(1, end_indices_xy, end_point.unsqueeze(1))
        elif effective_length is not None and effective_length < self.seq_length:
            # 统一 int 长度
            end_idx = effective_length - 1
            mask_xy[:, end_idx, :] = 1.0
            x_c[:, end_idx, :2] = end_point  # 仅 x, y
        else:
            # 默认: 终点在最后
            mask_xy[:, -1, :] = 1.0
            x_c[:, -1, :2] = end_point  # 仅 x, y

        # 生成距离缩放的高斯噪声 ε (全维度)
        noise = torch.randn(batch_size, self.seq_length, self.input_dim, device=device)
        scaled_noise = noise * sigma_epsilon  # (batch, seq_len, input_dim)

        # 论文 Eq.3: 中间点初始化为 p_c + ε (中点 + 噪声)
        # x, y 维度: 中点 + 噪声; dt 维度: 纯噪声
        middle_points = torch.zeros(batch_size, self.seq_length, self.input_dim, device=device)
        midpoint_expanded = midpoint.unsqueeze(1).expand(-1, self.seq_length, -1)  # (batch, seq_len, 2)
        middle_points[:, :, :2] = midpoint_expanded + scaled_noise[:, :, :2]  # x, y: 中点 + 噪声
        if self.input_dim > 2:
            middle_points[:, :, 2:] = scaled_noise[:, :, 2:]  # dt: 纯噪声 (将被后续处理)

        # 创建 padding mask (有效部分为1，padding部分为0)
        if is_per_sample:
            # Per-sample padding mask
            seq_indices = torch.arange(self.seq_length, device=device).unsqueeze(0)  # (1, seq_length)
            padding_mask = (seq_indices < lengths.unsqueeze(1)).float().unsqueeze(2)  # (batch, seq_length, 1)
            middle_points = middle_points * padding_mask
        elif effective_length is not None and effective_length < self.seq_length:
            padding_mask = torch.zeros(batch_size, self.seq_length, 1, device=device)
            padding_mask[:, :effective_length, :] = 1.0
            middle_points = middle_points * padding_mask

        # 合并 mask: x, y 和 dt 分别处理
        # X_R = {p_0} ∥ {p_c + ε} ∥ {p_m} ∥ {0}
        x = torch.zeros(batch_size, self.seq_length, self.input_dim, device=device)
        # x, y 维度: 使用 mask_xy
        x[:, :, :2] = mask_xy * x_c[:, :, :2] + (1 - mask_xy) * middle_points[:, :, :2]
        # dt 维度: 使用 mask_dt (仅起点 dt[0]=0)
        if self.input_dim > 2 and mask_dt is not None:
            x[:, :, 2:] = mask_dt * x_c[:, :, 2:] + (1 - mask_dt) * middle_points[:, :, 2:]

        return x

    def _enforce_boundary_inpainting(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        t: int,
        total_timesteps: int,
        effective_length=None,  # 支持 int 或 Tensor (batch,)
    ) -> torch.Tensor:
        """
        论文 Eq.(2) 风格的边界约束 (Inpainting)

        每步采样后强制边界点为条件值:
        x' = M ⊙ X_c + (1-M) ⊙ x

        M: 掩码 (边界点=1)
        X_c: 条件值 (起点/终点)
        effective_length: 有效轨迹长度，支持:
            - None: 终点在 seq_length-1
            - int: 所有样本终点在 effective_length-1
            - Tensor (batch,): 每个样本终点位置不同 (per-sample)

        边界约束:
        - x, y: 起点和终点
        - dt: 仅起点 (dt[0] = 0)
        """
        batch_size = x.shape[0]
        device = x.device
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]    # (batch, 2)

        # 处理 effective_length 的不同类型
        is_per_sample = isinstance(effective_length, torch.Tensor)

        # 创建掩码 M
        # x, y 维度: 起点和终点
        # dt 维度: 仅起点 (dt[0] = 0)
        mask = torch.zeros_like(x)
        mask[:, 0, :2] = 1.0  # 起点 (x, y)
        if self.input_dim > 2:
            mask[:, 0, 2:] = 1.0  # 起点 (dt[0] = 0)

        # 创建条件值 X_c
        x_c = torch.zeros_like(x)  # 从零开始构建条件值
        x_c[:, 0, :2] = start_point  # 起点 (x, y)
        if self.input_dim > 2:
            x_c[:, 0, 2:] = 0.0  # dt[0] = 0

        if is_per_sample:
            # Per-sample: 使用 scatter_ 设置每个样本的终点
            lengths = effective_length.long().to(device)
            end_indices = (lengths - 1).clamp(0, self.seq_length - 1)  # (batch,)
            # 为 mask 设置终点 (仅 x, y 维度)
            end_indices_mask = end_indices.view(batch_size, 1, 1).expand(-1, 1, 2)
            mask[:, :, :2].scatter_(1, end_indices_mask, 1.0)
            # 为 x_c 设置终点 (仅 x, y 维度)
            x_c[:, :, :2].scatter_(1, end_indices_mask, end_point.unsqueeze(1))
        elif effective_length is not None and effective_length < self.seq_length:
            end_idx = effective_length - 1
            mask[:, end_idx, :2] = 1.0  # 仅 x, y
            x_c[:, end_idx, :2] = end_point  # 仅 x, y
        else:
            end_idx = self.seq_length - 1
            mask[:, end_idx, :2] = 1.0  # 仅 x, y
            x_c[:, end_idx, :2] = end_point  # 仅 x, y

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
            target_complexity: 目标复杂度 ∈ [0, 1)
                - 0: 直线（最简单）
                - → 1: 复杂曲线
            max_attempts: 最大尝试次数
            device: 设备

        Returns:
            (trajectories, info_dict)
        """
        # 约束复杂度到 [0, 0.9]，避免极端值
        target_complexity = max(0.0, min(0.9, target_complexity))

        entropy_controller = EntropyController(
            target_complexity=target_complexity,
            complexity_tolerance=0.1,
        )

        best_trajectories = None
        best_complexity_diff = float('inf')
        info = {'attempts': 0, 'final_complexity': 0}

        for attempt in range(max_attempts):
            # 将 complexity ∈ [0, 1) 转换为 α = path_ratio ∈ [1, +∞)
            # complexity = 1 - 1/α → α = 1 / (1 - complexity)
            # complexity=0 → α=1, complexity=0.5 → α=2, complexity→1 → α→∞
            adjusted_complexity = target_complexity * (1 + 0.05 * (attempt - 1))
            adjusted_complexity = max(0.0, min(0.95, adjusted_complexity))
            alpha = 1.0 / (1.0 - adjusted_complexity + 1e-8)
            alpha = max(1.0, min(10.0, alpha))  # 限制到合理范围

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

    @torch.no_grad()
    def sample_with_auto_length(
        self,
        batch_size: int,
        condition: torch.Tensor,
        alpha: float = 1.5,
        num_inference_steps: int = 50,
        eta: float = 0.5,
        device: str = "cuda",
        use_entropy_stopping: bool = True,
        use_per_sample_length: bool = True,  # 是否使用 per-sample 长度
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用自动长度预测生成轨迹

        基于条件和α预测轨迹长度 m，然后生成相应长度的轨迹。

        Args:
            batch_size: 批次大小
            condition: 条件张量 (batch, 4) - [start_x, start_y, end_x, end_y]
            alpha: path_ratio (α ≥ 1, α=1 直线, α→∞ 复杂)
            num_inference_steps: 推理步数
            eta: DDIM随机性参数
            device: 设备
            use_entropy_stopping: 是否使用熵控制早停
            use_per_sample_length: 是否使用 per-sample 长度 (默认 True)
                - True: 每个样本使用模型预测的独立长度
                - False: 使用批次平均长度 (向后兼容)

        Returns:
            (trajectories, predicted_lengths)
            - trajectories: (batch, seq_length, input_dim) 生成的轨迹
            - predicted_lengths: (batch,) 每个样本的预测长度
        """
        # 检查模型是否支持长度预测
        if not hasattr(self.model, 'length_head') or self.model.length_head is None:
            raise RuntimeError("Model does not support length prediction. Enable enable_length_prediction=True.")

        self.model.eval()

        # 约束 alpha (path_ratio 语义: α ≥ 1)
        alpha_clamped = max(1.0, alpha)

        # 预测长度 (Shared Encoder 模式)
        alpha_tensor = torch.full((batch_size,), alpha_clamped, device=device, dtype=torch.float32)
        # 创建初始输入 (t=0 时刻的噪声轨迹) 用于 Shared Encoder 长度预测
        x_init = torch.randn(batch_size, self.seq_length, self.input_dim, device=device)
        log_length = self.model.predict_length(x_init, condition, alpha_tensor)
        predicted_lengths = self.model.decode_length(log_length, max_length=self.seq_length)

        # 选择使用 per-sample 长度还是平均长度
        if use_per_sample_length:
            # Per-sample: 每个样本使用独立的预测长度
            effective_length = predicted_lengths  # Tensor (batch,)
        else:
            # 向后兼容: 使用平均长度
            effective_length = int(predicted_lengths.float().mean().item())

        # 生成轨迹
        trajectories = self.sample(
            batch_size=batch_size,
            condition=condition,
            num_inference_steps=num_inference_steps,
            alpha=alpha,
            eta=eta,
            device=device,
            effective_length=effective_length,
            use_entropy_stopping=use_entropy_stopping,
            auto_length=False,  # 已经手动预测了长度
        )

        return trajectories, predicted_lengths

    def _apply_boundary_conditions(
        self,
        trajectory: torch.Tensor,
        condition: torch.Tensor,
        effective_length=None,  # 支持 int 或 Tensor (batch,)
    ) -> torch.Tensor:
        """
        应用边界条件：确保轨迹的起点和终点与条件匹配
        使用平滑插值避免突变 (仅对 x, y 维度)

        Args:
            trajectory: (batch, seq_len, input_dim)
            condition: (batch, 4) - [start_x, start_y, end_x, end_y]
            effective_length: 有效轨迹长度，支持:
                - None: 处理整个 seq_length
                - int: 所有样本使用相同长度
                - Tensor (batch,): 每个样本使用不同长度 (per-sample)
        """
        batch_size = trajectory.shape[0]
        device = trajectory.device
        start_point = condition[:, :2]  # (batch, 2)
        end_point = condition[:, 2:]  # (batch, 2)

        # 处理 effective_length 的不同类型
        is_per_sample = isinstance(effective_length, torch.Tensor)

        # 复制轨迹，仅修改 x, y 维度
        result = trajectory.clone()

        if is_per_sample:
            # Per-sample: 每个样本有不同的长度
            lengths = effective_length.long().to(device)  # (batch,)
            lengths_float = lengths.float().unsqueeze(1)  # (batch, 1)

            # 创建 per-sample 权重: weights[i, j] = j / (lengths[i] - 1)
            seq_indices = torch.arange(self.seq_length, device=device).float().unsqueeze(0)  # (1, seq_length)
            weights = seq_indices / (lengths_float - 1).clamp(min=1)  # (batch, seq_length)
            weights = weights.clamp(0, 1).unsqueeze(2)  # (batch, seq_length, 1)

            # 获取当前起点 (仅 x, y)
            current_start_xy = trajectory[:, 0:1, :2]  # (batch, 1, 2)

            # 获取每个样本的当前终点 (仅 x, y)
            end_indices = (lengths - 1).clamp(0, self.seq_length - 1)  # (batch,)
            end_indices_expanded = end_indices.view(batch_size, 1, 1).expand(-1, 1, 2)
            current_end_xy = trajectory[:, :, :2].gather(1, end_indices_expanded)  # (batch, 1, 2)

            # 插值修正 (仅 x, y)
            start_correction = start_point.unsqueeze(1) - current_start_xy  # (batch, 1, 2)
            end_correction = end_point.unsqueeze(1) - current_end_xy  # (batch, 1, 2)

            # 应用平滑修正 (只在有效区域)
            correction = start_correction * (1 - weights) + end_correction * weights  # (batch, seq_length, 2)

            # 创建 padding mask
            padding_mask = (seq_indices < lengths_float).float().unsqueeze(2)  # (batch, seq_length, 1)

            # 只对 x, y 维度应用修正
            result[:, :, :2] = trajectory[:, :, :2] + correction * padding_mask

            return result

        elif effective_length is not None:
            # 统一 int 长度
            m = effective_length

            # 创建平滑权重 (只对有效部分)
            weights = torch.linspace(0, 1, m, device=device)
            weights = weights.view(1, -1, 1).expand(batch_size, -1, 2)

            # 获取有效部分的 x, y
            traj_valid_xy = trajectory[:, :m, :2]

            # 计算当前起点和终点的偏移 (仅 x, y)
            current_start_xy = traj_valid_xy[:, 0:1, :]
            current_end_xy = traj_valid_xy[:, -1:, :]

            # 插值修正
            start_correction = start_point.unsqueeze(1) - current_start_xy
            end_correction = end_point.unsqueeze(1) - current_end_xy

            # 应用平滑修正
            correction = start_correction * (1 - weights) + end_correction * weights
            result[:, :m, :2] = traj_valid_xy + correction

            return result

        else:
            # 默认: 处理整个 seq_length
            m = self.seq_length

            # 创建平滑权重
            weights = torch.linspace(0, 1, m, device=device)
            weights = weights.view(1, -1, 1).expand(batch_size, -1, 2)

            # 计算当前起点和终点的偏移 (仅 x, y)
            current_start_xy = trajectory[:, 0:1, :2]
            current_end_xy = trajectory[:, -1:, :2]

            # 插值修正
            start_correction = start_point.unsqueeze(1) - current_start_xy
            end_correction = end_point.unsqueeze(1) - current_end_xy

            # 应用平滑修正 (仅 x, y)
            correction = start_correction * (1 - weights) + end_correction * weights
            result[:, :, :2] = trajectory[:, :, :2] + correction

            return result

    def get_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor = None,
        alpha: torch.Tensor = None,  # 复杂度参数
        mask: torch.Tensor = None,   # 有效位置掩码
        length: torch.Tensor = None,  # 轨迹真实长度 m (batch,) 或 (batch, 1)
        include_length_loss: bool = True,  # 是否计算长度预测损失
    ) -> dict:
        """
        计算训练损失

        Args:
            x_0: 原始轨迹 (batch, seq_len, input_dim)
            condition: 条件 (batch, 4) - 起点+终点
            t: 时间步 (batch,)
            alpha: 复杂度参数 (batch,) - 从轨迹计算得到
            mask: 有效位置掩码 (batch, seq_len) - 用于正确计算alpha
            length: 轨迹真实长度 m (batch,) 或 (batch, 1) - 用于长度预测损失
            include_length_loss: 是否计算长度预测损失 (需要模型启用 enable_length_prediction)

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
            alpha = self.compute_trajectory_alpha(x_0, mask)

        # 前向扩散
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # 预测噪声 - 论文Eq.10: ε_θ(x_t, t, c, α)
        predicted_noise = self.model(x_t, t, condition, alpha)

        # DDIM损失（MSE）
        ddim_loss = F.mse_loss(predicted_noise, noise)

        result = {
            'ddim_loss': ddim_loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'alpha': alpha,
        }

        # 长度预测损失 (Shared Encoder 模式)
        if include_length_loss and length is not None:
            if hasattr(self.model, 'length_head') and self.model.length_head is not None:
                # 预测 log(m+1)，使用 x_t 作为 encoder 输入
                predicted_log_length = self.model.predict_length(x_t, condition, alpha)
                result['predicted_log_length'] = predicted_log_length
                result['target_length'] = length
            else:
                # 模型不支持长度预测
                result['predicted_log_length'] = None
                result['target_length'] = None

        return result

    def compute_trajectory_alpha(
        self,
        trajectory: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        从轨迹计算复杂度参数α (论文方案A)

        α = path_length / straight_distance (ratio)
        注意: 仅使用 x, y 维度计算，忽略 dt

        论文语义:
        - α = 1: 直线轨迹（最简单）
        - α → ∞: 复杂曲线轨迹
        - 论文推荐范围: α ∈ [1, ~10]，实验用 [0.3, 0.8] 经 1/(α+1) 变换后

        Args:
            trajectory: (batch, seq_len, input_dim), 仅使用前两维 (x, y)
            mask: (batch, seq_len) 有效位置掩码，1表示有效，0表示padding

        Returns:
            alpha: (batch,) 范围 [1, +∞)，即 path_ratio
        """
        batch_size = trajectory.shape[0]
        device = trajectory.device

        # 仅使用 x, y 维度计算路径长度
        traj_xy = trajectory[:, :, :2]  # (batch, seq_len, 2)

        if mask is None:
            # 无mask时假设全部有效（向后兼容）
            segments = traj_xy[:, 1:, :] - traj_xy[:, :-1, :]
            path_length = torch.norm(segments, dim=-1).sum(dim=-1)
            straight_dist = torch.norm(
                traj_xy[:, -1, :] - traj_xy[:, 0, :], dim=-1
            ) + 1e-8
        else:
            # 使用mask计算真实的路径长度和终点
            # 计算每个样本的有效长度
            lengths = mask.sum(dim=-1).long()  # (batch,)

            # 计算段的mask: 只有当前点和下一点都有效时，这个段才有效
            # segment_mask[i] = mask[i] AND mask[i+1]
            segment_mask = mask[:, :-1] * mask[:, 1:]  # (batch, seq_len-1)

            # 计算所有段 (仅 x, y)
            segments = traj_xy[:, 1:, :] - traj_xy[:, :-1, :]  # (batch, seq_len-1, 2)
            segment_lengths = torch.norm(segments, dim=-1)  # (batch, seq_len-1)

            # 只累加有效段
            path_length = (segment_lengths * segment_mask).sum(dim=-1)  # (batch,)

            # 获取真实终点（使用gather按每个样本的有效长度取）
            # 终点索引是 lengths - 1
            end_indices = (lengths - 1).clamp(min=0)  # (batch,)
            end_indices = end_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)  # (batch, 1, 2)
            end_points = traj_xy.gather(1, end_indices).squeeze(1)  # (batch, 2)

            # 起点
            start_points = traj_xy[:, 0, :]  # (batch, 2)

            # 计算直线距离
            straight_dist = torch.norm(end_points - start_points, dim=-1) + 1e-8  # (batch,)

        # 路径长度比率 (论文中的 α)
        # α = path_length / straight_dist
        # α >= 1，α=1 表示直线
        alpha = path_length / straight_dist
        alpha = alpha.clamp(min=1.0)  # 确保 >= 1

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

    其中:
    - d: 维度 (2D轨迹为2)
    - m: 点数
    - L_i: MST边长 (使用相邻点距离近似)
    - c_d: 单位球体积 (2D为π)
    - C_E: 欧拉常数 ≈ 0.5772

    当生成的轨迹达到目标复杂度时，提前停止采样。
    """

    # 常量
    EULER_CONSTANT = 0.5772156649  # C_E
    C_2D = 3.14159265359  # π, 2D单位球体积

    def __init__(
        self,
        target_complexity: float = 0.5,
        complexity_tolerance: float = 0.1,
        min_steps: int = 10,
        use_full_entropy: bool = True,  # 是否使用完整熵公式
    ):
        """
        Args:
            target_complexity: 目标复杂度 (0-1)
            complexity_tolerance: 复杂度容差
            min_steps: 最小采样步数
            use_full_entropy: 是否使用完整的 MST 熵公式 (Eq.8-9)
        """
        self.target_complexity = target_complexity
        self.complexity_tolerance = complexity_tolerance
        self.min_steps = min_steps
        self.use_full_entropy = use_full_entropy

    def compute_mst_entropy(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算 MST 熵 (论文 Eq.8-9)

        论文推导:
        - Eq.8: L_MST ≃ Σ||p_i - p_{i+1}||_2 (路径总长度)
        - Eq.9: log(E(L_MST)) ≃ log(β) + ½·log(m) + ½·log(|Σ|)
                             = H({p_a}) + log(β) + ½·log(m) - log(2πe)

        因此: H ≈ log(L_path) - ½·log(m) + C
        其中 C = log(2πe) - log(β) 是常数

        Args:
            trajectory: (batch, seq_len, input_dim), 仅使用 x, y 计算

        Returns:
            entropy: (batch,) MST 熵估计值
        """
        batch_size, seq_len, d = trajectory.shape
        m = float(seq_len)

        # Eq.8: 计算路径总长度 L_path = Σ||p_i - p_{i+1}||
        # 只使用 x, y 维度计算路径长度
        traj_xy = trajectory[:, :, :2]
        segments = traj_xy[:, 1:, :] - traj_xy[:, :-1, :]
        path_length = torch.norm(segments, dim=-1).sum(dim=-1)  # (batch,)

        # 避免 log(0)
        path_length = path_length.clamp(min=1e-8)

        # Eq.9: H ≈ log(L_path) - ½·log(m) + log(2πe)
        # log(2πe) ≈ 2.837
        LOG_2PI_E = np.log(2 * np.pi * np.e)  # ≈ 2.837
        entropy = torch.log(path_length) - 0.5 * np.log(m) + LOG_2PI_E

        return entropy

    def compute_mst_ratio(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算 MST 比率 (简化复杂度度量)

        ratio = path_length / straight_distance
        注意: 只使用 x, y 维度计算

        Returns:
            ratio: (batch,) MST 比率
        """
        # 只使用 x, y 维度计算
        traj_xy = trajectory[:, :, :2]

        # 计算路径长度
        segments = traj_xy[:, 1:, :] - traj_xy[:, :-1, :]
        path_length = torch.norm(segments, dim=-1).sum(dim=-1)  # (batch,)

        # 计算起点到终点的直线距离
        straight_distance = torch.norm(
            traj_xy[:, -1, :] - traj_xy[:, 0, :],
            dim=-1
        ) + 1e-8  # (batch,)

        return path_length / straight_distance

    def compute_complexity(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹复杂度，归一化到 [0, 1]

        如果 use_full_entropy=True，使用完整 MST 熵公式
        否则使用简化的路径比率

        Returns:
            complexity: (batch,) 范围 [0, 1]
        """
        if self.use_full_entropy:
            # 使用完整熵公式
            entropy = self.compute_mst_entropy(trajectory)

            # 归一化熵到 [0, 1]
            # 经验值: 直线轨迹熵约 -2 到 0，复杂轨迹熵约 2 到 5
            # 使用 sigmoid 归一化
            complexity = torch.sigmoid((entropy - 1.0) / 2.0)
        else:
            # 使用简化的路径比率
            mst_ratio = self.compute_mst_ratio(trajectory)
            # 论文 Eq.8-9: β/(β+1) 公式
            # ratio=1 (直线) -> complexity=0
            # ratio→∞ -> complexity→1
            beta = mst_ratio - 1.0
            complexity = beta / (beta + 1.0 + 1e-8)
            complexity = torch.clamp(complexity, 0.0, 1.0)

        return complexity

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
        根据目标复杂度获取 alpha (path_ratio) 值

        将 complexity ∈ [0, 1) 转换为 α = path_ratio ∈ [1, +∞)
        complexity = 1 - 1/α → α = 1 / (1 - complexity)

        Args:
            target_complexity: 目标复杂度 ∈ [0, 1)
                - 0: 直线（最简单）
                - → 1: 复杂曲线

        Returns:
            alpha: path_ratio ≥ 1
        """
        # 限制复杂度范围
        target_complexity = max(0.0, min(0.95, target_complexity))
        # complexity = 1 - 1/α → α = 1 / (1 - complexity)
        alpha = 1.0 / (1.0 - target_complexity + 1e-8)
        return max(1.0, min(10.0, alpha))  # 限制到合理范围


def create_alpha_ddim(
    seq_length: int = 500,
    timesteps: int = 1000,
    base_channels: int = 96,  # 与 unet.py 保持一致
    input_dim: int = 3,  # x, y, dt
    device: str = "cuda",
) -> AlphaDDIM:
    """创建α-DDIM模型"""
    unet = TrajectoryUNet(
        seq_length=seq_length,
        input_dim=input_dim,
        base_channels=base_channels,
    )

    model = AlphaDDIM(
        model=unet,
        timesteps=timesteps,
        seq_length=seq_length,
        input_dim=input_dim,
    )

    return model.to(device)


if __name__ == "__main__":
    # 测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = create_alpha_ddim(seq_length=500, input_dim=3, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 4
    x_0 = torch.randn(batch_size, 500, 3, device=device)  # x, y, dt
    condition = torch.randn(batch_size, 4, device=device)

    losses = model.get_loss(x_0, condition)
    print(f"DDIM Loss: {losses['ddim_loss'].item():.4f}")

    # 测试采样
    print("\nTesting sampling...")
    samples = model.sample(
        batch_size=2,
        condition=condition[:2],
        num_inference_steps=20,
        alpha=1.5,
        device=device
    )
    print(f"Generated trajectory shape: {samples.shape}")  # (2, 500, 3)
