"""
U-Net噪声预测网络
用于DDIM扩散模型的噪声预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiscreteAlphaEmbedding(nn.Module):
    """
    离散α风格嵌入 (论文 Eq.10)

    将连续的α值离散化为S个bin，使用位置编码嵌入
    论文中α = 1/(β+1)，β是理论复杂度
    """

    def __init__(self, num_bins: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim

        # 离散bin的嵌入表
        self.embedding = nn.Embedding(num_bins, embedding_dim)

        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alpha: 连续α值 (batch,) 或 (batch, 1)，范围 [0, 1]
                   (代码中 α 已通过 compute_trajectory_alpha 归一化到 [0, 1])
        Returns:
            嵌入向量 (batch, embedding_dim)
        """
        if alpha.dim() == 2:
            alpha = alpha.squeeze(-1)

        # 将α离散化为bin索引
        # α ∈ [0, 1] -> bin ∈ [0, num_bins-1]
        # 注：论文 Eq.10 的 α'=1/(α+1) 变换适用于原始 α∈[0,+∞)
        #     但代码中 α 已归一化到 [0,1]，故不需要此变换
        bin_idx = (alpha * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

        # 获取嵌入
        emb = self.embedding(bin_idx)

        # 投影
        return self.proj(emb)


class LengthPredictionHead(nn.Module):
    """
    轨迹长度预测头 (Shared Encoder 模式)

    使用 U-Net encoder 的 bottleneck 特征进行长度预测
    y = log(m + 1)

    输入: encoder bottleneck 特征 (全局池化后) + 条件编码 + α编码
    输出: log(m + 1) 预测值
    """

    def __init__(
        self,
        encoder_dim: int,      # encoder bottleneck 通道数
        condition_dim: int,    # 条件嵌入维度 (time_emb_dim)
        hidden_dim: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()
        # 输入: encoder_feature + condition_emb + alpha_emb
        input_dim = encoder_dim + condition_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        encoder_feature: torch.Tensor,  # (batch, encoder_dim) - 全局池化后的特征
        condition_emb: torch.Tensor,    # (batch, condition_dim)
        alpha_emb: torch.Tensor,        # (batch, condition_dim)
    ) -> torch.Tensor:
        """
        Args:
            encoder_feature: encoder bottleneck 特征 (batch, encoder_dim)
            condition_emb: 条件编码 (batch, condition_dim)
            alpha_emb: α编码 (batch, condition_dim)

        Returns:
            log_length: log(m+1) 预测值 (batch, 1)
        """
        # 拼接所有特征
        combined = torch.cat([encoder_feature, condition_emb, alpha_emb], dim=-1)
        return self.mlp(combined)


class ConvBlock(nn.Module):
    """卷积块，包含两个卷积层和GroupNorm"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = None,
        groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        # 时间嵌入投影
        self.time_mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
            if time_emb_dim
            else None
        )

        # 残差连接
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor = None
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """自注意力块"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(b, 3, self.num_heads, self.head_dim, l)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # 注意力计算
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhlk,bhck->bhcl', attn, v)
        out = out.reshape(b, c, l)
        out = self.proj(out)

        return x + out


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> tuple:
        h = self.conv(x, time_emb)
        h = self.attention(h)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, in_channels, 4, stride=2, padding=1
        )
        self.conv = ConvBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.upsample(x)
        # 处理尺寸不匹配
        if h.shape[-1] != skip.shape[-1]:
            h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
        h = torch.cat([h, skip], dim=1)
        h = self.conv(h, time_emb)
        h = self.attention(h)
        return h


class TrajectoryUNet(nn.Module):
    """
    轨迹生成U-Net
    用于预测扩散过程中的噪声

    输入: (batch, seq_len, 2) - 噪声轨迹
    条件: 起点坐标、终点坐标、α (复杂度参数)
    输出: (batch, seq_len, 2) - 预测噪声

    论文公式10: ε_θ(x_t, t, c, α)

    新增功能: 轨迹长度预测
    基于条件和α预测轨迹长度 m，用于自适应长度生成
    """

    def __init__(
        self,
        seq_length: int = 500,
        input_dim: int = 2,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        time_emb_dim: int = 128,
        condition_dim: int = 4,  # 起点(2) + 终点(2)
        num_heads: int = 4,
        attention_levels: tuple = (False, True, True),
        enable_length_prediction: bool = True,  # 是否启用长度预测
    ):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.enable_length_prediction = enable_length_prediction

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 条件嵌入 (起点 + 终点)
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # α (复杂度/风格) 嵌入 - 论文Eq.10
        # 使用离散bin嵌入，而非连续MLP
        self.alpha_embedding = DiscreteAlphaEmbedding(
            num_bins=10,  # S=10 个风格bin
            embedding_dim=time_emb_dim
        )

        # 计算 encoder bottleneck 维度 (最深层通道数)
        self.bottleneck_dim = base_channels * channel_mults[-1]

        # 轨迹长度预测头 (Shared Encoder 模式)
        # 输入: encoder bottleneck 特征 + condition_emb + alpha_emb
        # 输出: log(m+1) 预测值
        if enable_length_prediction:
            self.length_head = LengthPredictionHead(
                encoder_dim=self.bottleneck_dim,
                condition_dim=time_emb_dim,
                hidden_dim=time_emb_dim,
                output_dim=1,
            )
        else:
            self.length_head = None

        # 输入投影
        self.input_proj = nn.Conv1d(input_dim, base_channels, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            self.down_blocks.append(
                DownBlock(
                    channels,
                    out_channels,
                    time_emb_dim,
                    has_attention=attention_levels[i]
                )
            )
            channels = out_channels

        # 中间块
        self.mid_block1 = ConvBlock(channels, channels, time_emb_dim)
        self.mid_attention = AttentionBlock(channels, num_heads)
        self.mid_block2 = ConvBlock(channels, channels, time_emb_dim)

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            in_channels = channels
            out_channels = base_channels * mult
            self.up_blocks.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    time_emb_dim,
                    has_attention=attention_levels[i]
                )
            )
            channels = out_channels

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, input_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, 2)
        time: torch.Tensor,  # (batch,)
        condition: torch.Tensor = None,  # (batch, 4) - 起点+终点
        alpha: torch.Tensor = None,  # (batch,) 或 (batch, 1) - 复杂度参数
    ) -> torch.Tensor:
        """
        前向传播 - 论文公式10: ε_θ(x_t, t, c, α)

        Args:
            x: 噪声轨迹 (batch, seq_len, 2)
            time: 时间步 (batch,)
            condition: 起点+终点条件 (batch, 4)
            alpha: 复杂度/风格参数 (batch,) 或 (batch, 1)
        """
        # 转换为 (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # 时间嵌入
        time_emb = self.time_mlp(time)

        # 条件嵌入
        if condition is not None:
            cond_emb = self.condition_mlp(condition)
            time_emb = time_emb + cond_emb

        # α 嵌入 - 论文Eq.10: 离散bin嵌入
        if alpha is not None:
            alpha_emb = self.alpha_embedding(alpha)
            time_emb = time_emb + alpha_emb

        # 输入投影
        h = self.input_proj(x)

        # 下采样
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skips.append(skip)

        # 中间处理
        h = self.mid_block1(h, time_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb)

        # 上采样
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, time_emb)

        # 输出
        h = self.output_proj(h)

        # 转换回 (batch, seq_len, 2)
        return h.transpose(1, 2)

    def _encode(
        self,
        x: torch.Tensor,          # (batch, seq_len, 2)
        time_emb: torch.Tensor,   # (batch, time_emb_dim)
    ) -> torch.Tensor:
        """
        运行 encoder 部分，获取 bottleneck 特征

        Args:
            x: 输入轨迹 (batch, seq_len, 2)
            time_emb: 时间嵌入 (batch, time_emb_dim)

        Returns:
            bottleneck: encoder bottleneck 特征 (batch, bottleneck_dim)
        """
        # 转换为 (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # 输入投影
        h = self.input_proj(x)

        # 下采样
        for down_block in self.down_blocks:
            h, _ = down_block(h, time_emb)

        # 中间处理
        h = self.mid_block1(h, time_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb)

        # 全局平均池化: (batch, channels, seq_len) -> (batch, channels)
        bottleneck = h.mean(dim=-1)

        return bottleneck

    def predict_length(
        self,
        x: torch.Tensor,          # (batch, seq_len, 2) - 输入轨迹 (噪声或初始化)
        condition: torch.Tensor,  # (batch, 4) - 起点+终点
        alpha: torch.Tensor,      # (batch,) 或 (batch, 1) - 复杂度参数
        time: torch.Tensor = None,  # (batch,) - 时间步 (默认 t=0)
    ) -> torch.Tensor:
        """
        预测轨迹长度 (Shared Encoder 模式)

        使用 U-Net encoder 特征 + 条件 + α 预测轨迹长度
        y = log(m + 1)

        Args:
            x: 输入轨迹 (batch, seq_len, 2)
            condition: 起点+终点条件 (batch, 4)
            alpha: 复杂度/风格参数 (batch,) 或 (batch, 1)
            time: 时间步 (batch,)，默认为 0

        Returns:
            log_length: log(m+1) 预测值 (batch, 1)
        """
        if self.length_head is None:
            raise RuntimeError("Length prediction is not enabled. Set enable_length_prediction=True.")

        batch_size = x.shape[0]
        device = x.device

        # 默认时间步为 0
        if time is None:
            time = torch.zeros(batch_size, device=device)

        # 获取时间嵌入
        time_emb = self.time_mlp(time)

        # 获取条件编码
        cond_emb = self.condition_mlp(condition)  # (batch, time_emb_dim)
        time_emb = time_emb + cond_emb

        # 获取α编码 (只用于 length_head，不传入 encoder)
        alpha_emb = self.alpha_embedding(alpha)  # (batch, time_emb_dim)

        # 运行 encoder 获取 bottleneck 特征 (不包含 α，让 encoder 专注于几何特征)
        encoder_feature = self._encode(x, time_emb)  # (batch, bottleneck_dim)

        # 通过长度预测头
        log_length = self.length_head(encoder_feature, cond_emb, alpha_emb)  # (batch, 1)

        return log_length

    def decode_length(self, log_length: torch.Tensor, max_length: int = None) -> torch.Tensor:
        """
        将 log(m+1) 解码为实际长度 m

        m̂ = round(exp(ŷ) - 1)

        Args:
            log_length: log(m+1) 预测值 (batch, 1)
            max_length: 最大长度限制 (默认使用 seq_length)

        Returns:
            length: 预测长度 (batch,) - 整数值
        """
        if max_length is None:
            max_length = self.seq_length

        # 反变换: m = exp(y) - 1
        length = torch.exp(log_length.squeeze(-1)) - 1

        # 取整并限制范围
        length = torch.round(length).long()
        length = torch.clamp(length, min=2, max=max_length)

        return length


if __name__ == "__main__":
    # 测试模型
    print("Testing TrajectoryUNet with Shared Encoder length prediction...")
    model = TrajectoryUNet(seq_length=500, input_dim=2, enable_length_prediction=True)

    batch_size = 4
    x = torch.randn(batch_size, 500, 2)
    t = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 4)
    alpha = torch.rand(batch_size)  # 复杂度参数 [0, 1]

    # 测试论文公式10: ε_θ(x_t, t, c, α)
    output = model(x, t, cond, alpha)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Alpha shape: {alpha.shape}")
    print(f"Bottleneck dim: {model.bottleneck_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试 Shared Encoder 长度预测
    print("\nTesting Shared Encoder length prediction...")
    log_length = model.predict_length(x, cond, alpha)
    print(f"Predicted log(m+1): {log_length.squeeze().tolist()}")

    # 测试解码
    decoded_length = model.decode_length(log_length)
    print(f"Decoded length m: {decoded_length.tolist()}")

    # 测试真实长度的log-transform
    true_lengths = torch.tensor([50, 100, 200, 500])
    true_log_lengths = torch.log(true_lengths.float() + 1)
    print(f"\nTrue lengths: {true_lengths.tolist()}")
    print(f"True log(m+1): {true_log_lengths.tolist()}")
