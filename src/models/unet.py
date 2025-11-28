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
    ):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.base_channels = base_channels

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
        self.alpha_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

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

        # α 嵌入 - 论文Eq.10的关键
        if alpha is not None:
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(-1)  # (batch,) -> (batch, 1)
            alpha_emb = self.alpha_mlp(alpha)
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


if __name__ == "__main__":
    # 测试模型
    model = TrajectoryUNet(seq_length=500, input_dim=2)

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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
