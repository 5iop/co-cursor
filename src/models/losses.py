"""
DMTG损失函数模块 (论文公式11-14)

L = w1·LDDIM + w2·Lsim + w3·Lstyle + w4·Llength

- LDDIM (Eq.11): 噪声预测MSE
- Lsim  (Eq.12): 生成轨迹与人类模板的MSE
- Lstyle(Eq.13): 生成轨迹复杂度与目标α的差距
- Llength: 轨迹长度预测L1损失 (新增)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DMTGLoss(nn.Module):
    """
    DMTG完整损失函数 (论文公式14 扩展)

    L = w1·LDDIM + w2·Lsim + w3·Lstyle + w4·Llength

    其中 (严格按照论文):
    - LDDIM: 噪声预测MSE (公式11)
    - Lsim: 生成轨迹与人类模板的MSE (公式12) - ||p_a - X̂||²
    - Lstyle: 生成轨迹复杂度与目标α的差距 (公式13) - ||α - ratio(p_a)||²
    - Llength: 轨迹长度预测L1损失 (新增) - |log(m+1) - log(m̂+1)|
    """

    def __init__(
        self,
        lambda_ddim: float = 1.0,     # w1: DDIM损失权重
        lambda_sim: float = 0.1,      # w2: 相似度损失权重
        lambda_style: float = 0.05,   # w3: 风格损失权重
        lambda_length: float = 0.1,   # w4: 长度预测损失权重
    ):
        super().__init__()
        self.lambda_ddim = lambda_ddim
        self.lambda_sim = lambda_sim
        self.lambda_style = lambda_style
        self.lambda_length = lambda_length

    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        predicted_x0: torch.Tensor = None,
        target_x0: torch.Tensor = None,  # 人类模板 X̂
        alpha: torch.Tensor = None,       # 目标复杂度 α
        t: torch.Tensor = None,
        timesteps: int = 1000,
        mask: torch.Tensor = None,        # 有效位置掩码 (batch, seq_len)
        predicted_log_length: torch.Tensor = None,  # 预测的 log(m+1)
        target_length: torch.Tensor = None,         # 目标长度 m
    ) -> Dict[str, torch.Tensor]:
        """
        计算DMTG总损失 (论文公式14 扩展)
        L = w1·LDDIM + w2·Lsim + w3·Lstyle + w4·Llength

        Args:
            predicted_noise: 预测的噪声 (batch, seq_len, 2)
            target_noise: 目标噪声 (batch, seq_len, 2)
            predicted_x0: 预测的轨迹 p_a (batch, seq_len, 2)
            target_x0: 人类模板轨迹 X̂ (batch, seq_len, 2)
            alpha: 目标复杂度参数 (batch,) - 用于Lstyle
            t: 时间步
            timesteps: 总时间步数
            mask: 有效位置掩码 (batch, seq_len)，1表示有效，0表示padding
            predicted_log_length: 预测的 log(m+1) (batch, 1)
            target_length: 目标长度 m (batch,) 或 (batch, 1)
        """
        losses = {}

        # 1. LDDIM: 噪声预测MSE (公式11)
        ddim_loss = self._masked_mse(predicted_noise, target_noise, mask)
        losses['ddim_loss'] = ddim_loss
        total_loss = self.lambda_ddim * ddim_loss

        # 2. 辅助损失 (论文 Eq.12-13)
        # 移除 t < 0.5T 约束，所有时间步都计算辅助损失
        if predicted_x0 is not None and target_x0 is not None:
            # Lsim (公式12): ||p_a - X̂||² - 生成轨迹与人类模板的差距
            sim_loss = self._similarity_loss(predicted_x0, target_x0, mask)
            losses['similarity_loss'] = sim_loss
            total_loss = total_loss + self.lambda_sim * sim_loss

            # Lstyle (公式13): ||α - ratio(p_a)||² - 复杂度与目标α的差距
            if alpha is not None:
                style_loss = self._style_loss(predicted_x0, alpha, mask)
                losses['style_loss'] = style_loss
                total_loss = total_loss + self.lambda_style * style_loss

        # 3. Llength: 长度预测L1损失 (新增)
        if predicted_log_length is not None and target_length is not None:
            length_loss = self._length_loss(predicted_log_length, target_length)
            losses['length_loss'] = length_loss
            total_loss = total_loss + self.lambda_length * length_loss

        losses['total_loss'] = total_loss
        return losses

    def _masked_mse(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """带掩码的MSE损失"""
        if mask is None:
            return F.mse_loss(predicted, target)

        # mask: (batch, seq_len) -> (batch, seq_len, 1)
        mask = mask.unsqueeze(-1)
        # 计算MSE
        mse = (predicted - target) ** 2
        # 应用掩码并求平均
        masked_mse = (mse * mask).sum() / (mask.sum() * predicted.shape[-1] + 1e-8)
        return masked_mse

    def _similarity_loss(
        self,
        predicted: torch.Tensor,
        template: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        相似度损失 (论文公式12)
        Lsim := ||p_a - X̂||²

        p_a: 生成的轨迹
        X̂: 人类模板轨迹
        """
        return self._masked_mse(predicted, template, mask)

    def _style_loss(
        self,
        predicted: torch.Tensor,
        alpha: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        风格损失 (论文 Eq.13 + Eq.8-9)
        Lstyle := ||α - C(p_a)||²

        使用 MST 近似熵作为复杂度度量 (论文 Eq.8-9):
        C(p_a) = (MST_ratio - 1) / 2, 归一化到 [0, 1]

        α: 目标复杂度参数 (batch,) ∈ [0.3, 0.8]
        C(p_a): 生成轨迹的 MST 复杂度 ∈ [0, 1]
        mask: 有效位置掩码 (batch, seq_len)
        """
        batch_size = predicted.shape[0]

        if mask is not None:
            # 对于每个样本，找到最后一个有效位置的索引
            # mask: (batch, seq_len)
            lengths = mask.sum(dim=1).long()  # (batch,)

            # 计算路径长度（只计算有效段）
            segments = predicted[:, 1:, :] - predicted[:, :-1, :]  # (batch, seq-1, 2)
            segment_norms = torch.norm(segments, dim=-1)  # (batch, seq-1)
            # 只计算mask有效范围内的段
            segment_mask = mask[:, 1:] * mask[:, :-1]  # (batch, seq-1)
            path_length = (segment_norms * segment_mask).sum(dim=-1)  # (batch,)

            # 获取每个样本的真实终点
            # 使用 lengths-1 作为索引获取最后一个有效点
            end_indices = (lengths - 1).clamp(min=0)  # (batch,)
            end_points = predicted[torch.arange(batch_size, device=predicted.device), end_indices]  # (batch, 2)
            start_points = predicted[:, 0, :]  # (batch, 2)
        else:
            # 无mask时使用全部数据
            segments = predicted[:, 1:, :] - predicted[:, :-1, :]
            path_length = torch.norm(segments, dim=-1).sum(dim=-1)
            end_points = predicted[:, -1, :]
            start_points = predicted[:, 0, :]

        # 直线距离
        straight_dist = torch.norm(end_points - start_points, dim=-1) + 1e-8  # (batch,)

        # MST 比率
        mst_ratio = path_length / straight_dist  # (batch,)

        # 论文 Eq.8-9: 使用 β/(β+1) 公式计算复杂度
        # β = mst_ratio - 1 (曲线相对于直线的额外长度比)
        # complexity = β / (β + 1) = (mst_ratio - 1) / mst_ratio
        # ratio=1 (直线) -> complexity=0
        # ratio→∞ (极复杂) -> complexity→1
        beta = mst_ratio - 1.0
        pred_complexity = beta / (beta + 1.0 + 1e-8)
        pred_complexity = torch.clamp(pred_complexity, 0.0, 1.0)

        # Lstyle = ||α - C(p_a)||²
        # α = 1/(β+1) 是论文的理论复杂度输入
        style_loss = F.mse_loss(pred_complexity, alpha)

        return style_loss

    def _length_loss(
        self,
        predicted_log_length: torch.Tensor,
        target_length: torch.Tensor,
    ) -> torch.Tensor:
        """
        轨迹长度预测L1损失

        使用 log-transform 来处理长度预测:
        y = log(m + 1)

        L1 损失: |log(m+1) - log(m̂+1)|

        Args:
            predicted_log_length: 预测的 log(m+1) (batch, 1)
            target_length: 目标长度 m (batch,) 或 (batch, 1)

        Returns:
            L1 损失标量
        """
        # 确保 target_length 是正确的形状
        if target_length.dim() == 1:
            target_length = target_length.unsqueeze(-1)  # (batch,) -> (batch, 1)

        # 计算目标的 log-transform: y = log(m + 1)
        target_log_length = torch.log(target_length.float() + 1)

        # L1 损失
        length_loss = F.l1_loss(predicted_log_length, target_log_length)

        return length_loss


if __name__ == "__main__":
    # 测试损失函数 (论文公式11-14 + 长度预测)
    print("Testing DMTG Loss (Paper Eq. 11-14 + Length Prediction)")
    print("=" * 50)

    batch_size = 4
    seq_len = 50

    predicted_noise = torch.randn(batch_size, seq_len, 2)
    target_noise = torch.randn(batch_size, seq_len, 2)
    predicted_x0 = torch.randn(batch_size, seq_len, 2)
    target_x0 = torch.randn(batch_size, seq_len, 2)  # human template
    alpha = torch.rand(batch_size)  # target complexity

    # 长度预测测试数据
    target_length = torch.randint(10, 500, (batch_size,))  # 真实长度
    predicted_log_length = torch.log(target_length.float() + 1) + torch.randn(batch_size, 1) * 0.5  # 有噪声的预测

    # Test paper Eq.14: L = w1*LDDIM + w2*Lsim + w3*Lstyle + w4*Llength
    loss_fn = DMTGLoss(
        lambda_ddim=1.0,   # w1
        lambda_sim=0.1,    # w2
        lambda_style=0.05, # w3
        lambda_length=0.1, # w4
    )
    losses = loss_fn(
        predicted_noise=predicted_noise,
        target_noise=target_noise,
        predicted_x0=predicted_x0,
        target_x0=target_x0,
        alpha=alpha,
        predicted_log_length=predicted_log_length,
        target_length=target_length,
    )

    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    print(f"\nAlpha values (target complexity): {[f'{a:.3f}' for a in alpha.tolist()]}")

    # Verify Lstyle: ||target_ratio - ratio(p_a)||^2
    segments = predicted_x0[:, 1:, :] - predicted_x0[:, :-1, :]
    path_length = torch.norm(segments, dim=-1).sum(dim=-1)
    straight_dist = torch.norm(predicted_x0[:, -1, :] - predicted_x0[:, 0, :], dim=-1) + 1e-8
    pred_ratio = path_length / straight_dist
    target_ratio = 1.0 + alpha * 2.0

    print(f"\nPredicted path ratios: {[f'{r:.3f}' for r in pred_ratio.tolist()]}")
    print(f"Target ratios (from alpha): {[f'{r:.3f}' for r in target_ratio.tolist()]}")

    # 验证长度预测
    print(f"\n--- Length Prediction ---")
    print(f"Target lengths: {target_length.tolist()}")
    print(f"Target log(m+1): {[f'{v:.3f}' for v in torch.log(target_length.float() + 1).tolist()]}")
    print(f"Predicted log(m+1): {[f'{v:.3f}' for v in predicted_log_length.squeeze().tolist()]}")

    # 反变换
    decoded_length = torch.round(torch.exp(predicted_log_length.squeeze()) - 1).long()
    print(f"Decoded predicted lengths: {decoded_length.tolist()}")
