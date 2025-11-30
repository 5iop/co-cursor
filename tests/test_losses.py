"""
DMTGLoss 损失函数测试

测试覆盖:
- L_DDIM (Eq.11): 噪声预测 MSE
- L_sim (Eq.12): 相似度损失
- L_style (Eq.13): 风格损失
- L_length: 长度预测 L1 损失
- 总损失计算
- 掩码处理
"""
import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.losses import DMTGLoss


# ==================== Fixtures ====================

@pytest.fixture
def loss_fn():
    """创建 DMTGLoss 实例"""
    return DMTGLoss(
        lambda_ddim=1.0,
        lambda_sim=0.1,
        lambda_style=0.05,
        lambda_length=0.1,
    )


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 50


@pytest.fixture
def predicted_noise(batch_size, seq_len):
    return torch.randn(batch_size, seq_len, 2)


@pytest.fixture
def target_noise(batch_size, seq_len):
    return torch.randn(batch_size, seq_len, 2)


@pytest.fixture
def predicted_x0(batch_size, seq_len):
    """生成预测轨迹"""
    return torch.randn(batch_size, seq_len, 2)


@pytest.fixture
def target_x0(batch_size, seq_len):
    """生成人类模板轨迹"""
    return torch.randn(batch_size, seq_len, 2)


@pytest.fixture
def alpha(batch_size):
    """生成目标复杂度 alpha (论文方案A: path_ratio ∈ [1, +∞))"""
    return torch.rand(batch_size) * 4.0 + 1.0  # [1.0, 5.0]


@pytest.fixture
def mask(batch_size, seq_len):
    """生成有效位置掩码"""
    # 随机长度 [20, seq_len]
    lengths = torch.randint(20, seq_len + 1, (batch_size,))
    mask = torch.zeros(batch_size, seq_len)
    for i in range(batch_size):
        mask[i, :lengths[i]] = 1.0
    return mask


@pytest.fixture
def target_length(batch_size):
    """生成目标长度"""
    return torch.randint(20, 100, (batch_size,))


@pytest.fixture
def predicted_log_length(target_length):
    """生成预测的 log(m+1)，带有一些噪声"""
    return torch.log(target_length.float() + 1).unsqueeze(-1) + torch.randn(target_length.shape[0], 1) * 0.3


# ==================== L_DDIM 测试 ====================

class TestLDDIM:
    """L_DDIM (Eq.11) 噪声预测 MSE 测试"""

    def test_ddim_loss_basic(self, loss_fn, predicted_noise, target_noise):
        """测试基本 DDIM 损失计算"""
        result = loss_fn(predicted_noise, target_noise)

        assert 'ddim_loss' in result
        assert 'total_loss' in result
        assert result['ddim_loss'].ndim == 0  # scalar
        assert result['ddim_loss'] >= 0

    def test_ddim_loss_zero_when_equal(self, loss_fn, target_noise):
        """测试预测等于目标时损失为 0"""
        result = loss_fn(target_noise, target_noise)

        assert result['ddim_loss'].item() < 1e-6

    def test_ddim_loss_with_mask(self, loss_fn, predicted_noise, target_noise, mask):
        """测试带掩码的 DDIM 损失"""
        result = loss_fn(predicted_noise, target_noise, mask=mask)

        assert 'ddim_loss' in result
        assert result['ddim_loss'] >= 0

    def test_ddim_loss_mask_effect(self, loss_fn, batch_size, seq_len):
        """测试掩码是否正确忽略 padding"""
        # 创建数据：前 10 个位置相同，后面不同
        predicted = torch.randn(batch_size, seq_len, 2)
        target = predicted.clone()
        target[:, 10:, :] = torch.randn(batch_size, seq_len - 10, 2)  # 后面不同

        # 掩码只覆盖前 10 个位置
        mask = torch.zeros(batch_size, seq_len)
        mask[:, :10] = 1.0

        result = loss_fn(predicted, target, mask=mask)

        # 有效区域完全相同，损失应接近 0
        assert result['ddim_loss'].item() < 1e-6

    def test_ddim_loss_weight(self, predicted_noise, target_noise):
        """测试 lambda_ddim 权重"""
        loss_fn_w1 = DMTGLoss(lambda_ddim=1.0)
        loss_fn_w2 = DMTGLoss(lambda_ddim=2.0)

        result1 = loss_fn_w1(predicted_noise, target_noise)
        result2 = loss_fn_w2(predicted_noise, target_noise)

        # 权重为 2 时，total_loss 应该是权重为 1 时的 2 倍
        assert abs(result2['total_loss'].item() - 2 * result1['total_loss'].item()) < 1e-6


# ==================== L_sim 测试 ====================

class TestLSim:
    """L_sim (Eq.12) 相似度损失测试"""

    def test_sim_loss_basic(self, loss_fn, predicted_noise, target_noise, predicted_x0, target_x0):
        """测试基本相似度损失"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
        )

        assert 'similarity_loss' in result
        assert result['similarity_loss'] >= 0

    def test_sim_loss_zero_when_equal(self, loss_fn, predicted_noise, target_noise, target_x0):
        """测试预测轨迹等于模板时损失为 0"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=target_x0,
            target_x0=target_x0,
        )

        assert result['similarity_loss'].item() < 1e-6

    def test_sim_loss_with_mask(self, loss_fn, predicted_noise, target_noise, predicted_x0, target_x0, mask):
        """测试带掩码的相似度损失"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            mask=mask,
        )

        assert 'similarity_loss' in result
        assert result['similarity_loss'] >= 0

    def test_sim_loss_weight(self, predicted_noise, target_noise, predicted_x0, target_x0):
        """测试 lambda_sim 权重"""
        loss_fn_w1 = DMTGLoss(lambda_ddim=0.0, lambda_sim=0.1, lambda_style=0.0)
        loss_fn_w2 = DMTGLoss(lambda_ddim=0.0, lambda_sim=0.2, lambda_style=0.0)

        result1 = loss_fn_w1(predicted_noise, target_noise, predicted_x0=predicted_x0, target_x0=target_x0)
        result2 = loss_fn_w2(predicted_noise, target_noise, predicted_x0=predicted_x0, target_x0=target_x0)

        # 权重为 0.2 时，total_loss 应该是权重为 0.1 时的 2 倍
        assert abs(result2['total_loss'].item() - 2 * result1['total_loss'].item()) < 1e-5

    def test_sim_loss_is_mse(self, loss_fn, predicted_noise, target_noise, predicted_x0, target_x0):
        """测试相似度损失是 MSE"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
        )

        # 手动计算 MSE
        expected_mse = F.mse_loss(predicted_x0, target_x0)

        assert abs(result['similarity_loss'].item() - expected_mse.item()) < 1e-6


# ==================== L_style 测试 ====================

class TestLStyle:
    """L_style (Eq.13) 风格损失测试"""

    def test_style_loss_basic(self, loss_fn, predicted_noise, target_noise, predicted_x0, target_x0, alpha):
        """测试基本风格损失"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            alpha=alpha,
        )

        assert 'style_loss' in result
        assert result['style_loss'] >= 0

    def test_style_loss_straight_line(self, loss_fn, predicted_noise, target_noise, batch_size, seq_len):
        """测试直线轨迹的复杂度应接近 1 (论文方案A: α = path_ratio)"""
        # 创建直线轨迹: 从 (0, 0) 到 (1, 1)
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)
        straight_traj = torch.stack([t, t], dim=-1)  # (batch, seq, 2)

        # alpha = 1 表示直线轨迹 (path_ratio = 1)
        alpha = torch.ones(batch_size)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=straight_traj,
            target_x0=straight_traj,
            alpha=alpha,
        )

        # 直线轨迹的 path_ratio ≈ 1，与 alpha=1 匹配，损失应该很小
        assert result['style_loss'].item() < 0.1

    def test_style_loss_complex_trajectory(self, loss_fn, predicted_noise, target_noise, batch_size, seq_len):
        """测试复杂轨迹的复杂度应更高 (论文方案A: α = path_ratio)"""
        # 创建曲折轨迹
        t = torch.linspace(0, 1, seq_len)
        # 正弦波路径
        x = t
        y = torch.sin(t * 4 * 3.14159)  # 两个完整周期
        complex_traj = torch.stack([x, y], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        # alpha = 3.0 表示期望复杂轨迹 (path_ratio = 3)
        alpha = torch.full((batch_size,), 3.0)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=complex_traj,
            target_x0=complex_traj,
            alpha=alpha,
        )

        # 风格损失应该存在
        assert 'style_loss' in result

    def test_style_loss_with_mask(self, loss_fn, predicted_noise, target_noise, predicted_x0, target_x0, alpha, mask):
        """测试带掩码的风格损失"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            alpha=alpha,
            mask=mask,
        )

        assert 'style_loss' in result
        assert result['style_loss'] >= 0

    def test_style_loss_complexity_formula(self, batch_size, seq_len):
        """测试 path_ratio 公式 (论文方案A): α = path_length / straight_distance"""
        # 创建简单轨迹用于验证公式
        loss_fn = DMTGLoss()

        # 直线: path_ratio = 1
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)
        straight = torch.stack([t, t], dim=-1)

        # 计算路径长度
        segments = straight[:, 1:, :] - straight[:, :-1, :]
        path_length = torch.norm(segments, dim=-1).sum(dim=-1)
        straight_dist = torch.norm(straight[:, -1, :] - straight[:, 0, :], dim=-1)
        path_ratio = path_length / (straight_dist + 1e-8)

        # 直线的 path_ratio 应接近 1
        assert (path_ratio - 1.0).abs().mean().item() < 0.1

    def test_style_loss_weight(self, predicted_noise, target_noise, predicted_x0, target_x0, alpha):
        """测试 lambda_style 权重"""
        loss_fn_w1 = DMTGLoss(lambda_ddim=0.0, lambda_sim=0.0, lambda_style=0.05)
        loss_fn_w2 = DMTGLoss(lambda_ddim=0.0, lambda_sim=0.0, lambda_style=0.1)

        result1 = loss_fn_w1(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0, target_x0=target_x0, alpha=alpha
        )
        result2 = loss_fn_w2(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0, target_x0=target_x0, alpha=alpha
        )

        # 权重为 0.1 时，total_loss 应该是权重为 0.05 时的 2 倍
        assert abs(result2['total_loss'].item() - 2 * result1['total_loss'].item()) < 1e-5


# ==================== L_length 测试 ====================

class TestLLength:
    """L_length 长度预测 L1 损失测试"""

    def test_length_loss_basic(self, loss_fn, predicted_noise, target_noise, predicted_log_length, target_length):
        """测试基本长度损失"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        assert 'length_loss' in result
        assert result['length_loss'] >= 0

    def test_length_loss_zero_when_equal(self, loss_fn, predicted_noise, target_noise, target_length):
        """测试预测等于目标时损失为 0"""
        # 精确的 log(m+1)
        predicted_log_length = torch.log(target_length.float() + 1).unsqueeze(-1)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        assert result['length_loss'].item() < 1e-6

    def test_length_loss_log_transform(self, loss_fn, predicted_noise, target_noise, batch_size):
        """测试 log-transform 是否正确应用"""
        target_length = torch.tensor([10, 100, 1000, 10000])
        predicted_log_length = torch.log(target_length.float() + 1).unsqueeze(-1)

        result = loss_fn(
            predicted_noise[:4], target_noise[:4],
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        # 精确预测，损失应为 0
        assert result['length_loss'].item() < 1e-6

    def test_length_loss_is_l1(self, loss_fn, predicted_noise, target_noise, predicted_log_length, target_length):
        """测试长度损失是 L1"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        # 手动计算 L1
        target_log = torch.log(target_length.float() + 1).unsqueeze(-1)
        expected_l1 = F.l1_loss(predicted_log_length, target_log)

        assert abs(result['length_loss'].item() - expected_l1.item()) < 1e-6

    def test_length_loss_weight(self, predicted_noise, target_noise, predicted_log_length, target_length):
        """测试 lambda_length 权重"""
        loss_fn_w1 = DMTGLoss(lambda_ddim=0.0, lambda_length=0.1)
        loss_fn_w2 = DMTGLoss(lambda_ddim=0.0, lambda_length=0.2)

        result1 = loss_fn_w1(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )
        result2 = loss_fn_w2(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        # 权重为 0.2 时，total_loss 应该是权重为 0.1 时的 2 倍
        assert abs(result2['total_loss'].item() - 2 * result1['total_loss'].item()) < 1e-5

    def test_length_loss_1d_target(self, loss_fn, predicted_noise, target_noise, batch_size):
        """测试 target_length 为 1D 张量"""
        target_length = torch.randint(20, 100, (batch_size,))  # (batch,)
        predicted_log_length = torch.log(target_length.float() + 1).unsqueeze(-1)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        assert result['length_loss'].item() < 1e-6

    def test_length_loss_2d_target(self, loss_fn, predicted_noise, target_noise, batch_size):
        """测试 target_length 为 2D 张量"""
        target_length = torch.randint(20, 100, (batch_size, 1))  # (batch, 1)
        predicted_log_length = torch.log(target_length.float() + 1)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        assert result['length_loss'].item() < 1e-6


# ==================== 总损失测试 ====================

class TestTotalLoss:
    """总损失计算测试"""

    def test_total_loss_all_components(
        self, loss_fn, predicted_noise, target_noise,
        predicted_x0, target_x0, alpha,
        predicted_log_length, target_length, mask
    ):
        """测试包含所有组件的总损失"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            alpha=alpha,
            mask=mask,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        # 检查所有损失组件
        assert 'ddim_loss' in result
        assert 'similarity_loss' in result
        assert 'style_loss' in result
        assert 'length_loss' in result
        assert 'total_loss' in result

        # 验证总损失计算
        expected_total = (
            1.0 * result['ddim_loss'] +
            0.1 * result['similarity_loss'] +
            0.05 * result['style_loss'] +
            0.1 * result['length_loss']
        )
        assert abs(result['total_loss'].item() - expected_total.item()) < 1e-5

    def test_total_loss_only_ddim(self, loss_fn, predicted_noise, target_noise):
        """测试只有 DDIM 损失"""
        result = loss_fn(predicted_noise, target_noise)

        assert 'ddim_loss' in result
        assert 'total_loss' in result
        assert 'similarity_loss' not in result
        assert 'style_loss' not in result
        assert 'length_loss' not in result

        # total_loss = lambda_ddim * ddim_loss
        assert abs(result['total_loss'].item() - 1.0 * result['ddim_loss'].item()) < 1e-6

    def test_total_loss_no_style(self, loss_fn, predicted_noise, target_noise, predicted_x0, target_x0):
        """测试没有 style 损失 (alpha=None)"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            alpha=None,
        )

        assert 'ddim_loss' in result
        assert 'similarity_loss' in result
        assert 'style_loss' not in result


# ==================== 数值稳定性测试 ====================

class TestNumericalStability:
    """数值稳定性测试"""

    def test_no_nan_output(
        self, loss_fn, predicted_noise, target_noise,
        predicted_x0, target_x0, alpha, mask,
        predicted_log_length, target_length
    ):
        """测试输出无 NaN"""
        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            alpha=alpha,
            mask=mask,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        for name, value in result.items():
            assert not torch.isnan(value), f"{name} 包含 NaN"
            assert not torch.isinf(value), f"{name} 包含 Inf"

    def test_zero_distance_trajectory(self, loss_fn, predicted_noise, target_noise, batch_size, seq_len):
        """测试起点终点相同的轨迹"""
        # 所有点都在同一位置
        zero_traj = torch.zeros(batch_size, seq_len, 2)
        alpha = torch.zeros(batch_size)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=zero_traj,
            target_x0=zero_traj,
            alpha=alpha,
        )

        # 不应该产生 NaN (因为有 epsilon 保护)
        assert not torch.isnan(result['style_loss']), "零距离轨迹产生 NaN"

    def test_large_values(self, loss_fn, batch_size, seq_len):
        """测试大值输入"""
        large_noise = torch.randn(batch_size, seq_len, 2) * 1000
        large_x0 = torch.randn(batch_size, seq_len, 2) * 1000
        alpha = torch.rand(batch_size)

        result = loss_fn(
            large_noise, large_noise.clone(),
            predicted_x0=large_x0,
            target_x0=large_x0.clone(),
            alpha=alpha,
        )

        for name, value in result.items():
            assert not torch.isnan(value), f"大值输入导致 {name} NaN"

    def test_gradient_flow(
        self, loss_fn, predicted_noise, target_noise,
        predicted_x0, target_x0, alpha,
        predicted_log_length, target_length
    ):
        """测试梯度流"""
        predicted_noise.requires_grad_(True)
        predicted_x0.requires_grad_(True)
        predicted_log_length.requires_grad_(True)

        result = loss_fn(
            predicted_noise, target_noise,
            predicted_x0=predicted_x0,
            target_x0=target_x0,
            alpha=alpha,
            predicted_log_length=predicted_log_length,
            target_length=target_length,
        )

        result['total_loss'].backward()

        assert predicted_noise.grad is not None
        assert predicted_x0.grad is not None
        assert predicted_log_length.grad is not None
        assert not torch.isnan(predicted_noise.grad).any()
        assert not torch.isnan(predicted_x0.grad).any()
        assert not torch.isnan(predicted_log_length.grad).any()
