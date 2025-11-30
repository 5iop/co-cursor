"""
TrajectoryUNet 模型测试

测试覆盖:
- 前向传播形状
- 长度预测功能 (Shared Encoder 模式)
- 梯度流
- 参数数量
"""
import pytest
import torch


class TestForwardPass:
    """前向传播测试"""

    def test_output_shape(self, unet, sample_trajectory, sample_condition, sample_alpha, device):
        """测试输出形状"""
        batch_size = sample_trajectory.shape[0]
        t = torch.randint(0, 100, (batch_size,), device=device)

        output = unet(sample_trajectory, t, sample_condition, sample_alpha)

        assert output.shape == sample_trajectory.shape, \
            f"输出形状不匹配: {output.shape} != {sample_trajectory.shape}"

    def test_output_dtype(self, unet, sample_trajectory, sample_condition, sample_alpha, device):
        """测试输出数据类型"""
        batch_size = sample_trajectory.shape[0]
        t = torch.randint(0, 100, (batch_size,), device=device)

        output = unet(sample_trajectory, t, sample_condition, sample_alpha)

        assert output.dtype == sample_trajectory.dtype

    def test_different_batch_sizes(self, unet, device):
        """测试不同 batch size"""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 100, 2, device=device)
            t = torch.randint(0, 100, (batch_size,), device=device)
            cond = torch.randn(batch_size, 4, device=device)
            alpha = torch.rand(batch_size, device=device) * 4.0 + 1.0  # [1, 5]

            output = unet(x, t, cond, alpha)
            assert output.shape == (batch_size, 100, 2)

    def test_no_condition(self, unet, sample_trajectory, device):
        """测试无条件输入"""
        batch_size = sample_trajectory.shape[0]
        t = torch.randint(0, 100, (batch_size,), device=device)

        # 无条件应该也能工作
        output = unet(sample_trajectory, t, None, None)
        assert output.shape == sample_trajectory.shape


class TestLengthPrediction:
    """长度预测测试 (Shared Encoder 模式)"""

    def test_predict_length_shape(self, unet, sample_trajectory, sample_condition, sample_alpha):
        """测试长度预测输出形状"""
        log_length = unet.predict_length(sample_trajectory, sample_condition, sample_alpha)

        batch_size = sample_trajectory.shape[0]
        assert log_length.shape == (batch_size, 1), \
            f"长度预测形状不正确: {log_length.shape}"

    def test_decode_length_range(self, unet, sample_trajectory, sample_condition, sample_alpha):
        """测试解码长度范围"""
        log_length = unet.predict_length(sample_trajectory, sample_condition, sample_alpha)
        decoded = unet.decode_length(log_length)

        batch_size = sample_trajectory.shape[0]
        assert decoded.shape == (batch_size,)
        assert (decoded >= 2).all(), "解码长度应 >= 2"
        assert (decoded <= 100).all(), "解码长度应 <= seq_length"

    def test_length_prediction_disabled(self, unet_no_length, sample_trajectory, sample_condition, sample_alpha):
        """测试禁用长度预测时的行为"""
        with pytest.raises(RuntimeError, match="Length prediction is not enabled"):
            unet_no_length.predict_length(sample_trajectory, sample_condition, sample_alpha)

    def test_bottleneck_dim(self, unet):
        """测试 bottleneck 维度"""
        # base_channels=32, channel_mults=(1, 2, 4)
        # bottleneck_dim = 32 * 4 = 128
        assert hasattr(unet, 'bottleneck_dim')
        assert unet.bottleneck_dim == 32 * 4  # 因为我们用 base_channels=32

    def test_encode_method(self, unet, sample_trajectory, device):
        """测试 _encode 方法"""
        batch_size = sample_trajectory.shape[0]
        time_emb = torch.zeros(batch_size, unet.time_emb_dim, device=device)

        # 测试编码器
        bottleneck = unet._encode(sample_trajectory, time_emb)

        assert bottleneck.shape == (batch_size, unet.bottleneck_dim)


class TestGradientFlow:
    """梯度流测试"""

    def test_gradient_through_forward(self, unet, sample_trajectory, sample_condition, sample_alpha, device):
        """测试前向传播的梯度流"""
        batch_size = sample_trajectory.shape[0]
        t = torch.randint(0, 100, (batch_size,), device=device)

        sample_trajectory.requires_grad_(True)
        output = unet(sample_trajectory, t, sample_condition, sample_alpha)
        loss = output.sum()
        loss.backward()

        assert sample_trajectory.grad is not None
        assert not torch.isnan(sample_trajectory.grad).any()

    def test_gradient_through_length_prediction(self, unet, sample_trajectory, sample_condition, sample_alpha):
        """测试长度预测的梯度流"""
        sample_trajectory.requires_grad_(True)

        log_length = unet.predict_length(sample_trajectory, sample_condition, sample_alpha)
        loss = log_length.sum()
        loss.backward()

        assert sample_trajectory.grad is not None
        assert not torch.isnan(sample_trajectory.grad).any()


class TestModelProperties:
    """模型属性测试"""

    def test_parameter_count(self, unet):
        """测试参数数量"""
        param_count = sum(p.numel() for p in unet.parameters())
        assert param_count > 0
        print(f"\nU-Net 参数数量: {param_count:,}")

    def test_length_head_exists(self, unet):
        """测试长度预测头存在"""
        assert hasattr(unet, 'length_head')
        assert unet.length_head is not None

    def test_length_head_not_exists(self, unet_no_length):
        """测试禁用时长度预测头不存在"""
        assert unet_no_length.length_head is None

    def test_alpha_embedding(self, unet, device):
        """测试 alpha 嵌入 (论文方案A: path_ratio ∈ [1, +∞))"""
        # α = path_ratio ∈ [1, 5]
        alpha = torch.rand(4, device=device) * 4.0 + 1.0  # [1, 5]
        alpha_emb = unet.alpha_embedding(alpha)

        assert alpha_emb.shape == (4, unet.time_emb_dim)

    def test_condition_mlp(self, unet, sample_condition):
        """测试条件 MLP"""
        cond_emb = unet.condition_mlp(sample_condition)

        assert cond_emb.shape == (sample_condition.shape[0], unet.time_emb_dim)


class TestNumericalStability:
    """数值稳定性测试"""

    def test_no_nan_output(self, unet, sample_trajectory, sample_condition, sample_alpha, device):
        """测试输出无 NaN"""
        batch_size = sample_trajectory.shape[0]
        t = torch.randint(0, 100, (batch_size,), device=device)

        output = unet(sample_trajectory, t, sample_condition, sample_alpha)

        assert not torch.isnan(output).any(), "输出包含 NaN"
        assert not torch.isinf(output).any(), "输出包含 Inf"

    def test_extreme_inputs(self, unet, device):
        """测试极端输入"""
        batch_size = 4

        # 大值输入
        x_large = torch.randn(batch_size, 100, 2, device=device) * 100
        t = torch.zeros(batch_size, device=device).long()
        cond = torch.randn(batch_size, 4, device=device)
        alpha = torch.rand(batch_size, device=device) * 4.0 + 1.0  # [1, 5]

        output = unet(x_large, t, cond, alpha)
        assert not torch.isnan(output).any(), "大值输入产生 NaN"

        # 小值输入
        x_small = torch.randn(batch_size, 100, 2, device=device) * 0.001
        output = unet(x_small, t, cond, alpha)
        assert not torch.isnan(output).any(), "小值输入产生 NaN"

    def test_extreme_timesteps(self, unet, sample_trajectory, sample_condition, sample_alpha, device):
        """测试极端时间步"""
        batch_size = sample_trajectory.shape[0]

        for t_val in [0, 50, 99]:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            output = unet(sample_trajectory, t, sample_condition, sample_alpha)
            assert not torch.isnan(output).any(), f"t={t_val} 产生 NaN"


class TestCodeQuality:
    """U-Net 代码质量检查"""

    def test_shared_encoder_implementation(self, unet_source):
        """检查 Shared Encoder 长度预测实现"""
        assert '_encode' in unet_source, "未找到 _encode 方法"
        assert 'bottleneck' in unet_source.lower(), "未找到 bottleneck 相关代码"

    def test_length_head_input(self, unet_source):
        """检查 LengthPredictionHead 输入"""
        # 应该包含 encoder_feature
        assert 'encoder_feature' in unet_source or 'encoder_dim' in unet_source, \
            "LengthPredictionHead 应该接收 encoder 特征"

    def test_no_early_fusion(self, unet_source):
        """检查已删除 early_fusion"""
        # early_fusion 模式应该已被移除
        # LengthPredictionHead 不应该只用 condition_emb + alpha_emb
        lines = unet_source.split('\n')
        for i, line in enumerate(lines):
            if 'class LengthPredictionHead' in line:
                # 检查初始化参数
                init_section = '\n'.join(lines[i:i + 30])
                # 应该有 encoder_dim 参数
                assert 'encoder_dim' in init_section, "LengthPredictionHead 应该有 encoder_dim 参数"
                break
