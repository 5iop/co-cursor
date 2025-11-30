"""
Alpha-DDIM 运行时功能测试

测试覆盖:
- 基本采样功能
- Per-sample 长度支持
- 边界条件满足
- 长度预测功能
- 熵控制早停
"""
import pytest
import torch


class TestBasicSampling:
    """基本采样功能测试"""

    def test_sample_shape(self, ddim, sample_condition, device):
        """测试采样输出形状"""
        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                num_inference_steps=10,
                device=device,
            )
        assert trajectories.shape == (4, 100, 2)

    def test_sample_with_fixed_length(self, ddim, sample_condition, device):
        """测试固定长度采样"""
        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                effective_length=50,
                num_inference_steps=10,
                device=device,
            )
        assert trajectories.shape == (4, 100, 2)

    def test_sample_different_alpha(self, ddim, sample_condition, device):
        """测试不同 alpha 值 (论文方案A: path_ratio ∈ [1, +∞))"""
        for alpha in [1.0, 1.5, 2.0, 3.0, 5.0]:
            with torch.no_grad():
                trajectories = ddim.sample(
                    batch_size=2,
                    condition=sample_condition[:2],
                    alpha=alpha,
                    num_inference_steps=5,
                    device=device,
                )
            assert trajectories.shape == (2, 100, 2)


class TestPerSampleLength:
    """Per-sample 长度支持测试"""

    def test_per_sample_length_basic(self, ddim, sample_condition, per_sample_lengths, device):
        """测试 per-sample 长度基本功能"""
        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                effective_length=per_sample_lengths,
                num_inference_steps=10,
                device=device,
            )
        assert trajectories.shape == (4, 100, 2)

    def test_per_sample_length_boundary_start(self, ddim, sample_condition, per_sample_lengths, device):
        """测试 per-sample 长度时起点边界条件"""
        start_points = sample_condition[:, :2]

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                effective_length=per_sample_lengths,
                num_inference_steps=10,
                device=device,
            )

        # 检查起点
        traj_start = trajectories[:, 0, :]
        start_error = (traj_start - start_points).abs().max().item()
        assert start_error < 0.01, f"起点误差过大: {start_error}"

    def test_per_sample_length_boundary_end(self, ddim, sample_condition, per_sample_lengths, device):
        """测试 per-sample 长度时终点边界条件"""
        end_points = sample_condition[:, 2:]
        batch_size = 4

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=batch_size,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                effective_length=per_sample_lengths,
                num_inference_steps=10,
                device=device,
            )

        # 检查每个样本的终点
        for i in range(batch_size):
            end_idx = per_sample_lengths[i].item() - 1
            traj_end = trajectories[i, end_idx, :]
            end_error = (traj_end - end_points[i]).abs().max().item()
            assert end_error < 0.01, f"样本 {i} 终点误差过大: {end_error}, 长度={end_idx+1}"

    def test_per_sample_length_edge_cases(self, ddim, sample_condition, device):
        """测试 per-sample 长度边界情况"""
        # 测试最小长度 (会被 clamp 到 2)
        lengths = torch.tensor([1, 2, 3, 5], device=device)

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                effective_length=lengths,
                num_inference_steps=10,
                device=device,
            )

        assert trajectories.shape == (4, 100, 2)
        # m=1 应该被 clamp 到 2，不应该崩溃


class TestBoundaryConditions:
    """边界条件测试"""

    def test_start_point_constraint(self, ddim, sample_condition, device):
        """测试起点约束"""
        start_points = sample_condition[:, :2]

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                num_inference_steps=10,
                device=device,
            )

        traj_start = trajectories[:, 0, :]
        max_error = (traj_start - start_points).abs().max().item()
        assert max_error < 0.01, f"起点约束未满足，最大误差: {max_error}"

    def test_end_point_constraint(self, ddim, sample_condition, device):
        """测试终点约束 (使用全长)"""
        end_points = sample_condition[:, 2:]

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                num_inference_steps=10,
                device=device,
            )

        traj_end = trajectories[:, -1, :]
        max_error = (traj_end - end_points).abs().max().item()
        assert max_error < 0.01, f"终点约束未满足，最大误差: {max_error}"

    def test_boundary_with_different_lengths(self, ddim, sample_condition, device):
        """测试不同长度下的边界条件"""
        start_points = sample_condition[:, :2]
        end_points = sample_condition[:, 2:]

        for length in [20, 50, 80, 100]:
            with torch.no_grad():
                trajectories = ddim.sample(
                    batch_size=4,
                    condition=sample_condition,
                    alpha=2.0,  # path_ratio=2 (中等复杂度)
                    effective_length=length,
                    num_inference_steps=10,
                    device=device,
                )

            # 检查起点
            start_error = (trajectories[:, 0, :] - start_points).abs().max().item()
            assert start_error < 0.01, f"长度={length} 起点误差: {start_error}"

            # 检查终点
            end_error = (trajectories[:, length - 1, :] - end_points).abs().max().item()
            assert end_error < 0.01, f"长度={length} 终点误差: {end_error}"


class TestLengthPrediction:
    """长度预测功能测试"""

    def test_auto_length_basic(self, ddim, sample_condition, device):
        """测试自动长度预测基本功能"""
        with torch.no_grad():
            trajectories, lengths = ddim.sample_with_auto_length(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                num_inference_steps=10,
                use_per_sample_length=True,
                device=device,
            )

        assert trajectories.shape == (4, 100, 2)
        assert lengths.shape == (4,)
        assert (lengths >= 2).all(), "预测长度应 >= 2"
        assert (lengths <= 100).all(), "预测长度应 <= seq_length"

    def test_auto_length_different_alpha(self, ddim, sample_condition, device):
        """测试不同 alpha 下的长度预测"""
        lengths_by_alpha = {}

        for alpha in [1.5, 2.0, 3.0]:  # path_ratio 范围
            with torch.no_grad():
                _, lengths = ddim.sample_with_auto_length(
                    batch_size=4,
                    condition=sample_condition,
                    alpha=alpha,
                    num_inference_steps=10,
                    device=device,
                )
            lengths_by_alpha[alpha] = lengths.float().mean().item()

        # 长度预测应该工作（不要求单调关系，只要不崩溃）
        for alpha, avg_len in lengths_by_alpha.items():
            assert 2 <= avg_len <= 100, f"alpha={alpha} 平均长度异常: {avg_len}"

    def test_no_length_prediction_fallback(self, ddim_no_length, sample_condition, device):
        """测试模型不支持长度预测时的回退"""
        # 应该抛出错误，因为模型不支持长度预测
        with pytest.raises(RuntimeError, match="does not support length prediction"):
            with torch.no_grad():
                ddim_no_length.sample_with_auto_length(
                    batch_size=4,
                    condition=sample_condition,
                    alpha=2.0,  # path_ratio=2 (中等复杂度)
                    device=device,
                )

    def test_sample_with_auto_length_flag(self, ddim, sample_condition, device):
        """测试 sample() 的 auto_length 参数"""
        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                auto_length=True,
                num_inference_steps=10,
                device=device,
            )

        assert trajectories.shape == (4, 100, 2)


class TestGetLoss:
    """损失计算测试"""

    def test_get_loss_basic(self, ddim, sample_trajectory, sample_condition):
        """测试基本损失计算"""
        result = ddim.get_loss(
            x_0=sample_trajectory,
            condition=sample_condition,
        )

        assert 'ddim_loss' in result
        assert result['ddim_loss'].ndim == 0  # scalar
        assert result['ddim_loss'].item() > 0

    def test_get_loss_with_length(self, ddim, sample_trajectory, sample_condition, per_sample_lengths):
        """测试带长度预测的损失计算"""
        result = ddim.get_loss(
            x_0=sample_trajectory,
            condition=sample_condition,
            length=per_sample_lengths,
            include_length_loss=True,
        )

        assert 'ddim_loss' in result
        assert 'predicted_log_length' in result
        assert 'target_length' in result
        assert result['predicted_log_length'].shape == (4, 1)

    def test_get_loss_without_length(self, ddim_no_length, sample_trajectory, sample_condition, per_sample_lengths):
        """测试不带长度预测模型的损失计算"""
        result = ddim_no_length.get_loss(
            x_0=sample_trajectory,
            condition=sample_condition,
            length=per_sample_lengths,
            include_length_loss=True,
        )

        assert 'ddim_loss' in result
        assert result['predicted_log_length'] is None


class TestNumericalStability:
    """数值稳定性测试"""

    def test_no_nan_in_output(self, ddim, sample_condition, device):
        """测试输出无 NaN"""
        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                num_inference_steps=10,
                device=device,
            )

        assert not torch.isnan(trajectories).any(), "输出包含 NaN"
        assert not torch.isinf(trajectories).any(), "输出包含 Inf"

    def test_extreme_alpha_values(self, ddim, sample_condition, device):
        """测试极端 alpha 值"""
        for alpha in [1.0, 10.0]:  # 极端 path_ratio 值
            with torch.no_grad():
                trajectories = ddim.sample(
                    batch_size=2,
                    condition=sample_condition[:2],
                    alpha=alpha,
                    num_inference_steps=5,
                    device=device,
                )

            assert not torch.isnan(trajectories).any(), f"alpha={alpha} 产生 NaN"

    def test_minimum_length_stability(self, ddim, sample_condition, device):
        """测试最小长度的数值稳定性 (m=2)"""
        # m=1 会被 clamp 到 2
        lengths = torch.tensor([2, 2, 2, 2], device=device)

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=sample_condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                effective_length=lengths,
                num_inference_steps=10,
                device=device,
            )

        assert not torch.isnan(trajectories).any(), "最小长度产生 NaN"

    def test_large_distance_stability(self, ddim, device):
        """测试大距离时的数值稳定性"""
        # 创建大距离的条件
        start_points = torch.zeros(4, 2, device=device)
        end_points = torch.ones(4, 2, device=device) * 1000  # 大距离
        condition = torch.cat([start_points, end_points], dim=1)

        with torch.no_grad():
            trajectories = ddim.sample(
                batch_size=4,
                condition=condition,
                alpha=2.0,  # path_ratio=2 (中等复杂度)
                num_inference_steps=10,
                device=device,
            )

        assert not torch.isnan(trajectories).any(), "大距离产生 NaN"
