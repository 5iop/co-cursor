"""
Pytest fixtures for DMTG tests
"""
import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import TrajectoryUNet
from src.models.alpha_ddim import AlphaDDIM


# ==================== 设备 Fixtures ====================

@pytest.fixture
def device():
    """返回可用设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 模型 Fixtures ====================

@pytest.fixture
def unet(device):
    """创建 TrajectoryUNet 实例"""
    model = TrajectoryUNet(
        seq_length=100,
        input_dim=2,
        base_channels=32,  # 使用较小的模型加速测试
        enable_length_prediction=True,
    ).to(device)
    return model


@pytest.fixture
def unet_no_length(device):
    """创建不带长度预测的 TrajectoryUNet 实例"""
    model = TrajectoryUNet(
        seq_length=100,
        input_dim=2,
        base_channels=32,
        enable_length_prediction=False,
    ).to(device)
    return model


@pytest.fixture
def ddim(unet, device):
    """创建 AlphaDDIM 实例"""
    ddim = AlphaDDIM(unet, seq_length=100, timesteps=100)
    # 移动 DDIM 内部张量到设备
    ddim.to(device)
    return ddim


@pytest.fixture
def ddim_no_length(unet_no_length, device):
    """创建不带长度预测的 AlphaDDIM 实例"""
    ddim = AlphaDDIM(unet_no_length, seq_length=100, timesteps=100)
    ddim.to(device)
    return ddim


# ==================== 数据 Fixtures ====================

@pytest.fixture
def batch_size():
    """默认 batch size"""
    return 4


@pytest.fixture
def sample_condition(batch_size, device):
    """生成样本条件 (起点 + 终点)"""
    start_points = torch.rand(batch_size, 2, device=device)
    end_points = torch.rand(batch_size, 2, device=device) + 1.0
    return torch.cat([start_points, end_points], dim=1)


@pytest.fixture
def sample_trajectory(batch_size, device):
    """生成样本轨迹"""
    return torch.randn(batch_size, 100, 2, device=device)


@pytest.fixture
def sample_alpha(batch_size, device):
    """生成样本 alpha 值"""
    return torch.rand(batch_size, device=device) * 0.5 + 0.3  # [0.3, 0.8]


@pytest.fixture
def per_sample_lengths(batch_size, device):
    """生成 per-sample 长度"""
    return torch.randint(20, 80, (batch_size,), device=device)


# ==================== 代码文件 Fixtures ====================

@pytest.fixture
def alpha_ddim_source():
    """读取 alpha_ddim.py 源代码"""
    file_path = PROJECT_ROOT / "src" / "models" / "alpha_ddim.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


@pytest.fixture
def alpha_ddim_lines(alpha_ddim_source):
    """返回 alpha_ddim.py 的行列表"""
    return alpha_ddim_source.split('\n')


@pytest.fixture
def unet_source():
    """读取 unet.py 源代码"""
    file_path = PROJECT_ROOT / "src" / "models" / "unet.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


@pytest.fixture
def unet_lines(unet_source):
    """返回 unet.py 的行列表"""
    return unet_source.split('\n')
