"""
DMTG配置文件
包含所有超参数和配置选项
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class DataConfig:
    """数据配置"""
    sapimouse_dir: str = "datasets/sapimouse"
    boun_dir: str = "datasets/boun-mouse-dynamics-dataset"
    seq_length: int = 50
    screen_size: Tuple[int, int] = (1920, 1080)
    min_trajectory_length: int = 10
    normalize: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    # U-Net参数
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    time_emb_dim: int = 128
    condition_dim: int = 4  # start(2) + end(2)
    num_heads: int = 4
    attention_levels: Tuple[bool, ...] = (False, True, True)

    # 扩散参数
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # "cosine" or "linear"

    # 输入参数
    seq_length: int = 50
    input_dim: int = 2


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    num_workers: int = 4

    # 损失权重
    lambda_sim: float = 1.0
    lambda_style: float = 0.5
    lambda_boundary: float = 2.0
    lambda_human: float = 0.3

    # 优化器参数
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # 检查点
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 10


@dataclass
class GenerationConfig:
    """生成配置"""
    num_inference_steps: int = 50
    default_alpha: float = 0.5
    eta: float = 0.0  # DDIM随机性参数


@dataclass
class EvaluationConfig:
    """评估配置"""
    num_samples: int = 500
    test_size: float = 0.2
    output_dir: str = "results"

    # 分类器列表
    classifiers: List[str] = field(default_factory=lambda: [
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "MLP",
    ])


@dataclass
class DMTGConfig:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    device: str = "cuda"
    seed: int = 42


def get_default_config() -> DMTGConfig:
    """获取默认配置"""
    return DMTGConfig()


# 预定义配置
CONFIGS = {
    "default": DMTGConfig(),
    "small": DMTGConfig(
        model=ModelConfig(base_channels=32, channel_mults=(1, 2)),
        training=TrainingConfig(batch_size=32, num_epochs=50),
    ),
    "large": DMTGConfig(
        model=ModelConfig(base_channels=128, channel_mults=(1, 2, 4, 8)),
        training=TrainingConfig(batch_size=128, num_epochs=200),
    ),
}


if __name__ == "__main__":
    # 打印默认配置
    config = get_default_config()
    print("Default DMTG Configuration:")
    print(f"  Data: {config.data}")
    print(f"  Model: {config.model}")
    print(f"  Training: {config.training}")
    print(f"  Generation: {config.generation}")
    print(f"  Evaluation: {config.evaluation}")
