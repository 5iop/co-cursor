"""
DMTG评估脚本
评估生成轨迹与真实轨迹的相似性
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import orjson

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import CombinedMouseDataset
from src.models.alpha_ddim import create_alpha_ddim
from src.evaluation.metrics import TrajectoryMetrics, ClassifierMetrics
from src.evaluation.generator import TrajectoryGenerator, BezierGenerator, LinearGenerator
from src.evaluation.visualize import TrajectoryVisualizer, create_comparison_report


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从检查点读取模型配置（如果有），否则使用默认值
    config = checkpoint.get('model_config', {})
    seq_length = config.get('seq_length', 500)
    timesteps = config.get('timesteps', 1000)
    input_dim = config.get('input_dim', 3)
    base_channels = config.get('base_channels', 96)
    enable_length_prediction = config.get('enable_length_prediction', False)

    # 创建与训练时相同配置的模型
    from src.models.unet import TrajectoryUNet
    unet = TrajectoryUNet(
        seq_length=seq_length,
        input_dim=input_dim,
        base_channels=base_channels,
        enable_length_prediction=enable_length_prediction,
    )
    model = AlphaDDIM(
        model=unet,
        timesteps=timesteps,
        seq_length=seq_length,
        input_dim=input_dim,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model: base_channels={base_channels}, length_pred={enable_length_prediction}")
    return model


def load_real_trajectories(
    sapimouse_dir: str = None,
    boun_dir: str = None,
    max_samples: int = 1000,
    seq_length: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载真实轨迹数据

    Returns:
        trajectories: (N, seq_length, 2) padded轨迹数组
        masks: (N, seq_length) 有效位置掩码
        lengths: (N,) 每条轨迹的真实长度
        start_points: (N, 2) 真实起点（非padding）
        end_points: (N, 2) 真实终点（非padding）
    """
    dataset = CombinedMouseDataset(
        sapimouse_dir=sapimouse_dir,
        boun_dir=boun_dir,
        max_length=seq_length,
        max_samples=max_samples,
    )

    # 分别收集轨迹、掩码、长度、起点、终点
    trajectories = []
    masks = []
    lengths = []
    start_points = []
    end_points = []

    for i in range(len(dataset)):
        item = dataset[i]
        trajectories.append(item['trajectory'].numpy())
        masks.append(item['mask'].numpy())
        lengths.append(item['length'].numpy()[0])
        start_points.append(item['start_point'].numpy())
        end_points.append(item['end_point'].numpy())

    return (
        np.stack(trajectories),
        np.stack(masks),
        np.array(lengths),
        np.stack(start_points),
        np.stack(end_points),
    )


def estimate_generated_masks(
    trajectories: np.ndarray,
    end_points: np.ndarray,
    threshold: float = 0.02,
    min_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """估计生成轨迹的有效长度和mask

    基于轨迹何时首次到达终点附近来估计有效长度。
    这解决了训练时使用mask但评估时不使用导致的指标失真问题。

    Args:
        trajectories: (N, seq_len, 2) 生成的轨迹
        end_points: (N, 2) 目标终点
        threshold: 判定到达终点的距离阈值（归一化坐标）
        min_length: 最小有效长度

    Returns:
        masks: (N, seq_len) 有效位置掩码
        lengths: (N,) 估计的有效长度
    """
    batch_size, seq_len, _ = trajectories.shape
    masks = np.zeros((batch_size, seq_len), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        traj = trajectories[i]  # (seq_len, 2)
        end_pt = end_points[i]  # (2,)

        # 计算每个点到终点的距离
        distances = np.linalg.norm(traj - end_pt, axis=1)

        # 找到第一次到达终点附近的位置
        reached_indices = np.where(distances < threshold)[0]

        if len(reached_indices) > 0:
            # 使用第一次到达终点的位置作为有效长度
            effective_length = max(reached_indices[0] + 1, min_length)
        else:
            # 如果从未到达终点，使用整个轨迹
            effective_length = seq_len

        effective_length = min(effective_length, seq_len)
        lengths[i] = effective_length
        masks[i, :effective_length] = 1.0

    return masks, lengths


def generate_trajectories(
    generator: TrajectoryGenerator,
    start_points: np.ndarray,
    end_points: np.ndarray,
    alpha: float = 1.5,
    num_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用模型生成轨迹

    Args:
        generator: 轨迹生成器
        start_points: (N, 2) 真实起点
        end_points: (N, 2) 真实终点
        alpha: 复杂度参数
        num_samples: 生成样本数

    Returns:
        generated: (N, seq_len, 2) 生成的轨迹
        masks: (N, seq_len) 估计的有效位置掩码
        selected_ends: (N, 2) 使用的终点（用于mask估计）
    """
    if num_samples is None:
        num_samples = len(start_points)

    # 随机选择样本（确保不超过可用样本数）
    actual_samples = min(num_samples, len(start_points))
    if actual_samples < num_samples:
        print(f"Warning: Requested {num_samples} samples but only {len(start_points)} available")
    indices = np.random.choice(len(start_points), actual_samples, replace=False)
    selected_starts = start_points[indices]
    selected_ends = end_points[indices]

    print(f"Generating {actual_samples} trajectories with alpha={alpha}...")
    generated = generator.generate_batch(
        start_points=selected_starts,
        end_points=selected_ends,
        alpha=alpha,
        normalized_input=True,
        return_normalized=True,
    )

    # 估计生成轨迹的有效mask
    masks, lengths = estimate_generated_masks(generated, selected_ends)
    avg_length = lengths.mean()
    print(f"  Estimated average trajectory length: {avg_length:.1f}")

    return generated, masks, selected_ends


def evaluate_distribution(
    generated: np.ndarray,
    real: np.ndarray,
    generated_masks: np.ndarray = None,
    real_masks: np.ndarray = None,
) -> dict:
    """评估分布相似性"""
    metrics = TrajectoryMetrics()

    print("Computing distribution metrics...")
    results = metrics.compare_distributions(
        generated, real,
        generated_masks=generated_masks,
        real_masks=real_masks,
    )

    return results


def evaluate_classifiers(
    generated: np.ndarray,
    real: np.ndarray,
    generated_masks: np.ndarray = None,
    real_masks: np.ndarray = None,
) -> dict:
    """使用分类器评估"""
    clf_metrics = ClassifierMetrics()

    print("Training classifiers...")
    results = clf_metrics.evaluate_with_classifiers(
        generated, real,
        generated_masks=generated_masks,
        real_masks=real_masks,
    )

    return results


def compare_baselines(
    start_points: np.ndarray,
    end_points: np.ndarray,
    dmtg_generator: TrajectoryGenerator,
    num_samples: int = 100,
    alpha: float = 1.5,
    seq_length: int = 500,
) -> Tuple[dict, dict, np.ndarray]:
    """与基线方法对比

    Args:
        start_points: (N, 2) 真实起点
        end_points: (N, 2) 真实终点
        dmtg_generator: DMTG轨迹生成器
        num_samples: 生成样本数
        alpha: 复杂度参数
        seq_length: 轨迹序列长度（与DMTG保持一致）

    Returns:
        trajectories: 各方法生成的轨迹 {method_name: (N, seq_len, 2)}
        masks: 各方法的轨迹mask {method_name: (N, seq_len)}
        selected_ends: 使用的终点 (N, 2)
    """
    trajectories = {}
    masks = {}

    # 选择样本（确保不超过可用样本数）
    actual_samples = min(num_samples, len(start_points))
    if actual_samples < num_samples:
        print(f"Warning: Requested {num_samples} samples but only {len(start_points)} available")
    indices = np.random.choice(len(start_points), actual_samples, replace=False)
    selected_starts = start_points[indices]
    selected_ends = end_points[indices]

    # 1. DMTG
    print("Generating DMTG trajectories...")
    dmtg_trajectories = dmtg_generator.generate_batch(
        selected_starts, selected_ends,
        alpha=alpha,
        normalized_input=True,
        return_normalized=True,
    )
    trajectories['DMTG'] = dmtg_trajectories
    dmtg_masks, _ = estimate_generated_masks(dmtg_trajectories, selected_ends)
    masks['DMTG'] = dmtg_masks

    # 2. Bezier（与DMTG使用相同序列长度，全长有效）
    print("Generating Bezier trajectories...")
    bezier_trajectories = []
    for start, end in zip(selected_starts, selected_ends):
        traj = BezierGenerator.generate(tuple(start), tuple(end), num_points=seq_length)
        bezier_trajectories.append(traj)
    trajectories['Bezier'] = np.array(bezier_trajectories)
    masks['Bezier'] = np.ones((actual_samples, seq_length), dtype=np.float32)

    # 3. Linear（与DMTG使用相同序列长度，全长有效）
    print("Generating Linear trajectories...")
    linear_trajectories = []
    for start, end in zip(selected_starts, selected_ends):
        traj = LinearGenerator.generate(tuple(start), tuple(end), num_points=seq_length, noise_level=0.01)
        linear_trajectories.append(traj)
    trajectories['Linear'] = np.array(linear_trajectories)
    masks['Linear'] = np.ones((actual_samples, seq_length), dtype=np.float32)

    return trajectories, masks, selected_ends


def main():
    parser = argparse.ArgumentParser(description="Evaluate DMTG model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="模型检查点路径"
    )
    parser.add_argument(
        "--sapimouse_dir",
        type=str,
        default="datasets/sapimouse",
        help="SapiMouse数据集目录"
    )
    parser.add_argument(
        "--boun_dir",
        type=str,
        default="datasets/boun-processed",
        help="BOUN数据集目录（自动检测Parquet或JSONL格式）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="评估样本数"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="生成alpha参数 (path_ratio ∈ [1, +∞), 1=直线, 1.5=默认)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="跳过基线对比"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DMTG Evaluation")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据集
    base_dir = Path(__file__).parent
    sapimouse_dir = base_dir / args.sapimouse_dir
    boun_dir = base_dir / args.boun_dir

    sapimouse_path = str(sapimouse_dir) if sapimouse_dir.exists() else None
    boun_path = str(boun_dir) if boun_dir.exists() else None

    if sapimouse_path is None and boun_path is None:
        print("Error: No dataset found!")
        return

    # 加载真实轨迹
    print("\nLoading real trajectories...")
    real_trajectories, real_masks, real_lengths, real_start_points, real_end_points = load_real_trajectories(
        sapimouse_dir=sapimouse_path,
        boun_dir=boun_path,
        max_samples=args.num_samples * 2,
    )
    print(f"Loaded {len(real_trajectories)} real trajectories")

    # 创建生成器
    checkpoint_path = base_dir / args.checkpoint
    if checkpoint_path.exists():
        print(f"\nLoading model from {checkpoint_path}...")
        model = load_model(str(checkpoint_path), args.device)
        generator = TrajectoryGenerator(model=model, device=args.device)
    else:
        print("\nNo checkpoint found, using untrained model...")
        generator = TrajectoryGenerator(device=args.device)

    # 生成轨迹
    generated_trajectories, generated_masks, _ = generate_trajectories(
        generator,
        real_start_points,
        real_end_points,
        alpha=args.alpha,
        num_samples=args.num_samples,
    )

    # 评估分布
    print("\n" + "=" * 40)
    print("Distribution Metrics")
    print("=" * 40)
    num_eval = len(generated_trajectories)
    dist_results = evaluate_distribution(
        generated_trajectories,
        real_trajectories[:num_eval],
        generated_masks=generated_masks,
        real_masks=real_masks[:num_eval],
    )
    for name, value in dist_results.items():
        print(f"  {name}: {value:.6f}")

    # 分类器评估
    print("\n" + "=" * 40)
    print("Classifier Evaluation")
    print("=" * 40)
    clf_results = evaluate_classifiers(
        generated_trajectories,
        real_trajectories[:num_eval],
        generated_masks=generated_masks,
        real_masks=real_masks[:num_eval],
    )
    for clf_name, scores in clf_results.items():
        print(f"\n  {clf_name}:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")

    # 基线对比
    if not args.skip_baselines:
        print("\n" + "=" * 40)
        print("Baseline Comparison")
        print("=" * 40)

        baseline_num = min(100, args.num_samples)
        baseline_trajectories, baseline_masks, _ = compare_baselines(
            real_start_points,
            real_end_points,
            generator,
            num_samples=baseline_num,
            alpha=args.alpha,
        )

        baseline_results = {}
        for method_name, trajectories in baseline_trajectories.items():
            print(f"\n  Evaluating {method_name}...")
            method_mask = baseline_masks[method_name]
            metrics = evaluate_distribution(
                trajectories,
                real_trajectories[:len(trajectories)],
                generated_masks=method_mask,
                real_masks=real_masks[:len(trajectories)],
            )
            baseline_results[method_name] = metrics

            # 打印关键指标
            print(f"    Speed JSD: {metrics.get('speed_jsd', 'N/A'):.6f}")
            print(f"    Acceleration JSD: {metrics.get('acceleration_jsd', 'N/A'):.6f}")

    # 可视化
    print("\n" + "=" * 40)
    print("Generating Visualizations")
    print("=" * 40)

    visualizer = TrajectoryVisualizer()

    # 轨迹对比图
    fig = visualizer.plot_multiple_trajectories(
        [generated_trajectories[0], real_trajectories[0]],
        labels=['Generated (DMTG)', 'Real'],
        title='Generated vs Real Trajectory',
    )
    fig.savefig(output_dir / 'trajectory_comparison.png')
    plt.close(fig)

    # 速度曲线
    fig = visualizer.plot_velocity_profile(
        generated_trajectories[0],
        title='Generated Trajectory Velocity Profile'
    )
    fig.savefig(output_dir / 'generated_velocity_profile.png')
    plt.close(fig)

    # 保存结果
    results = {
        'distribution_metrics': dist_results,
        'classifier_results': {
            clf: {k: float(v) for k, v in scores.items()}
            for clf, scores in clf_results.items()
        },
        'parameters': {
            'alpha': args.alpha,
            'num_samples': args.num_samples,
        }
    }

    with open(output_dir / 'evaluation_results.json', 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    print(f"\nResults saved to {output_dir}")
    print("\nEvaluation complete!")


# 需要导入matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    main()
