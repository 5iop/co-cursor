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
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import CombinedMouseDataset
from src.models.alpha_ddim import create_alpha_ddim
from src.evaluation.metrics import TrajectoryMetrics, ClassifierMetrics
from src.evaluation.generator import TrajectoryGenerator, BezierGenerator, LinearGenerator
from src.evaluation.visualize import TrajectoryVisualizer, create_comparison_report


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_alpha_ddim(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_real_trajectories(
    sapimouse_dir: str = None,
    boun_dir: str = None,
    max_samples: int = 1000,
    seq_length: int = 50,
) -> np.ndarray:
    """加载真实轨迹数据"""
    dataset = CombinedMouseDataset(
        sapimouse_dir=sapimouse_dir,
        boun_dir=boun_dir,
        seq_length=seq_length,
        max_samples=max_samples,
    )

    trajectories = np.array(dataset.all_trajectories)
    return trajectories


def generate_trajectories(
    generator: TrajectoryGenerator,
    real_trajectories: np.ndarray,
    alpha: float = 0.5,
    num_samples: int = None,
) -> np.ndarray:
    """使用模型生成轨迹"""
    if num_samples is None:
        num_samples = len(real_trajectories)

    # 使用真实轨迹的起点和终点作为条件
    indices = np.random.choice(len(real_trajectories), num_samples, replace=False)
    start_points = real_trajectories[indices, 0, :]  # (n, 2)
    end_points = real_trajectories[indices, -1, :]  # (n, 2)

    print(f"Generating {num_samples} trajectories with alpha={alpha}...")
    generated = generator.generate_batch(
        start_points=start_points,
        end_points=end_points,
        alpha=alpha,
        normalized_input=True,
        return_normalized=True,
    )

    return generated


def evaluate_distribution(
    generated: np.ndarray,
    real: np.ndarray,
) -> dict:
    """评估分布相似性"""
    metrics = TrajectoryMetrics()

    print("Computing distribution metrics...")
    results = metrics.compare_distributions(generated, real)

    return results


def evaluate_classifiers(
    generated: np.ndarray,
    real: np.ndarray,
) -> dict:
    """使用分类器评估"""
    clf_metrics = ClassifierMetrics()

    print("Training classifiers...")
    results = clf_metrics.evaluate_with_classifiers(generated, real)

    return results


def compare_baselines(
    real_trajectories: np.ndarray,
    dmtg_generator: TrajectoryGenerator,
    num_samples: int = 100,
    alpha: float = 0.5,
) -> dict:
    """与基线方法对比"""
    results = {}

    # 选择样本
    indices = np.random.choice(len(real_trajectories), num_samples, replace=False)
    start_points = real_trajectories[indices, 0, :]
    end_points = real_trajectories[indices, -1, :]

    # 1. DMTG
    print("Generating DMTG trajectories...")
    dmtg_trajectories = dmtg_generator.generate_batch(
        start_points, end_points,
        alpha=alpha,
        normalized_input=True,
        return_normalized=True,
    )
    results['DMTG'] = dmtg_trajectories

    # 2. Bezier
    print("Generating Bezier trajectories...")
    bezier_trajectories = []
    for start, end in zip(start_points, end_points):
        traj = BezierGenerator.generate(tuple(start), tuple(end), num_points=50)
        bezier_trajectories.append(traj)
    results['Bezier'] = np.array(bezier_trajectories)

    # 3. Linear
    print("Generating Linear trajectories...")
    linear_trajectories = []
    for start, end in zip(start_points, end_points):
        traj = LinearGenerator.generate(tuple(start), tuple(end), num_points=50, noise_level=0.01)
        linear_trajectories.append(traj)
    results['Linear'] = np.array(linear_trajectories)

    return results


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
        default="datasets/boun-mouse-dynamics-dataset",
        help="BOUN数据集目录"
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
        default=0.5,
        help="生成alpha参数"
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
    real_trajectories = load_real_trajectories(
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
    generated_trajectories = generate_trajectories(
        generator,
        real_trajectories,
        alpha=args.alpha,
        num_samples=args.num_samples,
    )

    # 评估分布
    print("\n" + "=" * 40)
    print("Distribution Metrics")
    print("=" * 40)
    dist_results = evaluate_distribution(
        generated_trajectories,
        real_trajectories[:args.num_samples]
    )
    for name, value in dist_results.items():
        print(f"  {name}: {value:.6f}")

    # 分类器评估
    print("\n" + "=" * 40)
    print("Classifier Evaluation")
    print("=" * 40)
    clf_results = evaluate_classifiers(
        generated_trajectories,
        real_trajectories[:args.num_samples]
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

        baseline_trajectories = compare_baselines(
            real_trajectories,
            generator,
            num_samples=min(100, args.num_samples),
            alpha=args.alpha,
        )

        baseline_results = {}
        for method_name, trajectories in baseline_trajectories.items():
            print(f"\n  Evaluating {method_name}...")
            metrics = evaluate_distribution(
                trajectories,
                real_trajectories[:len(trajectories)]
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

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("\nEvaluation complete!")


# 需要导入matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    main()
