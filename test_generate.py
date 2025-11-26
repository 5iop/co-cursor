"""
DMTG 轨迹生成测试脚本
测试训练好的模型生成人类风格鼠标轨迹
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

from src.models.alpha_ddim import create_alpha_ddim, EntropyController


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    model = create_alpha_ddim(seq_length=50, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch}")

    return model


def compute_trajectory_metrics(trajectory: np.ndarray) -> dict:
    """计算轨迹的各种指标"""
    # 路径长度
    segments = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    path_length = np.sum(segment_lengths)

    # 直线距离
    straight_dist = np.linalg.norm(trajectory[-1] - trajectory[0])

    # 路径比率 (复杂度)
    path_ratio = path_length / (straight_dist + 1e-8)

    # 曲率变化 (平滑度)
    if len(segments) > 1:
        # 计算方向变化
        directions = segments / (np.linalg.norm(segments, axis=1, keepdims=True) + 1e-8)
        dot_products = np.sum(directions[:-1] * directions[1:], axis=1)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(dot_products)
        curvature = np.mean(angles)
    else:
        curvature = 0

    # 速度变化
    speed_std = np.std(segment_lengths)

    return {
        'path_length': path_length,
        'straight_dist': straight_dist,
        'path_ratio': path_ratio,
        'curvature': curvature,
        'speed_std': speed_std,
    }


def generate_trajectories(
    model,
    start_points: list,
    end_points: list,
    alphas: list,
    device: str = "cuda",
    num_inference_steps: int = 50,
):
    """批量生成轨迹"""
    results = []

    for (start, end) in zip(start_points, end_points):
        for alpha in alphas:
            condition = torch.tensor(
                [[start[0], start[1], end[0], end[1]]],
                device=device
            )

            with torch.no_grad():
                traj = model.sample(
                    batch_size=1,
                    condition=condition,
                    alpha=alpha,
                    num_inference_steps=num_inference_steps,
                    device=device,
                )

            traj_np = traj[0].cpu().numpy()
            metrics = compute_trajectory_metrics(traj_np)

            results.append({
                'start': start,
                'end': end,
                'alpha': alpha,
                'trajectory': traj_np,
                'metrics': metrics,
            })

    return results


def plot_trajectories_by_alpha(results: list, save_path: str = None):
    """按 alpha 值分组绘制轨迹"""
    alphas = sorted(set(r['alpha'] for r in results))
    n_alphas = len(alphas)

    fig, axes = plt.subplots(1, n_alphas, figsize=(5 * n_alphas, 5))
    if n_alphas == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        alpha_results = [r for r in results if r['alpha'] == alpha]

        for r in alpha_results:
            traj = r['trajectory']
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=80, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=80, zorder=5)

        # 计算平均指标
        avg_ratio = np.mean([r['metrics']['path_ratio'] for r in alpha_results])
        avg_curv = np.mean([r['metrics']['curvature'] for r in alpha_results])

        ax.set_title(f'α={alpha}\nratio={avg_ratio:.2f}, curv={avg_curv:.3f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_trajectories_grid(results: list, save_path: str = None):
    """绘制轨迹网格图"""
    # 按起点-终点对分组
    pairs = list(set((tuple(r['start']), tuple(r['end'])) for r in results))
    alphas = sorted(set(r['alpha'] for r in results))

    n_pairs = len(pairs)
    n_alphas = len(alphas)

    fig, axes = plt.subplots(n_pairs, n_alphas, figsize=(4 * n_alphas, 4 * n_pairs))

    if n_pairs == 1:
        axes = [axes]
    if n_alphas == 1:
        axes = [[ax] for ax in axes]

    for i, (start, end) in enumerate(pairs):
        for j, alpha in enumerate(alphas):
            ax = axes[i][j] if n_pairs > 1 else axes[j]

            # 找到对应的结果
            matching = [r for r in results
                       if tuple(r['start']) == start
                       and tuple(r['end']) == end
                       and r['alpha'] == alpha]

            for r in matching:
                traj = r['trajectory']
                ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
                ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, zorder=5)
                ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, zorder=5)

                # 绘制直线参考
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       'k--', alpha=0.3, linewidth=1)

            if matching:
                ratio = matching[0]['metrics']['path_ratio']
                ax.set_title(f'α={alpha}, ratio={ratio:.2f}')

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_metrics_comparison(results: list, save_path: str = None):
    """绘制指标对比图"""
    alphas = sorted(set(r['alpha'] for r in results))

    # 按 alpha 分组计算指标
    metrics_by_alpha = {}
    for alpha in alphas:
        alpha_results = [r for r in results if r['alpha'] == alpha]
        metrics_by_alpha[alpha] = {
            'path_ratio': [r['metrics']['path_ratio'] for r in alpha_results],
            'curvature': [r['metrics']['curvature'] for r in alpha_results],
            'speed_std': [r['metrics']['speed_std'] for r in alpha_results],
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 路径比率
    ax = axes[0]
    data = [metrics_by_alpha[a]['path_ratio'] for a in alphas]
    bp = ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Path Ratio')
    ax.set_title('Path Ratio vs Alpha')
    ax.grid(True, alpha=0.3)

    # 曲率
    ax = axes[1]
    data = [metrics_by_alpha[a]['curvature'] for a in alphas]
    bp = ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Curvature (rad)')
    ax.set_title('Curvature vs Alpha')
    ax.grid(True, alpha=0.3)

    # 速度变化
    ax = axes[2]
    data = [metrics_by_alpha[a]['speed_std'] for a in alphas]
    bp = ax.boxplot(data, labels=[f'{a}' for a in alphas])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Speed Std')
    ax.set_title('Speed Variation vs Alpha')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def print_metrics_table(results: list):
    """打印指标表格"""
    print("\n" + "=" * 70)
    print("轨迹指标统计")
    print("=" * 70)
    print(f"{'Alpha':<8} {'Path Ratio':<12} {'Curvature':<12} {'Speed Std':<12}")
    print("-" * 70)

    alphas = sorted(set(r['alpha'] for r in results))

    for alpha in alphas:
        alpha_results = [r for r in results if r['alpha'] == alpha]

        ratios = [r['metrics']['path_ratio'] for r in alpha_results]
        curvs = [r['metrics']['curvature'] for r in alpha_results]
        speeds = [r['metrics']['speed_std'] for r in alpha_results]

        print(f"{alpha:<8} {np.mean(ratios):<12.3f} {np.mean(curvs):<12.3f} {np.mean(speeds):<12.4f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test DMTG trajectory generation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples per configuration")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for plots")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")

    # 加载模型
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # 定义测试配置
    # 不同的起点-终点对
    test_pairs = [
        ((0.1, 0.1), (0.9, 0.9)),  # 对角线
        ((0.1, 0.5), (0.9, 0.5)),  # 水平
        ((0.5, 0.1), (0.5, 0.9)),  # 垂直
        ((0.2, 0.8), (0.8, 0.2)),  # 反对角线
    ]

    # 不同的 alpha 值
    alphas = [0.3, 0.5, 0.8]

    # 生成轨迹
    print(f"\nGenerating trajectories...")
    print(f"  - {len(test_pairs)} start-end pairs")
    print(f"  - {len(alphas)} alpha values: {alphas}")
    print(f"  - {args.num_samples} samples each")

    all_results = []

    for start, end in test_pairs:
        for alpha in alphas:
            for _ in range(args.num_samples):
                condition = torch.tensor(
                    [[start[0], start[1], end[0], end[1]]],
                    device=args.device
                )

                with torch.no_grad():
                    traj = model.sample(
                        batch_size=1,
                        condition=condition,
                        alpha=alpha,
                        num_inference_steps=50,
                        device=args.device,
                    )

                traj_np = traj[0].cpu().numpy()
                metrics = compute_trajectory_metrics(traj_np)

                all_results.append({
                    'start': start,
                    'end': end,
                    'alpha': alpha,
                    'trajectory': traj_np,
                    'metrics': metrics,
                })

    print(f"Generated {len(all_results)} trajectories")

    # 打印指标
    print_metrics_table(all_results)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 绘制图表
    print("\nGenerating plots...")

    # 1. 按 alpha 分组的轨迹图
    plot_trajectories_by_alpha(
        all_results,
        save_path=output_dir / f"trajectories_by_alpha_{timestamp}.png"
    )

    # 2. 指标对比图
    plot_metrics_comparison(
        all_results,
        save_path=output_dir / f"metrics_comparison_{timestamp}.png"
    )

    # 3. 单个起点-终点对的详细图
    single_pair_results = [r for r in all_results
                          if r['start'] == (0.1, 0.1) and r['end'] == (0.9, 0.9)]
    if single_pair_results:
        plot_trajectories_by_alpha(
            single_pair_results,
            save_path=output_dir / f"diagonal_trajectories_{timestamp}.png"
        )

    print(f"\nAll plots saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
