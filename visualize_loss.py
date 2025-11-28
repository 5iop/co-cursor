"""
损失函数分布可视化脚本
显示 LDDIM, Lsim, Lstyle 的分布情况
"""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免阻塞

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.losses import DMTGLoss


def generate_trajectory_pairs(n_samples: int, seq_len: int, noise_level: float = 0.02):
    """生成轨迹对用于测试"""
    t = np.linspace(0, 1, seq_len)

    predicted = []
    target = []

    for _ in range(n_samples):
        # 目标轨迹：带曲线的轨迹
        curve_amp = np.random.uniform(0.05, 0.2)
        curve_freq = np.random.uniform(1, 3)
        x_t = t + np.random.randn(seq_len) * noise_level
        y_t = t + curve_amp * np.sin(curve_freq * np.pi * t) + np.random.randn(seq_len) * noise_level
        target.append(np.stack([x_t, y_t], axis=-1))

        # 预测轨迹：相似但有偏差
        pred_noise = np.random.uniform(0.01, 0.1)
        x_p = t + np.random.randn(seq_len) * pred_noise
        y_p = t + curve_amp * 0.8 * np.sin(curve_freq * np.pi * t) + np.random.randn(seq_len) * pred_noise
        predicted.append(np.stack([x_p, y_p], axis=-1))

    return torch.FloatTensor(np.array(predicted)), torch.FloatTensor(np.array(target))


def compute_loss_distribution(n_batches: int = 100, batch_size: int = 32, seq_len: int = 50):
    """计算损失分布"""
    loss_fn = DMTGLoss(
        lambda_ddim=1.0,
        lambda_sim=0.1,
        lambda_style=0.05,
    )

    ddim_losses = []
    sim_losses = []
    style_losses = []
    total_losses = []

    # 路径长度比率
    pred_ratios = []
    target_ratios = []

    for _ in range(n_batches):
        # 生成数据
        predicted_x0, target_x0 = generate_trajectory_pairs(batch_size, seq_len)
        predicted_noise = torch.randn(batch_size, seq_len, 2)
        target_noise = torch.randn(batch_size, seq_len, 2)

        # 生成随机alpha（论文推荐范围 [0.3, 0.8]）
        alpha = 0.3 + 0.5 * torch.rand(batch_size)

        # 计算损失
        losses = loss_fn(predicted_noise, target_noise, predicted_x0, target_x0, alpha=alpha)

        ddim_losses.append(losses['ddim_loss'].item())
        sim_losses.append(losses['similarity_loss'].item())
        style_losses.append(losses['style_loss'].item())
        total_losses.append(losses['total_loss'].item())

        # 计算路径长度比率
        pred_segments = predicted_x0[:, 1:, :] - predicted_x0[:, :-1, :]
        pred_path_length = torch.norm(pred_segments, dim=-1).sum(dim=-1)
        pred_straight_dist = torch.norm(predicted_x0[:, -1, :] - predicted_x0[:, 0, :], dim=-1) + 1e-8
        pred_ratios.extend((pred_path_length / pred_straight_dist).tolist())

        target_segments = target_x0[:, 1:, :] - target_x0[:, :-1, :]
        target_path_length = torch.norm(target_segments, dim=-1).sum(dim=-1)
        target_straight_dist = torch.norm(target_x0[:, -1, :] - target_x0[:, 0, :], dim=-1) + 1e-8
        target_ratios.extend((target_path_length / target_straight_dist).tolist())

    return {
        'ddim': np.array(ddim_losses),
        'sim': np.array(sim_losses),
        'style': np.array(style_losses),
        'total': np.array(total_losses),
        'pred_ratios': np.array(pred_ratios),
        'target_ratios': np.array(target_ratios),
    }


def plot_loss_distribution(results: dict, save_path: str = None):
    """绘制损失分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. DDIM Loss 分布
    ax = axes[0, 0]
    ax.hist(results['ddim'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(results['ddim'].mean(), color='red', linestyle='--', label=f'Mean: {results["ddim"].mean():.4f}')
    ax.set_xlabel('LDDIM (Noise MSE)')
    ax.set_ylabel('Frequency')
    ax.set_title('LDDIM Distribution (Eq. 11)')
    ax.legend()

    # 2. Similarity Loss 分布
    ax = axes[0, 1]
    ax.hist(results['sim'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(results['sim'].mean(), color='red', linestyle='--', label=f'Mean: {results["sim"].mean():.4f}')
    ax.set_xlabel('Lsim (Point MSE)')
    ax.set_ylabel('Frequency')
    ax.set_title('Lsim Distribution (Eq. 12)')
    ax.legend()

    # 3. Style Loss 分布
    ax = axes[0, 2]
    ax.hist(results['style'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(results['style'].mean(), color='red', linestyle='--', label=f'Mean: {results["style"].mean():.4f}')
    ax.set_xlabel('Lstyle (Path Ratio MSE)')
    ax.set_ylabel('Frequency')
    ax.set_title('Lstyle Distribution (Eq. 13)')
    ax.legend()

    # 4. Total Loss 分布
    ax = axes[1, 0]
    ax.hist(results['total'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(results['total'].mean(), color='red', linestyle='--', label=f'Mean: {results["total"].mean():.4f}')
    ax.set_xlabel('Total Loss')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Loss Distribution (Eq. 14)\nL = w1·LDDIM + w2·Lsim + w3·Lstyle')
    ax.legend()

    # 5. 路径长度比率分布
    ax = axes[1, 1]
    ax.hist(results['pred_ratios'], bins=30, alpha=0.5, color='blue', label='Predicted', edgecolor='black')
    ax.hist(results['target_ratios'], bins=30, alpha=0.5, color='green', label='Target (α)', edgecolor='black')
    ax.set_xlabel('Path Length Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Path Length Ratio Distribution\n(path_length / straight_distance)')
    ax.legend()

    # 6. 损失组成比例
    ax = axes[1, 2]
    weighted_ddim = results['ddim'].mean() * 1.0
    weighted_sim = results['sim'].mean() * 0.1
    weighted_style = results['style'].mean() * 0.05

    components = ['w1·LDDIM\n(1.0)', 'w2·Lsim\n(0.1)', 'w3·Lstyle\n(0.05)']
    values = [weighted_ddim, weighted_sim, weighted_style]
    colors = ['blue', 'green', 'orange']

    bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Weighted Loss Value')
    ax.set_title('Loss Components Contribution')

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_loss_vs_complexity(save_path: str = None):
    """绘制损失与轨迹复杂度的关系"""
    loss_fn = DMTGLoss()

    complexities = np.linspace(0.05, 0.5, 20)  # 曲线幅度
    style_losses = []
    sim_losses = []

    seq_len = 50
    batch_size = 32
    t = np.linspace(0, 1, seq_len)

    for amp in complexities:
        # 目标：固定复杂度
        target = []
        for _ in range(batch_size):
            x = t + np.random.randn(seq_len) * 0.01
            y = t + 0.1 * np.sin(2 * np.pi * t) + np.random.randn(seq_len) * 0.01
            target.append(np.stack([x, y], axis=-1))
        target_x0 = torch.FloatTensor(np.array(target))

        # 预测：变化的复杂度
        predicted = []
        for _ in range(batch_size):
            x = t + np.random.randn(seq_len) * 0.01
            y = t + amp * np.sin(2 * np.pi * t) + np.random.randn(seq_len) * 0.01
            predicted.append(np.stack([x, y], axis=-1))
        predicted_x0 = torch.FloatTensor(np.array(predicted))

        # 计算损失（使用固定alpha=0.5作为目标复杂度）
        noise_p = torch.randn(batch_size, seq_len, 2)
        noise_t = torch.randn(batch_size, seq_len, 2)
        alpha = torch.full((batch_size,), 0.5)  # 固定目标复杂度
        losses = loss_fn(noise_p, noise_t, predicted_x0, target_x0, alpha=alpha)

        style_losses.append(losses['style_loss'].item())
        sim_losses.append(losses['similarity_loss'].item())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Style Loss vs Complexity
    ax = axes[0]
    ax.plot(complexities, style_losses, 'o-', color='orange', linewidth=2, markersize=6)
    ax.axvline(0.1, color='green', linestyle='--', alpha=0.7, label='Target complexity (0.1)')
    ax.set_xlabel('Predicted Trajectory Curve Amplitude')
    ax.set_ylabel('Lstyle (Path Ratio MSE)')
    ax.set_title('Lstyle vs Trajectory Complexity\n(Target amp = 0.1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Similarity Loss vs Complexity
    ax = axes[1]
    ax.plot(complexities, sim_losses, 'o-', color='green', linewidth=2, markersize=6)
    ax.axvline(0.1, color='green', linestyle='--', alpha=0.7, label='Target complexity (0.1)')
    ax.set_xlabel('Predicted Trajectory Curve Amplitude')
    ax.set_ylabel('Lsim (Point MSE)')
    ax.set_title('Lsim vs Trajectory Complexity\n(Target amp = 0.1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def print_statistics(results: dict):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("Loss Function Statistics (Paper Eq. 11-14)")
    print("=" * 60)

    print("\n[LDDIM] Noise Prediction MSE (Eq. 11):")
    print(f"  Mean: {results['ddim'].mean():.6f}")
    print(f"  Std:  {results['ddim'].std():.6f}")
    print(f"  Min:  {results['ddim'].min():.6f}")
    print(f"  Max:  {results['ddim'].max():.6f}")

    print("\n[Lsim] Point MSE (Eq. 12):")
    print(f"  Mean: {results['sim'].mean():.6f}")
    print(f"  Std:  {results['sim'].std():.6f}")
    print(f"  Min:  {results['sim'].min():.6f}")
    print(f"  Max:  {results['sim'].max():.6f}")

    print("\n[Lstyle] Path Length Ratio MSE (Eq. 13):")
    print(f"  Mean: {results['style'].mean():.6f}")
    print(f"  Std:  {results['style'].std():.6f}")
    print(f"  Min:  {results['style'].min():.6f}")
    print(f"  Max:  {results['style'].max():.6f}")

    print("\n[Total] Weighted Sum (Eq. 14):")
    print(f"  L = 1.0*LDDIM + 0.1*Lsim + 0.05*Lstyle")
    print(f"  Mean: {results['total'].mean():.6f}")
    print(f"  Std:  {results['total'].std():.6f}")

    print("\n[Path Ratios] Trajectory Complexity:")
    print(f"  Predicted - Mean: {results['pred_ratios'].mean():.4f}, Std: {results['pred_ratios'].std():.4f}")
    print(f"  Target    - Mean: {results['target_ratios'].mean():.4f}, Std: {results['target_ratios'].std():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Computing loss distribution...")
    results = compute_loss_distribution(n_batches=100, batch_size=32, seq_len=50)

    # 打印统计信息
    print_statistics(results)

    # 创建输出目录
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 绘制损失分布图
    print("\nPlotting loss distribution...")
    plot_loss_distribution(results, save_path=str(output_dir / "loss_distribution.png"))

    # 绘制损失与复杂度的关系
    print("\nPlotting loss vs complexity...")
    plot_loss_vs_complexity(save_path=str(output_dir / "loss_vs_complexity.png"))

    print("\nDone!")
