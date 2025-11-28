"""
测试短间距、小alpha的轨迹生成
验证 effective_length 是否正确工作
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.alpha_ddim import create_alpha_ddim


def test_short_trajectory():
    """测试短间距轨迹生成"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建模型（未训练，仅测试结构）
    model = create_alpha_ddim(
        seq_length=500,
        timesteps=1000,
        device=device
    )
    model.eval()

    # 测试配置
    test_configs = [
        # (起点, 终点, alpha, effective_length, 描述)
        ((0.1, 0.1), (0.2, 0.2), 0.1, None, "短间距, α=0.1, 无effective_length"),
        ((0.1, 0.1), (0.2, 0.2), 0.1, 50, "短间距, α=0.1, m=50"),
        ((0.1, 0.1), (0.2, 0.2), 0.1, 100, "短间距, α=0.1, m=100"),
        ((0.1, 0.1), (0.5, 0.5), 0.3, 100, "中等间距, α=0.3, m=100"),
        ((0.1, 0.1), (0.9, 0.9), 0.5, None, "长间距, α=0.5, 无effective_length"),
    ]

    results = []

    for start, end, alpha, eff_len, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"测试: {desc}")
        print(f"  起点: {start}, 终点: {end}")
        print(f"  alpha: {alpha}, effective_length: {eff_len}")

        # 构建条件
        condition = torch.FloatTensor([
            start[0], start[1], end[0], end[1]
        ]).unsqueeze(0).to(device)

        # 生成轨迹
        with torch.no_grad():
            trajectory = model.sample(
                batch_size=1,
                condition=condition,
                num_inference_steps=50,
                alpha=alpha,
                device=device,
                effective_length=eff_len,
            )

        traj = trajectory[0].cpu().numpy()

        # 分析轨迹
        # 找到非零点的数量
        non_zero_mask = ~np.all(traj == 0, axis=1)
        non_zero_count = np.sum(non_zero_mask)

        # 计算实际起点和终点
        actual_start = traj[0]
        if eff_len:
            actual_end = traj[eff_len - 1]
        else:
            actual_end = traj[-1]

        # 计算路径长度和直线距离
        if eff_len:
            valid_traj = traj[:eff_len]
        else:
            valid_traj = traj

        segments = valid_traj[1:] - valid_traj[:-1]
        path_length = np.sum(np.linalg.norm(segments, axis=1))
        straight_dist = np.linalg.norm(np.array(end) - np.array(start))
        ratio = path_length / (straight_dist + 1e-8)

        print(f"\n  结果:")
        print(f"    非零点数: {non_zero_count}")
        print(f"    实际起点: ({actual_start[0]:.4f}, {actual_start[1]:.4f})")
        print(f"    实际终点: ({actual_end[0]:.4f}, {actual_end[1]:.4f})")
        print(f"    目标起点: {start}")
        print(f"    目标终点: {end}")
        print(f"    路径长度: {path_length:.4f}")
        print(f"    直线距离: {straight_dist:.4f}")
        print(f"    路径比率: {ratio:.2f}")

        # 检查边界是否正确
        start_error = np.linalg.norm(actual_start - np.array(start))
        end_error = np.linalg.norm(actual_end - np.array(end))
        print(f"    起点误差: {start_error:.6f}")
        print(f"    终点误差: {end_error:.6f}")

        results.append({
            'desc': desc,
            'trajectory': traj,
            'effective_length': eff_len,
            'start': start,
            'end': end,
            'alpha': alpha,
            'ratio': ratio,
        })

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, r in enumerate(results):
        if i >= len(axes):
            break
        ax = axes[i]
        traj = r['trajectory']
        eff_len = r['effective_length']

        if eff_len:
            valid_traj = traj[:eff_len]
            ax.plot(valid_traj[:, 0], valid_traj[:, 1], 'b-', alpha=0.7, label='有效轨迹')
            # 显示padding部分
            if eff_len < len(traj):
                padding_traj = traj[eff_len:]
                ax.scatter(padding_traj[:, 0], padding_traj[:, 1], c='gray', s=1, alpha=0.3, label='padding')
        else:
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7)

        # 标记起点和终点
        ax.scatter([r['start'][0]], [r['start'][1]], c='green', s=100, marker='o', label='目标起点', zorder=5)
        ax.scatter([r['end'][0]], [r['end'][1]], c='red', s=100, marker='o', label='目标终点', zorder=5)

        # 标记实际起点和终点
        ax.scatter([traj[0, 0]], [traj[0, 1]], c='lime', s=50, marker='x', label='实际起点', zorder=6)
        if eff_len:
            ax.scatter([traj[eff_len-1, 0]], [traj[eff_len-1, 1]], c='orange', s=50, marker='x', label='实际终点', zorder=6)
        else:
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], c='orange', s=50, marker='x', label='实际终点', zorder=6)

        ax.set_title(f"{r['desc']}\nratio={r['ratio']:.2f}", fontsize=9)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "test_short_trajectory.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n图像已保存到: {output_path}")
    plt.close()


if __name__ == "__main__":
    test_short_trajectory()
