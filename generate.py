"""
DMTG轨迹生成脚本
使用训练好的模型生成鼠标轨迹
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import orjson

sys.path.insert(0, str(Path(__file__).parent))

from src.models.alpha_ddim import create_alpha_ddim
from src.evaluation.generator import TrajectoryGenerator
from src.evaluation.visualize import TrajectoryVisualizer


def main():
    parser = argparse.ArgumentParser(description="Generate mouse trajectories with DMTG")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="模型检查点路径"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="100,100",
        help="起点坐标 (x,y)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="500,400",
        help="终点坐标 (x,y)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="熵控制参数 (0-1, 越大轨迹越复杂)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="生成轨迹数量"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="可视化轨迹"
    )
    parser.add_argument(
        "--add_timing",
        action="store_true",
        help="添加时间信息"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=500,
        help="轨迹总时长（毫秒）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # 解析坐标
    start_point = tuple(map(float, args.start.split(',')))
    end_point = tuple(map(float, args.end.split(',')))

    print("=" * 50)
    print("DMTG Trajectory Generator")
    print("=" * 50)
    print(f"Start: {start_point}")
    print(f"End: {end_point}")
    print(f"Alpha: {args.alpha}")
    print(f"Device: {args.device}")

    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"\nLoading model from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location=args.device)

        # 从检查点读取模型配置（如果有），否则使用默认值
        model_config = checkpoint.get('model_config', {})
        seq_length = model_config.get('seq_length', 500)
        timesteps = model_config.get('timesteps', 1000)
        base_channels = model_config.get('base_channels', 64)

        model = create_alpha_ddim(
            seq_length=seq_length,
            timesteps=timesteps,
            base_channels=base_channels,
            device=args.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        generator = TrajectoryGenerator(model=model, device=args.device)
        print(f"  Model config: seq_length={seq_length}, timesteps={timesteps}, base_channels={base_channels}")
    else:
        print("\nNo checkpoint found, using untrained model...")
        generator = TrajectoryGenerator(device=args.device)

    # 生成轨迹
    print(f"\nGenerating {args.num_samples} trajectory(s)...")

    trajectories = []
    for i in range(args.num_samples):
        trajectory = generator.generate(
            start_point=start_point,
            end_point=end_point,
            alpha=args.alpha,
            num_inference_steps=args.num_inference_steps,
        )

        if args.add_timing:
            trajectory = generator.add_timing(
                trajectory,
                total_duration_ms=args.duration,
                human_like=True
            )

        trajectories.append(trajectory)
        print(f"  Generated trajectory {i+1}: {len(trajectory)} points")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_samples == 1:
        trajectory_data = trajectories[0].tolist()
    else:
        trajectory_data = [t.tolist() for t in trajectories]

    result = {
        'trajectories': trajectory_data,
        'parameters': {
            'start_point': list(start_point),
            'end_point': list(end_point),
            'alpha': args.alpha,
            'num_inference_steps': args.num_inference_steps,
            'duration_ms': args.duration if args.add_timing else None,
        }
    }

    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))

    print(f"\nTrajectory saved to {output_path}")

    # 可视化
    if args.visualize:
        print("\nGenerating visualization...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        visualizer = TrajectoryVisualizer()

        if args.num_samples == 1:
            traj_2d = trajectories[0][:, :2] if trajectories[0].shape[1] > 2 else trajectories[0]
            fig = visualizer.plot_single_trajectory(
                traj_2d,
                title=f"Generated Trajectory (α={args.alpha})",
                show_velocity=True,
            )
        else:
            trajs_2d = [t[:, :2] if t.shape[1] > 2 else t for t in trajectories]
            fig = visualizer.plot_multiple_trajectories(
                trajs_2d,
                labels=[f"Trajectory {i+1}" for i in range(len(trajectories))],
                title=f"Generated Trajectories (α={args.alpha})",
            )

        viz_path = output_path.with_suffix('.png')
        fig.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved to {viz_path}")

        # 速度曲线
        if args.num_samples == 1:
            traj_2d = trajectories[0][:, :2] if trajectories[0].shape[1] > 2 else trajectories[0]
            fig = visualizer.plot_velocity_profile(
                traj_2d,
                title="Velocity Profile"
            )
            vel_path = output_path.with_name(output_path.stem + '_velocity.png')
            fig.savefig(vel_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Velocity profile saved to {vel_path}")

    print("\nDone!")


def generate_demo():
    """生成演示轨迹"""
    print("Generating demo trajectories...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = TrajectoryGenerator(device=device)

    # 生成不同alpha值的轨迹
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    trajectories = []

    for alpha in alphas:
        traj = generator.generate(
            start_point=(100, 100),
            end_point=(500, 400),
            alpha=alpha,
        )
        trajectories.append(traj)
        print(f"  Alpha={alpha}: generated {len(traj)} points")

    # 可视化
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    visualizer = TrajectoryVisualizer()
    fig = visualizer.plot_alpha_comparison(
        trajectories,
        alphas,
        title="Effect of Alpha Parameter on Trajectory Complexity"
    )
    fig.savefig('demo_alpha_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nDemo saved to demo_alpha_comparison.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        generate_demo()
    else:
        main()
