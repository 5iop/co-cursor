"""
生成论文 Fig. 6 风格的 t-SNE 分布图
比较模型生成的轨迹与人类轨迹的分布差异

使用轨迹特征进行 t-SNE 降维可视化：
- 路径长度比率 (path_ratio)
- 平均曲率 (curvature)
- 速度变化 (speed_std)
- 方向变化 (direction_change)
- 起终点距离 (straight_dist)
- 轨迹长度 (length)
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
import platform
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.alpha_ddim import create_alpha_ddim
from src.data.dataset import CombinedMouseDataset


def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def extract_trajectory_features(trajectory: np.ndarray) -> np.ndarray:
    """
    从轨迹中提取特征向量

    Args:
        trajectory: (seq_len, 2) 轨迹坐标

    Returns:
        features: (n_features,) 特征向量
    """
    # 移除零填充
    non_zero_mask = ~np.all(trajectory == 0, axis=1)
    if non_zero_mask.sum() < 2:
        return None
    traj = trajectory[non_zero_mask]

    if len(traj) < 3:
        return None

    # 1. 路径长度
    segments = traj[1:] - traj[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    path_length = np.sum(segment_lengths)

    # 2. 起终点直线距离
    straight_dist = np.linalg.norm(traj[-1] - traj[0]) + 1e-8

    # 3. 路径比率
    path_ratio = path_length / straight_dist

    # 4. 曲率 (方向变化的平均值)
    if len(segments) >= 2:
        # 计算相邻段的夹角
        angles = []
        for i in range(len(segments) - 1):
            v1 = segments[i]
            v2 = segments[i + 1]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        curvature = np.mean(angles) if angles else 0
        curvature_std = np.std(angles) if angles else 0
    else:
        curvature = 0
        curvature_std = 0

    # 5. 速度变化 (段长度的标准差)
    speed_std = np.std(segment_lengths)
    speed_mean = np.mean(segment_lengths)

    # 6. 轨迹长度 (点数)
    traj_length = len(traj)

    # 7. 方向熵 (方向分布的均匀程度)
    if len(segments) > 0:
        directions = np.arctan2(segments[:, 1], segments[:, 0])
        # 分成8个方向bin
        hist, _ = np.histogram(directions, bins=8, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-8)
        direction_entropy = -np.sum(hist * np.log(hist + 1e-8))
    else:
        direction_entropy = 0

    # 8. 边界框面积
    bbox_width = traj[:, 0].max() - traj[:, 0].min()
    bbox_height = traj[:, 1].max() - traj[:, 1].min()
    bbox_area = bbox_width * bbox_height

    # 9. 复杂度 (β/(β+1))
    beta = path_ratio - 1.0
    complexity = beta / (beta + 1.0 + 1e-8)

    features = np.array([
        path_ratio,
        curvature,
        curvature_std,
        speed_std,
        speed_mean,
        np.log(traj_length + 1),  # log变换
        direction_entropy,
        bbox_area,
        complexity,
        straight_dist,
    ])

    return features


def load_human_trajectories(data_dir: str, max_samples: int = 1000) -> np.ndarray:
    """加载人类轨迹并提取特征"""
    print(f"Loading human trajectories from {data_dir}...")

    dataset = CombinedMouseDataset(
        boun_dir=data_dir if Path(data_dir).exists() else None,
        open_images_dir=data_dir if Path(data_dir).exists() else None,
        max_length=500,
        max_samples=max_samples,
    )

    features_list = []
    for i in tqdm(range(min(len(dataset), max_samples)), desc="Extracting human features"):
        sample = dataset[i]
        traj = sample['trajectory'].numpy()
        feat = extract_trajectory_features(traj)
        if feat is not None:
            features_list.append(feat)

    return np.array(features_list)


def generate_model_trajectories(
    model,
    num_samples: int,
    alphas: list,
    device: str,
    use_fixed_length: bool = False,
) -> np.ndarray:
    """使用模型生成轨迹并提取特征"""
    print(f"Generating {num_samples} model trajectories...")

    features_list = []
    samples_per_alpha = num_samples // len(alphas)
    failed_count = 0

    for alpha in alphas:
        for _ in tqdm(range(samples_per_alpha), desc=f"Alpha={alpha}"):
            # 随机起终点
            start = np.random.rand(2) * 0.6 + 0.2  # [0.2, 0.8]
            end = np.random.rand(2) * 0.6 + 0.2

            condition = torch.tensor(
                [[start[0], start[1], end[0], end[1]]],
                device=device,
                dtype=torch.float32
            )

            try:
                with torch.no_grad():
                    if use_fixed_length:
                        # 使用固定长度（适用于未训练模型）
                        traj = model.sample(
                            batch_size=1,
                            condition=condition,
                            alpha=alpha,
                            num_inference_steps=50,
                            device=device,
                            effective_length=100,  # 固定100点
                        )
                        m = 100
                    else:
                        traj, pred_lengths = model.sample_with_auto_length(
                            batch_size=1,
                            condition=condition,
                            alpha=alpha,
                            num_inference_steps=50,
                            device=device,
                        )
                        m = int(pred_lengths[0].item())
                        m = max(m, 10)  # 最少10个点

                    traj_np = traj[0].cpu().numpy()
                    traj_valid = traj_np[:m]

                    feat = extract_trajectory_features(traj_valid)
                    if feat is not None:
                        features_list.append(feat)
                    else:
                        failed_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:
                    print(f"Warning: Failed to generate trajectory: {e}")

    if failed_count > 0:
        print(f"Warning: {failed_count} trajectories failed feature extraction")

    if len(features_list) == 0:
        print("Error: No valid trajectories generated. Try using --use_fixed_length flag.")
        return np.array([]).reshape(0, 10)

    return np.array(features_list)


def plot_tsne_distribution(
    human_features: np.ndarray,
    model_features: np.ndarray,
    save_path: str = None,
    model_name: str = "DMTG",
):
    """
    绘制 t-SNE 分布图 (论文 Fig. 6 风格)

    左侧: 模型分布
    右侧: 人类分布
    """
    # 验证输入
    if len(human_features) == 0:
        print("Error: No human features available. Cannot create t-SNE plot.")
        return
    if len(model_features) == 0:
        print("Error: No model features available. Cannot create t-SNE plot.")
        print("Hint: Try running with --use_fixed_length if using untrained model.")
        return

    print(f"Human samples: {len(human_features)}, Model samples: {len(model_features)}")

    setup_chinese_font()

    print("Computing t-SNE embedding...")

    # 合并特征进行 t-SNE
    all_features = np.vstack([human_features, model_features])

    # 标准化
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8
    all_features_norm = (all_features - mean) / std

    # t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(all_features) // 4),
        random_state=42,
        max_iter=1000,
    )
    embeddings = tsne.fit_transform(all_features_norm)

    # 分离人类和模型的嵌入
    n_human = len(human_features)
    human_emb = embeddings[:n_human]
    model_emb = embeddings[n_human:]

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 设置颜色
    human_color = '#2E86AB'  # 蓝色
    model_color = '#E94F37'  # 红色

    # 左侧: 模型分布 (叠加人类分布作为参考)
    ax = axes[0]
    ax.scatter(human_emb[:, 0], human_emb[:, 1], c=human_color, alpha=0.2, s=10, label='Human (reference)')
    ax.scatter(model_emb[:, 0], model_emb[:, 1], c=model_color, alpha=0.6, s=15, label=model_name)
    ax.set_title(f'{model_name} 分布', fontsize=14)
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 右侧: 人类分布
    ax = axes[1]
    ax.scatter(human_emb[:, 0], human_emb[:, 1], c=human_color, alpha=0.6, s=15, label='Human')
    ax.set_title('人类轨迹分布', fontsize=14)
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 统一坐标范围
    all_x = embeddings[:, 0]
    all_y = embeddings[:, 1]
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    for ax in axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    # 计算分布重叠度量
    print("\n分布统计:")
    print(f"  人类样本数: {len(human_features)}")
    print(f"  模型样本数: {len(model_features)}")

    # 计算中心距离
    human_center = human_emb.mean(axis=0)
    model_center = model_emb.mean(axis=0)
    center_dist = np.linalg.norm(human_center - model_center)
    print(f"  分布中心距离: {center_dist:.4f}")

    # 计算覆盖率 (模型点落在人类凸包内的比例)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(human_emb)
        from matplotlib.path import Path
        hull_path = Path(human_emb[hull.vertices])
        inside = hull_path.contains_points(model_emb)
        coverage = inside.mean()
        print(f"  模型覆盖率 (落在人类凸包内): {coverage:.2%}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE distribution plot (Paper Fig. 6)")

    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model.pt",
                       help="Model checkpoint path (default: checkpoints/best_model.pt)")
    parser.add_argument("--human_data", type=str, default="datasets/boun-processed",
                       help="Human trajectory data directory")
    parser.add_argument("--num_human", type=int, default=500,
                       help="Number of human samples")
    parser.add_argument("--num_model", type=int, default=500,
                       help="Number of model samples")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                       help="Alpha values for generation")
    parser.add_argument("--output", "-o", type=str, default="outputs/tsne_distribution.png",
                       help="Output path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_fixed_length", action="store_true",
                       help="Use fixed length (100) for generation (useful for untrained models)")

    args = parser.parse_args()

    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("t-SNE Distribution Plot (Paper Fig. 6 Style)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Human samples: {args.num_human}")
    print(f"Model samples: {args.num_model}")
    print(f"Alphas: {args.alphas}")

    # 加载模型
    print("\nLoading model...")
    model = create_alpha_ddim(seq_length=500, timesteps=1000, device=args.device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using untrained model (random weights)")
        if not args.use_fixed_length:
            print("Hint: Consider using --use_fixed_length for untrained models")

    model.eval()

    # 加载人类轨迹
    base_dir = Path(__file__).parent
    human_data_path = base_dir / args.human_data

    if human_data_path.exists():
        human_features = load_human_trajectories(str(human_data_path), args.num_human)
    else:
        print(f"Warning: Human data not found at {human_data_path}")
        print("Generating synthetic human data for demo...")
        # 生成模拟的人类数据用于演示
        human_features = []
        for _ in range(args.num_human):
            # 模拟人类轨迹特征
            feat = np.array([
                np.random.lognormal(0.3, 0.3),  # path_ratio
                np.random.exponential(0.2),      # curvature
                np.random.exponential(0.1),      # curvature_std
                np.random.exponential(0.02),     # speed_std
                np.random.exponential(0.01),     # speed_mean
                np.random.uniform(3, 6),         # log(length)
                np.random.uniform(1, 2),         # direction_entropy
                np.random.uniform(0.01, 0.3),    # bbox_area
                np.random.beta(2, 5),            # complexity
                np.random.uniform(0.1, 1.0),     # straight_dist
            ])
            human_features.append(feat)
        human_features = np.array(human_features)

    # 生成模型轨迹
    model_features = generate_model_trajectories(
        model, args.num_model, args.alphas, args.device,
        use_fixed_length=args.use_fixed_length,
    )

    # 绘制 t-SNE 分布图
    plot_tsne_distribution(
        human_features,
        model_features,
        save_path=str(output_path),
        model_name="DMTG",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
