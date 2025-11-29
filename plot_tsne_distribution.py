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
import sys
from pathlib import Path
from datetime import datetime

# 解析参数以决定是否使用 Agg 后端（必须在 import matplotlib.pyplot 之前）
_temp_parser = argparse.ArgumentParser(add_help=False)
_temp_parser.add_argument("--no_display", action="store_true")
_temp_args, _ = _temp_parser.parse_known_args()

if _temp_args.no_display:
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端，避免 X11 依赖

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import platform

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
    """加载人类轨迹并提取特征（使用 lazy 模式避免全量加载）"""
    print(f"Loading human trajectories from {data_dir}...")

    # 使用 lazy=True 避免一次性加载所有数据到内存
    dataset = CombinedMouseDataset(
        boun_dir=data_dir if Path(data_dir).exists() else None,
        open_images_dir=data_dir if Path(data_dir).exists() else None,
        max_length=500,
        lazy=True,  # 懒加载模式，按需读取
    )

    # 随机选取索引（避免顺序偏差）
    total_size = len(dataset)
    if total_size > max_samples:
        indices = np.random.choice(total_size, max_samples, replace=False)
    else:
        indices = np.arange(total_size)

    features_list = []
    for i in tqdm(indices, desc="Extracting human features"):
        sample = dataset[i]
        traj = sample['trajectory'].numpy()
        # 使用 dataset 返回的 length 正确截取有效轨迹，而非依赖零检测
        length = int(sample['length'].item())
        traj_valid = traj[:length]
        feat = extract_trajectory_features(traj_valid)
        if feat is not None:
            features_list.append(feat)

    return np.array(features_list)


def generate_model_trajectories(
    model,
    num_samples: int,
    alphas: list,
    device: str,
    use_fixed_length: bool = False,
    batch_size: int = 35,
) -> tuple:
    """
    使用模型批量生成轨迹并提取特征

    Args:
        model: 模型
        num_samples: 总样本数
        alphas: alpha 值列表
        device: 设备
        use_fixed_length: 是否使用固定长度
        batch_size: 批量大小 (默认 32)

    Returns:
        features: (N, n_features) 特征数组
        alpha_labels: (N,) 每个样本对应的 alpha 值
    """
    print(f"Generating {num_samples} model trajectories (batch_size={batch_size})...")

    features_list = []
    alpha_labels_list = []
    samples_per_alpha = num_samples // len(alphas)
    failed_count = 0

    for alpha in alphas:
        # 计算需要多少批次
        num_batches = (samples_per_alpha + batch_size - 1) // batch_size
        generated = 0

        with tqdm(total=samples_per_alpha, desc=f"Alpha={alpha}") as pbar:
            for batch_idx in range(num_batches):
                # 当前批次大小
                current_batch = min(batch_size, samples_per_alpha - generated)
                if current_batch <= 0:
                    break

                # 批量生成随机起终点
                starts = np.random.rand(current_batch, 2) * 0.6 + 0.2  # [0.2, 0.8]
                ends = np.random.rand(current_batch, 2) * 0.6 + 0.2

                conditions = torch.tensor(
                    np.hstack([starts, ends]),
                    device=device,
                    dtype=torch.float32
                )

                try:
                    with torch.no_grad():
                        if use_fixed_length:
                            # 使用固定长度（适用于未训练模型）
                            trajs = model.sample(
                                batch_size=current_batch,
                                condition=conditions,
                                alpha=alpha,
                                num_inference_steps=50,
                                device=device,
                                effective_length=100,
                            )
                            lengths = [100] * current_batch
                        else:
                            trajs, pred_lengths = model.sample_with_auto_length(
                                batch_size=current_batch,
                                condition=conditions,
                                alpha=alpha,
                                num_inference_steps=50,
                                device=device,
                            )
                            lengths = [max(int(l.item()), 10) for l in pred_lengths]

                        # 批量提取特征
                        trajs_np = trajs.cpu().numpy()
                        for i in range(current_batch):
                            m = lengths[i]
                            traj_valid = trajs_np[i, :m]

                            feat = extract_trajectory_features(traj_valid)
                            if feat is not None:
                                features_list.append(feat)
                                alpha_labels_list.append(alpha)
                            else:
                                failed_count += 1

                except Exception as e:
                    failed_count += current_batch
                    if failed_count <= 3:
                        print(f"Warning: Failed to generate batch: {e}")

                generated += current_batch
                pbar.update(current_batch)

    if failed_count > 0:
        print(f"Warning: {failed_count} trajectories failed feature extraction")

    if len(features_list) == 0:
        print("Error: No valid trajectories generated. Try using --use_fixed_length flag.")
        return np.array([]).reshape(0, 10), np.array([])

    return np.array(features_list), np.array(alpha_labels_list)


def plot_tsne_distribution(
    human_features: np.ndarray,
    model_features: np.ndarray,
    save_path: str = None,
    model_name: str = "DMTG",
    label: str = None,
    alpha_labels: np.ndarray = None,
    no_display: bool = False,
):
    """
    绘制 t-SNE 分布图 (论文 Fig. 6 风格)

    左侧: 模型分布（不同 alpha 用不同颜色）
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

    # 创建图表（单图）
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 添加总标题（如果有 label）
    if label:
        fig.suptitle(f'实验: {label}', fontsize=16, fontweight='bold')

    # 设置颜色
    human_color = '#2E86AB'  # 蓝色
    # 不同 alpha 的颜色映射
    alpha_colors = {
        0.3: '#FF9999',  # 粉红
        0.5: '#F18F01',  # 橙色
        0.7: '#C73E1D',  # 深红
        0.8: '#A23B72',  # 紫色
    }
    default_model_color = '#E94F37'

    # 模型分布 (叠加人类分布作为参考)
    ax.scatter(human_emb[:, 0], human_emb[:, 1], c=human_color, alpha=0.2, s=10, label='Human (ref)')

    if alpha_labels is not None and len(alpha_labels) == len(model_emb):
        # 按不同 alpha 值分别绘制
        unique_alphas = sorted(set(alpha_labels))
        for alpha in unique_alphas:
            mask = alpha_labels == alpha
            color = alpha_colors.get(alpha, default_model_color)
            ax.scatter(model_emb[mask, 0], model_emb[mask, 1],
                      c=color, alpha=0.6, s=15, label=f'α={alpha}')
    else:
        ax.scatter(model_emb[:, 0], model_emb[:, 1], c=default_model_color, alpha=0.6, s=15, label=model_name)

    ax.set_title(f'{model_name} vs Human 分布', fontsize=14)
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 设置坐标范围
    all_x = embeddings[:, 0]
    all_y = embeddings[:, 1]
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    plt.tight_layout()
    if label:
        plt.subplots_adjust(top=0.9)  # 给 suptitle 留出空间

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if no_display:
        plt.close(fig)
    else:
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
    parser.add_argument("--output_dir", "-o", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--label", "-l", type=str, default=None,
                       help="Label for this experiment (shown in title and appended to filename)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_fixed_length", action="store_true",
                       help="Use fixed length (100) for generation (useful for untrained models)")
    parser.add_argument("--no_display", action="store_true",
                       help="Don't display the plot (useful for background execution)")

    args = parser.parse_args()

    # 创建输出目录和文件名（与 test_generate.py 格式一致）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.label:
        filename = f"tsne_{args.label}.png"
    else:
        filename = f"tsne_{timestamp}.png"
    output_path = output_dir / filename

    print("=" * 60)
    print("t-SNE Distribution Plot (Paper Fig. 6 Style)")
    print("=" * 60)
    if args.label:
        print(f"Label: {args.label}")
    print(f"Device: {args.device}")
    print(f"Human samples: {args.num_human}")
    print(f"Model samples: {args.num_model}")
    print(f"Alphas: {args.alphas}")
    print(f"Output: {output_path}")

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
    model_features, alpha_labels = generate_model_trajectories(
        model, args.num_model, args.alphas, args.device,
        use_fixed_length=args.use_fixed_length,
    )

    # 绘制 t-SNE 分布图
    plot_tsne_distribution(
        human_features,
        model_features,
        save_path=str(output_path),
        model_name="DMTG",
        label=args.label,
        alpha_labels=alpha_labels,
        no_display=args.no_display,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
