"""
生成轨迹长度和时间特征的 t-SNE 分布图
比较模型生成的轨迹与人类轨迹在时间维度上的分布差异

使用时间相关特征进行 t-SNE 降维可视化：
- 轨迹长度 (length)
- 总时间 (total_time_ms)
- 平均帧间隔 (mean_dt_ms)
- 帧间隔标准差 (dt_std_ms)
- 帧间隔分布熵 (dt_entropy)
- 速度统计 (speed_mean, speed_std) - 使用真实时间计算
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import platform

sys.path.insert(0, str(Path(__file__).parent))

import apprise

from src.models.alpha_ddim import create_alpha_ddim
from src.data.dataset import CombinedMouseDataset, denormalize_dt


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


def extract_temporal_features(trajectory: np.ndarray, length: int = None) -> np.ndarray:
    """
    从轨迹中提取时间相关特征

    Args:
        trajectory: (seq_len, 3) 轨迹坐标 [x, y, dt_norm]
        length: 有效轨迹长度，如果为 None 则自动检测

    Returns:
        features: (n_features,) 特征向量，如果无效返回 None
    """
    if trajectory.shape[1] < 3:
        return None  # 需要 dt 维度

    # 获取有效轨迹
    if length is not None:
        traj = trajectory[:length]
    else:
        # 移除零填充
        non_zero_mask = ~np.all(trajectory[:, :2] == 0, axis=1)
        if non_zero_mask.sum() < 2:
            return None
        traj = trajectory[non_zero_mask]

    if len(traj) < 3:
        return None

    # 提取各维度
    traj_xy = traj[:, :2]
    dt_norm = traj[:, 2]

    # 反归一化得到真实毫秒值
    # 注意: 生成的 dt_norm 可能超出 [0, 1] 范围，需要 clip 防止 overflow
    dt_norm_clipped = np.clip(dt_norm, 0, 1)
    dt_ms = denormalize_dt(dt_norm_clipped)

    # ========== 时间特征 ==========
    # 1. 轨迹长度 (点数)
    traj_length = len(traj)

    # 2. 总时间 (ms)
    total_time_ms = np.sum(dt_ms)

    # 3. 平均帧间隔 (ms) - 跳过第一个点 (dt[0] 通常为 0)
    mean_dt_ms = np.mean(dt_ms[1:]) if len(dt_ms) > 1 else 0

    # 4. 帧间隔标准差 (ms)
    dt_std_ms = np.std(dt_ms[1:]) if len(dt_ms) > 1 else 0

    # 5. 帧间隔分布熵 (衡量时间分布的均匀程度)
    if len(dt_ms) > 1:
        # 分成 10 个 bin (0-100ms 每 10ms 一个 bin，>100ms 为一个 bin)
        dt_clipped = np.clip(dt_ms[1:], 0, 100)
        hist, _ = np.histogram(dt_clipped, bins=10, range=(0, 100))
        hist = hist / (hist.sum() + 1e-8)
        dt_entropy = -np.sum(hist * np.log(hist + 1e-8))
    else:
        dt_entropy = 0

    # 6. dt 中位数 (更稳健的中心趋势)
    dt_median_ms = np.median(dt_ms[1:]) if len(dt_ms) > 1 else 0

    # ========== 速度特征 (使用真实时间) ==========
    # 计算段长度
    segments = traj_xy[1:] - traj_xy[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)  # 像素

    # 计算速度 (像素/ms)
    dt_valid = dt_ms[1:]
    dt_valid = np.where(dt_valid < 1e-3, 1e-3, dt_valid)  # 避免除零
    speeds = segment_lengths / dt_valid  # 像素/ms

    # 7. 平均速度 (像素/ms)
    speed_mean = np.mean(speeds)

    # 8. 速度标准差 (像素/ms)
    speed_std = np.std(speeds)

    # 9. 速度变异系数 (CV = std/mean)
    speed_cv = speed_std / (speed_mean + 1e-8)

    # ========== 几何特征 (用于对比) ==========
    # 10. 路径长度
    path_length = np.sum(segment_lengths)

    # 11. 起终点直线距离
    straight_dist = np.linalg.norm(traj_xy[-1] - traj_xy[0]) + 1e-8

    # 12. 路径比率
    path_ratio = path_length / straight_dist

    features = np.array([
        np.log(traj_length + 1),      # 0: log(长度)
        np.log(total_time_ms + 1),    # 1: log(总时间)
        mean_dt_ms,                    # 2: 平均帧间隔
        dt_std_ms,                     # 3: 帧间隔标准差
        dt_entropy,                    # 4: 帧间隔熵
        dt_median_ms,                  # 5: 帧间隔中位数
        speed_mean,                    # 6: 平均速度
        speed_std,                     # 7: 速度标准差
        speed_cv,                      # 8: 速度变异系数
        np.log(path_length + 1),      # 9: log(路径长度)
        np.log(straight_dist + 1),    # 10: log(直线距离)
        path_ratio,                    # 11: 路径比率
    ])

    return features


FEATURE_NAMES = [
    'log(Length)', 'log(TotalTime)', 'MeanDt(ms)', 'DtStd(ms)',
    'DtEntropy', 'MedianDt(ms)', 'SpeedMean', 'SpeedStd',
    'SpeedCV', 'log(PathLen)', 'log(StraightDist)', 'PathRatio'
]


def load_human_trajectories(
    boun_dir: str = None,
    open_images_dir: str = None,
    max_samples: int = 1000,
) -> tuple:
    """
    加载人类轨迹并提取时间特征

    Args:
        boun_dir: BOUN 数据集目录
        open_images_dir: Open Images 数据集目录
        max_samples: 最大样本数

    Returns:
        features: (N, n_features) 特征数组
        sources: (N,) 每个样本的来源标签
    """
    print(f"Loading human trajectories...")
    if boun_dir:
        print(f"  BOUN: {boun_dir}")
    if open_images_dir:
        print(f"  Open Images: {open_images_dir}")

    boun_path = Path(boun_dir) if boun_dir else None
    open_images_path = Path(open_images_dir) if open_images_dir else None

    dataset = CombinedMouseDataset(
        boun_dir=str(boun_path) if boun_path and boun_path.exists() else None,
        open_images_dir=str(open_images_path) if open_images_path and open_images_path.exists() else None,
        max_length=500,
        lazy=True,
    )

    total_size = len(dataset)
    if total_size == 0:
        print("Warning: No human trajectories found!")
        return np.array([]), np.array([])

    if total_size > max_samples:
        indices = np.random.choice(total_size, max_samples, replace=False)
    else:
        indices = np.arange(total_size)

    features_list = []
    sources_list = []
    for i in tqdm(indices, desc="Extracting human temporal features"):
        sample = dataset[i]
        traj = sample['trajectory'].numpy()
        length = int(sample['length'].item())
        feat = extract_temporal_features(traj, length)
        if feat is not None:
            features_list.append(feat)
            sources_list.append(sample.get('source', 'unknown'))

    print(f"  Extracted {len(features_list)} valid samples")
    return np.array(features_list), np.array(sources_list)


def generate_model_trajectories(
    model,
    num_samples: int,
    alphas: list,
    device: str,
    batch_size: int = 32,
) -> tuple:
    """
    使用模型批量生成轨迹并提取时间特征

    Args:
        model: AlphaDDIM 模型
        num_samples: 总样本数
        alphas: alpha 值列表
        device: 设备
        batch_size: 批量大小

    Returns:
        features: (N, n_features) 特征数组
        alpha_labels: (N,) 每个样本对应的 alpha 值
    """
    print(f"Generating {num_samples} model trajectories (batch_size={batch_size})...")

    features_list = []
    alpha_labels_list = []
    samples_per_alpha = num_samples // len(alphas)

    for alpha in alphas:
        num_batches = (samples_per_alpha + batch_size - 1) // batch_size
        generated = 0

        with tqdm(total=samples_per_alpha, desc=f"Alpha={alpha}") as pbar:
            for batch_idx in range(num_batches):
                current_batch = min(batch_size, samples_per_alpha - generated)
                if current_batch <= 0:
                    break

                # 随机起终点
                start_x = np.random.uniform(100, 1820, current_batch)
                start_y = np.random.uniform(100, 980, current_batch)
                end_x = np.random.uniform(100, 1820, current_batch)
                end_y = np.random.uniform(100, 980, current_batch)

                conditions = torch.tensor(
                    np.column_stack([start_x, start_y, end_x, end_y]),
                    dtype=torch.float32
                ).to(device)

                # 批量生成
                with torch.no_grad():
                    trajectories = model.sample(
                        batch_size=current_batch,
                        condition=conditions,
                        alpha=alpha,
                        num_inference_steps=50,
                        auto_length=True,
                        device=device,
                    )

                # 提取特征
                for traj in trajectories.cpu().numpy():
                    feat = extract_temporal_features(traj)
                    if feat is not None:
                        features_list.append(feat)
                        alpha_labels_list.append(alpha)
                        generated += 1

                pbar.update(current_batch)

    print(f"  Generated {len(features_list)} valid samples")
    return np.array(features_list), np.array(alpha_labels_list)


def plot_tsne_temporal(
    human_features: np.ndarray,
    human_sources: np.ndarray,
    model_features: np.ndarray,
    model_alphas: np.ndarray,
    output_path: str = None,
    show: bool = True,
):
    """
    绘制时间特征的 t-SNE 分布图和特征分布直方图（合并在一张图中）

    Args:
        human_features: 人类轨迹特征
        human_sources: 人类轨迹来源
        model_features: 模型生成轨迹特征
        model_alphas: 模型 alpha 值
        output_path: 输出文件路径
        show: 是否显示图形
    """
    from matplotlib.gridspec import GridSpec

    print("Running t-SNE...")

    # 合并特征
    all_features = np.vstack([human_features, model_features])

    # 标准化
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # t-SNE 降维 (scikit-learn >= 1.2 使用 max_iter 代替 n_iter)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(all_features_scaled)

    # 分割结果
    n_human = len(human_features)
    human_tsne = tsne_result[:n_human]
    model_tsne = tsne_result[n_human:]

    # ========== 绘图 (使用 GridSpec 布局) ==========
    # 布局: 左侧 t-SNE 大图，右侧 3 列 x 4 行 = 12 个特征分布图
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1.5, 1, 1, 1], hspace=0.3, wspace=0.3)

    # ===== 左侧: t-SNE 散点图 (占据左边一列，4 行高) =====
    ax_tsne = fig.add_subplot(gs[:, 0])

    # 为每个 alpha 使用不同颜色 (使用 tab10 色板，更易区分)
    unique_alphas = np.unique(model_alphas)
    alpha_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_alphas), 10)))[:len(unique_alphas)]

    # 人类 vs 生成 (按 alpha 着色)
    ax_tsne.scatter(human_tsne[:, 0], human_tsne[:, 1],
                    c='gray', alpha=0.4, s=15, label='Human', zorder=1)

    for i, alpha in enumerate(unique_alphas):
        mask = model_alphas == alpha
        ax_tsne.scatter(model_tsne[mask, 0], model_tsne[mask, 1],
                        c=[alpha_colors[i]], alpha=0.6, s=25, label=f'Gen α={alpha}', zorder=2)

    ax_tsne.set_xlabel('t-SNE 1')
    ax_tsne.set_ylabel('t-SNE 2')
    ax_tsne.set_title('Temporal Features: Human vs Generated', fontsize=12)
    ax_tsne.legend(loc='upper right', fontsize=9)
    ax_tsne.grid(True, alpha=0.3)

    # ===== 右侧: 12 个特征分布图 (3 列 x 4 行) =====
    n_features = human_features.shape[1]
    for i in range(n_features):
        row = i // 3
        col = (i % 3) + 1  # 从第 1 列开始 (第 0 列是 t-SNE)
        ax = fig.add_subplot(gs[row, col])

        ax.hist(human_features[:, i], bins=30, alpha=0.5, label='Human', density=True)
        ax.hist(model_features[:, i], bins=30, alpha=0.5, label='Generated', density=True)
        ax.set_title(FEATURE_NAMES[i], fontsize=10)
        if i == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.suptitle('Temporal Feature Analysis: t-SNE & Distributions', fontsize=14, y=0.98)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def print_feature_statistics(human_features: np.ndarray, model_features: np.ndarray):
    """打印特征统计对比"""
    print("\n" + "=" * 80)
    print("Feature Statistics: Human vs Generated")
    print("=" * 80)
    print(f"{'Feature':<20} {'Human Mean':>12} {'Human Std':>12} {'Gen Mean':>12} {'Gen Std':>12}")
    print("-" * 80)

    for i, name in enumerate(FEATURE_NAMES):
        h_mean = np.mean(human_features[:, i])
        h_std = np.std(human_features[:, i])
        g_mean = np.mean(model_features[:, i])
        g_std = np.std(model_features[:, i])
        print(f"{name:<20} {h_mean:>12.3f} {h_std:>12.3f} {g_mean:>12.3f} {g_std:>12.3f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Plot t-SNE for temporal features (length, dt)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples per category")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for plots")
    parser.add_argument("--boun_dir", type=str, default="datasets/boun-processed",
                        help="BOUN dataset directory")
    parser.add_argument("--open_images_dir", type=str, default="datasets/open_images_v6",
                        help="Open Images dataset directory")
    parser.add_argument("--alphas", type=str, default="1.0,1.5,2.0,3.0",
                        help="Comma-separated alpha values")
    parser.add_argument("--no_display", action="store_true",
                        help="Don't display the plot")
    parser.add_argument("--webhook", type=str, default=None,
                        help="Webhook URL to send result images")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for output filename")
    args = parser.parse_args()

    setup_chinese_font()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    print(f"Device: {args.device}")

    # 解析 alpha 值
    alphas = [float(a) for a in args.alphas.split(',')]
    print(f"Alpha values: {alphas}")

    # 加载模型
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # 创建模型
    model = create_alpha_ddim(seq_length=500, timesteps=1000, input_dim=3, device=args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)  # 确保模型在正确设备上
    model.model.eval()

    # 从 checkpoint 读取配置 (用于显示)
    model_config = checkpoint.get('model_config', {})
    seq_length = model_config.get('seq_length', 500)
    input_dim = model_config.get('input_dim', 3)
    print(f"Model loaded: seq_length={seq_length}, input_dim={input_dim}")

    # 加载人类轨迹
    human_features, human_sources = load_human_trajectories(
        boun_dir=args.boun_dir,
        open_images_dir=args.open_images_dir,
        max_samples=args.num_samples,
    )

    if len(human_features) == 0:
        print("Error: No human trajectories loaded!")
        return

    # 生成模型轨迹
    model_features, model_alphas = generate_model_trajectories(
        model=model,
        num_samples=args.num_samples,
        alphas=alphas,
        device=args.device,
    )

    if len(model_features) == 0:
        print("Error: No model trajectories generated!")
        return

    # 打印统计
    print_feature_statistics(human_features, model_features)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.label:
        filename = f"tsne_temporal_{args.label}_{timestamp}.png"
    else:
        filename = f"tsne_temporal_{timestamp}.png"
    output_path = str(output_dir / filename)

    # 绘制合并图 (t-SNE + 特征分布)
    plot_tsne_temporal(
        human_features=human_features,
        human_sources=human_sources,
        model_features=model_features,
        model_alphas=model_alphas,
        output_path=output_path,
        show=not args.no_display,
    )

    # 发送 webhook
    if args.webhook:
        print("\n正在发送通知到 webhook...")
        apobj = apprise.Apprise()
        apobj.add(args.webhook)
        success = apobj.notify(
            title=f"DMTG Temporal t-SNE - {args.label or 'distribution'}",
            body=f"Samples: {len(human_features)} human, {len(model_features)} generated",
            attach=output_path,
        )
        print("通知发送成功!" if success else "通知发送失败")


if __name__ == "__main__":
    main()
