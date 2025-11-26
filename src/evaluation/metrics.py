"""
DMTG评估指标模块
实现论文中的白盒测试指标
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error


class DistributionMetrics:
    """
    分布度量指标
    用于比较生成轨迹和真实轨迹的分布相似性
    """

    @staticmethod
    def compute_jsd(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
        """
        Jensen-Shannon散度
        衡量两个分布的相似性，值越小越相似
        """
        # 计算直方图
        all_data = np.concatenate([p, q])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)

        p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

        # 避免零值
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10

        # 归一化
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()

        return jensenshannon(p_hist, q_hist) ** 2

    @staticmethod
    def compute_emd(p: np.ndarray, q: np.ndarray) -> float:
        """
        Earth Mover's Distance (Wasserstein距离)
        衡量两个分布之间的"搬运"代价
        """
        return wasserstein_distance(p, q)

    @staticmethod
    def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
        """均方误差"""
        return mean_squared_error(target, pred)

    @staticmethod
    def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
        """均方根误差"""
        return np.sqrt(mean_squared_error(target, pred))

    @staticmethod
    def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)


class TrajectoryMetrics:
    """
    轨迹特征度量
    提取和比较轨迹的各种统计特征
    """

    def __init__(self):
        self.dist_metrics = DistributionMetrics()

    def extract_features(self, trajectory: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取轨迹特征

        Args:
            trajectory: 轨迹数组 (seq_len, 2) 或 (batch, seq_len, 2)

        Returns:
            特征字典
        """
        if trajectory.ndim == 2:
            trajectory = trajectory[np.newaxis, :]

        features = {}

        # 1. 速度特征
        velocity = np.diff(trajectory, axis=1)  # (batch, seq_len-1, 2)
        speed = np.linalg.norm(velocity, axis=-1)  # (batch, seq_len-1)
        features['speed'] = speed.flatten()
        features['speed_mean'] = np.mean(speed, axis=-1)
        features['speed_std'] = np.std(speed, axis=-1)

        # 2. 加速度特征
        acceleration = np.diff(velocity, axis=1)  # (batch, seq_len-2, 2)
        accel_magnitude = np.linalg.norm(acceleration, axis=-1)
        features['acceleration'] = accel_magnitude.flatten()
        features['acceleration_mean'] = np.mean(accel_magnitude, axis=-1)

        # 3. 曲率特征
        curvature = self._compute_curvature(trajectory)
        features['curvature'] = curvature.flatten()
        features['curvature_mean'] = np.mean(curvature, axis=-1)

        # 4. 方向变化
        direction_change = self._compute_direction_change(velocity)
        features['direction_change'] = direction_change.flatten()

        # 5. 轨迹长度
        path_length = np.sum(speed, axis=-1)
        features['path_length'] = path_length

        # 6. 直线距离
        straight_distance = np.linalg.norm(
            trajectory[:, -1, :] - trajectory[:, 0, :],
            axis=-1
        )
        features['straight_distance'] = straight_distance

        # 7. 效率（直线距离/路径长度）
        features['efficiency'] = straight_distance / (path_length + 1e-8)

        # 8. 方向性加速度（论文中的关键特征）
        directional_accel = self._compute_directional_acceleration(trajectory)
        features.update(directional_accel)

        return features

    def _compute_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """计算轨迹曲率"""
        dx = np.diff(trajectory[:, :, 0], axis=1)
        dy = np.diff(trajectory[:, :, 1], axis=1)

        ddx = np.diff(dx, axis=1)
        ddy = np.diff(dy, axis=1)

        numerator = np.abs(dx[:, :-1] * ddy - dy[:, :-1] * ddx)
        denominator = (dx[:, :-1] ** 2 + dy[:, :-1] ** 2) ** 1.5 + 1e-8

        return numerator / denominator

    def _compute_direction_change(self, velocity: np.ndarray) -> np.ndarray:
        """计算方向变化角度"""
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True) + 1e-8
        unit_velocity = velocity / speed

        # 计算相邻方向向量的点积
        dot_product = np.sum(
            unit_velocity[:, 1:, :] * unit_velocity[:, :-1, :],
            axis=-1
        )
        dot_product = np.clip(dot_product, -1, 1)

        return np.arccos(dot_product)

    def _compute_directional_acceleration(
        self,
        trajectory: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        计算方向性加速度
        论文关键发现：人类在不同方向上的加速度分布不同
        """
        velocity = np.diff(trajectory, axis=1)
        acceleration = np.diff(velocity, axis=1)

        # 水平和垂直加速度
        accel_x = acceleration[:, :, 0]
        accel_y = acceleration[:, :, 1]

        # 按方向分类
        # 向右移动时的加速度
        moving_right = velocity[:, :-1, 0] > 0
        # 向左移动
        moving_left = velocity[:, :-1, 0] < 0
        # 向上移动
        moving_up = velocity[:, :-1, 1] < 0  # y轴向下为正
        # 向下移动
        moving_down = velocity[:, :-1, 1] > 0

        features = {
            'accel_right': np.abs(accel_x[moving_right]).mean() if moving_right.any() else 0,
            'accel_left': np.abs(accel_x[moving_left]).mean() if moving_left.any() else 0,
            'accel_up': np.abs(accel_y[moving_up]).mean() if moving_up.any() else 0,
            'accel_down': np.abs(accel_y[moving_down]).mean() if moving_down.any() else 0,
        }

        return features

    def compare_distributions(
        self,
        generated: np.ndarray,
        real: np.ndarray
    ) -> Dict[str, float]:
        """
        比较生成轨迹和真实轨迹的分布

        Returns:
            各项指标的字典
        """
        gen_features = self.extract_features(generated)
        real_features = self.extract_features(real)

        metrics = {}

        # 比较各特征的分布
        feature_names = ['speed', 'acceleration', 'curvature', 'direction_change']

        for name in feature_names:
            if name in gen_features and name in real_features:
                gen_feat = gen_features[name]
                real_feat = real_features[name]

                # JSD
                metrics[f'{name}_jsd'] = self.dist_metrics.compute_jsd(gen_feat, real_feat)

                # EMD
                metrics[f'{name}_emd'] = self.dist_metrics.compute_emd(gen_feat, real_feat)

        # 整体相似度
        gen_speed = gen_features['speed']
        real_speed = real_features['speed']
        metrics['speed_cosine_sim'] = self.dist_metrics.compute_cosine_similarity(
            gen_speed[:min(len(gen_speed), len(real_speed))],
            real_speed[:min(len(gen_speed), len(real_speed))]
        )

        # 效率比较
        gen_eff = gen_features['efficiency']
        real_eff = real_features['efficiency']
        metrics['efficiency_mse'] = self.dist_metrics.compute_mse(
            gen_eff[:min(len(gen_eff), len(real_eff))],
            real_eff[:min(len(gen_eff), len(real_eff))]
        )

        return metrics


class ClassifierMetrics:
    """
    分类器评估指标
    使用机器学习分类器区分生成轨迹和真实轨迹
    """

    def __init__(self):
        self.feature_extractor = TrajectoryMetrics()

    def prepare_data(
        self,
        generated: np.ndarray,
        real: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备分类器训练数据

        Args:
            generated: 生成的轨迹 (n_gen, seq_len, 2)
            real: 真实轨迹 (n_real, seq_len, 2)

        Returns:
            X: 特征矩阵
            y: 标签（0=生成，1=真实）
        """
        features_list = []
        labels = []

        # 处理生成轨迹
        for i in range(len(generated)):
            feat = self._extract_feature_vector(generated[i:i+1])
            features_list.append(feat)
            labels.append(0)

        # 处理真实轨迹
        for i in range(len(real)):
            feat = self._extract_feature_vector(real[i:i+1])
            features_list.append(feat)
            labels.append(1)

        return np.array(features_list), np.array(labels)

    def _extract_feature_vector(self, trajectory: np.ndarray) -> np.ndarray:
        """提取单条轨迹的特征向量"""
        features = self.feature_extractor.extract_features(trajectory)

        # 构建特征向量
        vector = [
            features['speed_mean'][0],
            features['speed_std'][0],
            features['acceleration_mean'][0],
            features['curvature_mean'][0],
            features['path_length'][0],
            features['efficiency'][0],
            features.get('accel_right', 0),
            features.get('accel_left', 0),
            features.get('accel_up', 0),
            features.get('accel_down', 0),
        ]

        return np.array(vector)

    def evaluate_with_classifiers(
        self,
        generated: np.ndarray,
        real: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        使用多种分类器评估

        论文使用: DT, RF, XGBoost, GB, MLP, CNN, RNN, LSTM, BiLSTM, TCN
        这里实现基础的sklearn分类器
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        X, y = self.prepare_data(generated, real)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        classifiers = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500),
        }

        results = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro'),
            }

        return results


if __name__ == "__main__":
    # 测试评估指标
    np.random.seed(42)

    # 生成模拟数据
    n_samples = 100
    seq_len = 50

    # 模拟真实轨迹（带有一定的曲率）
    t = np.linspace(0, 1, seq_len)
    real_trajectories = []
    for _ in range(n_samples):
        noise = np.random.randn(seq_len, 2) * 0.02
        traj = np.stack([t, t + 0.1 * np.sin(5 * np.pi * t)], axis=-1) + noise
        real_trajectories.append(traj)
    real_trajectories = np.array(real_trajectories)

    # 模拟生成轨迹（直线+噪声）
    gen_trajectories = []
    for _ in range(n_samples):
        noise = np.random.randn(seq_len, 2) * 0.05
        traj = np.stack([t, t], axis=-1) + noise
        gen_trajectories.append(traj)
    gen_trajectories = np.array(gen_trajectories)

    # 测试特征提取
    metrics = TrajectoryMetrics()
    features = metrics.extract_features(real_trajectories[0])
    print("Extracted features:")
    for name, value in features.items():
        if isinstance(value, np.ndarray) and len(value) > 5:
            print(f"  {name}: shape={value.shape}, mean={value.mean():.4f}")
        else:
            print(f"  {name}: {value}")

    # 测试分布比较
    print("\nDistribution comparison:")
    comparison = metrics.compare_distributions(gen_trajectories, real_trajectories)
    for name, value in comparison.items():
        print(f"  {name}: {value:.4f}")

    # 测试分类器评估
    print("\nClassifier evaluation:")
    clf_metrics = ClassifierMetrics()
    results = clf_metrics.evaluate_with_classifiers(gen_trajectories, real_trajectories)
    for clf_name, scores in results.items():
        print(f"  {clf_name}:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")
