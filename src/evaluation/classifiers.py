"""
深度学习分类器模块
用于区分生成轨迹和真实轨迹 (论文白盒评估)

实现: CNN, RNN, LSTM, BiLSTM, TCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


class CNNClassifier(nn.Module):
    """1D CNN分类器"""

    def __init__(
        self,
        input_dim: int = 2,
        seq_length: int = 50,
        num_filters: int = 64,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            for k in kernel_sizes
        ])

        # 计算flatten后的维度
        conv_out_len = seq_length // 2
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(kernel_sizes) * conv_out_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))

        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RNNClassifier(nn.Module):
    """RNN分类器"""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        # 使用最后一层的隐藏状态
        out = hidden[-1]
        return self.fc(out)


class LSTMClassifier(nn.Module):
    """LSTM分类器"""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.fc(out)


class BiLSTMClassifier(nn.Module):
    """双向LSTM分类器"""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        # 连接前向和后向的最后隐藏状态
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(out)


class TCNBlock(nn.Module):
    """时间卷积网络块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + residual)

        return out


class TCNClassifier(nn.Module):
    """时间卷积网络 (TCN) 分类器"""

    def __init__(
        self,
        input_dim: int = 2,
        num_channels: List[int] = [32, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)
        in_channels = input_dim

        for i in range(num_levels):
            dilation = 2 ** i
            out_channels = num_channels[i]
            layers.append(TCNBlock(
                in_channels, out_channels,
                kernel_size, dilation, dropout
            ))
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return self.fc(x)


class DeepClassifierEvaluator:
    """
    深度学习分类器评估器

    用于训练和评估多种深度学习分类器
    来区分生成轨迹和真实轨迹
    """

    def __init__(
        self,
        input_dim: int = 2,
        seq_length: int = 50,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.device = device

        # 定义所有分类器
        self.classifiers = {
            'CNN': CNNClassifier(input_dim, seq_length),
            'RNN': RNNClassifier(input_dim),
            'LSTM': LSTMClassifier(input_dim),
            'BiLSTM': BiLSTMClassifier(input_dim),
            'TCN': TCNClassifier(input_dim),
        }

    def prepare_data(
        self,
        generated: np.ndarray,
        real: np.ndarray,
        test_size: float = 0.3,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和测试数据

        Args:
            generated: 生成轨迹 (n_gen, seq_len, 2)
            real: 真实轨迹 (n_real, seq_len, 2)
            test_size: 测试集比例

        Returns:
            (train_loader, test_loader)
        """
        # 组合数据
        X = np.concatenate([generated, real], axis=0)
        y = np.concatenate([
            np.zeros(len(generated)),  # 0 = generated
            np.ones(len(real)),  # 1 = real
        ])

        # 分割数据集 (7:3 按论文)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 转换为Tensor
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def train_classifier(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        num_epochs: int = 50,
        lr: float = 1e-3,
    ) -> nn.Module:
        """训练单个分类器"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        return model

    def evaluate_classifier(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """评估分类器"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            # 按论文指标：检测准确率（分类器越难区分，说明生成效果越好）
            'detection_accuracy': accuracy_score(all_labels, all_preds),
        }

    def evaluate_all(
        self,
        generated: np.ndarray,
        real: np.ndarray,
        num_epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        使用所有分类器评估

        Returns:
            {classifier_name: {metric_name: value}}
        """
        train_loader, test_loader = self.prepare_data(generated, real)

        results = {}
        for name, model in self.classifiers.items():
            if verbose:
                print(f"Training {name}...")

            # 重新初始化模型
            if name == 'CNN':
                model = CNNClassifier(self.input_dim, self.seq_length)
            elif name == 'RNN':
                model = RNNClassifier(self.input_dim)
            elif name == 'LSTM':
                model = LSTMClassifier(self.input_dim)
            elif name == 'BiLSTM':
                model = BiLSTMClassifier(self.input_dim)
            elif name == 'TCN':
                model = TCNClassifier(self.input_dim)

            trained_model = self.train_classifier(model, train_loader, num_epochs)
            metrics = self.evaluate_classifier(trained_model, test_loader)
            results[name] = metrics

            if verbose:
                print(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

        return results


def evaluate_with_deep_classifiers(
    generated: np.ndarray,
    real: np.ndarray,
    device: str = "cuda",
    num_epochs: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    便捷函数：使用深度学习分类器评估

    Args:
        generated: 生成轨迹 (n, seq_len, 2)
        real: 真实轨迹 (n, seq_len, 2)
        device: 设备
        num_epochs: 训练轮数
        verbose: 是否打印进度

    Returns:
        评估结果字典
    """
    evaluator = DeepClassifierEvaluator(
        input_dim=2,
        seq_length=generated.shape[1],
        device=device,
    )

    return evaluator.evaluate_all(generated, real, num_epochs, verbose)


if __name__ == "__main__":
    # 测试深度学习分类器
    print("Testing deep learning classifiers...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    seq_len = 50

    # 模拟真实轨迹（带曲线）
    t = np.linspace(0, 1, seq_len)
    real = []
    for _ in range(n_samples):
        noise = np.random.randn(seq_len, 2) * 0.02
        traj = np.stack([t, t + 0.1 * np.sin(5 * np.pi * t)], axis=-1) + noise
        real.append(traj)
    real = np.array(real)

    # 模拟生成轨迹（直线+噪声）
    generated = []
    for _ in range(n_samples):
        noise = np.random.randn(seq_len, 2) * 0.05
        traj = np.stack([t, t], axis=-1) + noise
        generated.append(traj)
    generated = np.array(generated)

    # 评估
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Real shape: {real.shape}")

    results = evaluate_with_deep_classifiers(
        generated, real,
        device=device,
        num_epochs=30,
        verbose=True
    )

    print("\n" + "=" * 50)
    print("Summary (Lower detection accuracy = better generation)")
    print("=" * 50)
    for clf_name, metrics in results.items():
        print(f"{clf_name}: Detection Accuracy = {metrics['detection_accuracy']:.4f}")
