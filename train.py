"""
DMTG训练脚本
训练α-DDIM模型生成人类鼠标轨迹
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import orjson
from datetime import datetime

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import CombinedMouseDataset, create_dataloader
from src.models.unet import TrajectoryUNet
from src.models.alpha_ddim import AlphaDDIM
from src.models.losses import DMTGLoss


class Trainer:
    """DMTG训练器"""

    def __init__(
        self,
        model: AlphaDDIM,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 1e-4,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.device = device
        self.model = model.to(device)

        # 保存原始模型引用（用于访问模型方法和属性）
        self.model_raw = self.model

        # 多GPU支持 (DataParallel)
        self.multi_gpu = torch.cuda.device_count() > 1
        if self.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

        # 学习率调度器 (T_max 会在恢复训练时重新设置)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # 默认值，恢复训练时会更新
            eta_min=1e-6,
        )

        # 损失函数
        self.loss_fn = DMTGLoss()

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        loss_components = {'ddim': 0, 'sim': 0, 'style': 0, 'boundary': 0}

        for batch in pbar:
            trajectory = batch['trajectory'].to(self.device)
            mask = batch['mask'].to(self.device)  # 有效位置掩码
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)

            # 构建条件
            condition = torch.cat([start_point, end_point], dim=1)
            batch_size = trajectory.shape[0]

            # 随机采样α (论文核心: 模型学习按指定α生成轨迹)
            # 论文推荐范围: 0.3-0.8
            # α=0.3: 简单轨迹 (接近直线)
            # α=0.8: 复杂轨迹 (更多曲线变化)
            alpha = 0.3 + 0.5 * torch.rand(batch_size, device=self.device)  # [0.3, 0.8]

            # 随机采样时间步（使用原始模型引用访问属性）
            t = torch.randint(0, self.model_raw.timesteps, (batch_size,), device=self.device)

            # 前向扩散（使用原始模型引用访问方法）
            noise = torch.randn_like(trajectory)
            x_t, _ = self.model_raw.q_sample(trajectory, t, noise)

            # 预测噪声 - 论文Eq.10: ε_θ(x_t, t, c, α)
            predicted_noise = self.model_raw.model(x_t, t, condition, alpha)

            # 计算预测的x0（用于辅助损失）
            predicted_x0 = self.model_raw.predict_x0_from_noise(x_t, t, predicted_noise)

            # 使用DMTGLoss计算完整损失 (论文公式14)
            # L = w1·LDDIM + w2·Lsim + w3·Lstyle
            loss_dict = self.loss_fn(
                predicted_noise=predicted_noise,
                target_noise=noise,
                predicted_x0=predicted_x0,
                target_x0=trajectory,  # 人类模板 X̂
                alpha=alpha,           # 目标复杂度 (论文Eq.13)
                t=t,
                timesteps=self.model_raw.timesteps,
                mask=mask,             # 有效位置掩码
            )
            loss = loss_dict['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            loss_components['ddim'] += loss_dict['ddim_loss'].item()
            if 'similarity_loss' in loss_dict:
                loss_components['sim'] += loss_dict['similarity_loss'].item()
            if 'style_loss' in loss_dict:
                loss_components['style'] += loss_dict['style_loss'].item()
            if 'boundary_loss' in loss_dict:
                loss_components['boundary'] += loss_dict['boundary_loss'].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ddim': f"{loss_dict['ddim_loss'].item():.4f}",
            })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """验证 - 使用与训练相同的完整损失"""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            trajectory = batch['trajectory'].to(self.device)
            mask = batch['mask'].to(self.device)  # 有效位置掩码
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)

            condition = torch.cat([start_point, end_point], dim=1)
            batch_size = trajectory.shape[0]

            # 随机采样α (论文推荐范围 0.3-0.8)
            alpha = 0.3 + 0.5 * torch.rand(batch_size, device=self.device)
            t = torch.randint(0, self.model_raw.timesteps, (batch_size,), device=self.device)

            noise = torch.randn_like(trajectory)
            x_t, _ = self.model_raw.q_sample(trajectory, t, noise)
            predicted_noise = self.model_raw.model(x_t, t, condition, alpha)

            # 计算预测的x0
            predicted_x0 = self.model_raw.predict_x0_from_noise(x_t, t, predicted_noise)

            # 使用DMTGLoss计算完整损失 (论文公式14)
            loss_dict = self.loss_fn(
                predicted_noise=predicted_noise,
                target_noise=noise,
                predicted_x0=predicted_x0,
                target_x0=trajectory,  # 人类模板 X̂
                alpha=alpha,           # 目标复杂度 (论文Eq.13)
                t=t,
                timesteps=self.model_raw.timesteps,
                mask=mask,             # 有效位置掩码
            )
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model_raw.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model_raw.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])

        print(f"Loaded checkpoint from {path}, epoch {self.epoch}")

    def train(self, num_epochs: int):
        """完整训练流程"""
        start_epoch = self.epoch  # 从当前epoch继续（恢复训练时不为0）
        end_epoch = start_epoch + num_epochs

        print(f"Starting training from epoch {start_epoch + 1} to {end_epoch}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")

        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch + 1

            # 训练
            train_loss = self.train_epoch()
            print(f"Epoch {self.epoch} - Train Loss: {train_loss:.4f}")

            # 验证
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {self.epoch} - Val Loss: {val_loss:.4f}")

                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            else:
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.save_checkpoint("best_model.pt")

            # 更新学习率
            self.scheduler.step()

            # 定期保存检查点
            if self.epoch % 10 == 0:
                self.save_checkpoint()

            # 保存训练日志
            self.save_training_log()

        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        print("Training completed!")

    def save_training_log(self):
        """保存训练日志"""
        log = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'timestamp': datetime.now().isoformat(),
        }

        log_path = self.log_dir / "training_log.json"
        with open(log_path, 'wb') as f:
            f.write(orjson.dumps(log, option=orjson.OPT_INDENT_2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train DMTG model")

    # 数据参数
    parser.add_argument(
        "--sapimouse_dir",
        type=str,
        default="datasets/sapimouse",
        help="SapiMouse数据集目录"
    )
    parser.add_argument(
        "--boun_dir",
        type=str,
        default="datasets/boun-processed",
        help="BOUN预处理后数据集目录（自动检测Parquet或JSONL格式）"
    )
    parser.add_argument(
        "--open_images_dir",
        type=str,
        default="datasets/open_images_v6",
        help="Open Images Localized Narratives数据集目录"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=500,
        help="最大轨迹序列长度 N（论文默认500，使用 Padding with Masking）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（用于快速测试）"
    )

    # 模型参数
    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
        help="U-Net基础通道数"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="扩散时间步数"
    )

    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="批次大小"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,  # 多线程数据加载
        help="数据加载工作进程数"
    )

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="检查点保存目录"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("DMTG Training")
    print("=" * 50)

    # 检查数据集路径
    base_dir = Path(__file__).parent
    sapimouse_dir = base_dir / args.sapimouse_dir
    boun_dir = base_dir / args.boun_dir if args.boun_dir else None
    open_images_dir = base_dir / args.open_images_dir if args.open_images_dir else None

    sapimouse_path = str(sapimouse_dir) if sapimouse_dir.exists() else None
    boun_path = str(boun_dir) if boun_dir and boun_dir.exists() else None
    open_images_path = str(open_images_dir) if open_images_dir and open_images_dir.exists() else None

    if sapimouse_path is None and boun_path is None and open_images_path is None:
        print("Error: No dataset found!")
        print(f"  Checked: {sapimouse_dir}")
        print(f"  Checked: {boun_dir}")
        print(f"  Checked: {open_images_dir}")
        return

    print(f"SapiMouse: {sapimouse_path or 'Not found'}")
    print(f"BOUN: {boun_path or 'Not found'}")
    print(f"Open Images: {open_images_path or 'Not found'}")

    # 创建数据加载器 (分割训练集和验证集)
    # 使用 Sequence Padding with Masking 方法（与论文一致）
    print("\nLoading dataset...")
    print(f"Max sequence length: {args.max_length} (Padding with Masking)")
    train_loader, val_loader = create_dataloader(
        sapimouse_dir=sapimouse_path,
        boun_dir=boun_path,
        open_images_dir=open_images_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        val_split=0.1,  # 10% 验证集
        return_val=True,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # 创建模型
    print("\nCreating model...")
    unet = TrajectoryUNet(
        seq_length=args.max_length,  # 使用 max_length 作为序列长度
        input_dim=2,
        base_channels=args.base_channels,
    )

    model = AlphaDDIM(
        model=unet,
        timesteps=args.timesteps,
        seq_length=args.max_length,  # 使用 max_length 作为序列长度
        input_dim=2,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # 添加验证集
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()
