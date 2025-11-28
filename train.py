"""
DMTG训练脚本
训练α-DDIM模型生成人类鼠标轨迹

支持单GPU和多GPU (DDP) 训练:
- 单GPU: python train.py
- 多GPU: torchrun --nproc_per_node=NUM_GPUS train.py
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


# ==================== DDP 工具函数 ====================

def is_distributed() -> bool:
    """检查是否在分布式环境中运行"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """获取当前进程的 rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """是否为主进程 (rank 0)"""
    return get_rank() == 0


def setup_distributed():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 设置当前设备
        torch.cuda.set_device(local_rank)

        # 初始化进程组
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        return local_rank
    return 0


def cleanup_distributed():
    """清理分布式环境"""
    if is_distributed():
        dist.destroy_process_group()


class Trainer:
    """DMTG训练器"""

    def __init__(
        self,
        model: AlphaDDIM,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        # 损失权重 (论文 Eq.14)
        lambda_ddim: float = 1.0,
        lambda_sim: float = 0.1,
        lambda_style: float = 0.05,
    ):
        self.device = device
        self.model = model.to(device)

        # 保存原始模型引用（用于访问模型方法和属性，如 timesteps, q_sample 等）
        self.model_raw = self.model

        # DDP 多GPU支持
        self.distributed = is_distributed()
        if self.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            # DDP 包装内部的 UNet（实际需要并行的部分）
            self.model_raw.model = DDP(
                self.model_raw.model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
            if is_main_process():
                print(f"Using {get_world_size()} GPUs with DistributedDataParallel")

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

        # 学习率调度器
        self.num_epochs = num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        # 损失函数 (论文 Eq.14: L = w1·LDDIM + w2·Lsim + w3·Lstyle)
        self.loss_fn = DMTGLoss(
            lambda_ddim=lambda_ddim,
            lambda_sim=lambda_sim,
            lambda_style=lambda_style,
        )

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

        # DDP: 设置 sampler 的 epoch 以确保每个 epoch 数据打乱不同
        if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)

        # 只在主进程显示进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", disable=not is_main_process())
        loss_components = {'ddim': 0, 'sim': 0, 'style': 0, 'boundary': 0}

        for batch in pbar:
            trajectory = batch['trajectory'].to(self.device)
            mask = batch['mask'].to(self.device)  # 有效位置掩码
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)

            # 构建条件
            condition = torch.cat([start_point, end_point], dim=1)
            batch_size = trajectory.shape[0]

            # 从轨迹计算真实的复杂度α (而非随机采样)
            # 这确保 style loss 学习正确的 α → 复杂度映射
            # α 基于路径长度比率: α = clamp((path_length/straight_dist - 1) / 2, 0, 1)
            # 必须传入mask以忽略padding部分，否则会错误计算终点和路径长度
            alpha = self.model_raw.compute_trajectory_alpha(trajectory, mask)
            # 约束到论文推荐范围 [0.3, 0.8]
            alpha = 0.3 + 0.5 * alpha  # 映射 [0,1] -> [0.3, 0.8]

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

        for batch in tqdm(self.val_loader, desc="Validation", disable=not is_main_process()):
            trajectory = batch['trajectory'].to(self.device)
            mask = batch['mask'].to(self.device)  # 有效位置掩码
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)

            condition = torch.cat([start_point, end_point], dim=1)
            batch_size = trajectory.shape[0]

            # 从轨迹计算真实的复杂度α (与训练一致)
            # 必须传入mask以忽略padding部分
            alpha = self.model_raw.compute_trajectory_alpha(trajectory, mask)
            alpha = 0.3 + 0.5 * alpha  # 映射 [0,1] -> [0.3, 0.8]
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
        """保存检查点 (只在主进程保存)"""
        if not is_main_process():
            return

        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"

        # DDP: 需要 unwrap DDP wrapper 来获取原始权重
        # 构建一个干净的 state_dict（不含 DDP 的 module. 前缀）
        if self.distributed:
            # 获取 AlphaDDIM 的 state_dict，但 UNet 部分需要 unwrap
            state_dict = {}
            for k, v in self.model_raw.state_dict().items():
                # model.model. 开头的是 DDP 包装的 UNet
                if k.startswith('model.module.'):
                    # 去掉 module. 前缀
                    new_key = k.replace('model.module.', 'model.')
                    state_dict[new_key] = v
                else:
                    state_dict[k] = v
            model_state = state_dict
        else:
            model_state = self.model_raw.state_dict()

        # 提取模型配置（用于正确加载检查点）
        model_config = {
            'seq_length': self.model_raw.seq_length,
            'timesteps': self.model_raw.timesteps,
            'base_channels': self.model_raw.model.base_channels if hasattr(self.model_raw.model, 'base_channels') else 64,
        }

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'model_config': model_config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, total_epochs: int = None):
        """
        加载检查点

        Args:
            path: 检查点路径
            total_epochs: 总训练轮数（用于重新计算 LR scheduler）
                         如果为 None，使用 Trainer 初始化时的 num_epochs
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])

        # 重新创建 scheduler 以匹配目标总轮数
        # 这避免了 T_max 不匹配导致的 LR 衰减过快/过慢问题
        target_epochs = total_epochs if total_epochs else self.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=target_epochs,
            eta_min=1e-6,
            last_epoch=self.epoch,  # 从当前 epoch 继续
        )

        print(f"Loaded checkpoint from {path}, epoch {self.epoch}")
        print(f"LR scheduler reset with T_max={target_epochs}, last_epoch={self.epoch}")

    def train(self, num_epochs: int = None):
        """
        完整训练流程

        Args:
            num_epochs: 目标总轮数（非附加轮数）。如果为 None，使用初始化时的 num_epochs
                       例如：resume 在 epoch 50，num_epochs=100 表示训练到 epoch 100
        """
        target_epochs = num_epochs if num_epochs else self.num_epochs
        start_epoch = self.epoch  # 从当前epoch继续（恢复训练时不为0）
        end_epoch = target_epochs

        if start_epoch >= end_epoch:
            if is_main_process():
                print(f"Already at epoch {start_epoch}, target is {end_epoch}. Nothing to do.")
            return

        if is_main_process():
            print(f"Starting training from epoch {start_epoch + 1} to {end_epoch}")
            print(f"Device: {self.device}")
            print(f"Train samples: {len(self.train_loader.dataset)}")
            if self.distributed:
                print(f"World size: {get_world_size()}")

        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch + 1

            # 训练
            train_loss = self.train_epoch()

            # DDP: 同步所有进程的 loss 取平均
            if self.distributed:
                loss_tensor = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = loss_tensor.item()

            if is_main_process():
                print(f"Epoch {self.epoch} - Train Loss: {train_loss:.4f}")

            # 验证
            if self.val_loader is not None:
                val_loss = self.validate()

                if self.distributed:
                    loss_tensor = torch.tensor([val_loss], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = loss_tensor.item()

                if is_main_process():
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

            # DDP: 同步屏障，确保所有进程完成当前 epoch
            if self.distributed:
                dist.barrier()

        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        if is_main_process():
            print("Training completed!")

    def save_training_log(self):
        """保存训练日志 (只在主进程保存)"""
        if not is_main_process():
            return

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

    # 损失权重 (论文 Eq.14)
    parser.add_argument(
        "--lambda_ddim",
        type=float,
        default=1.0,
        help="DDIM噪声预测损失权重 (w1)"
    )
    parser.add_argument(
        "--lambda_sim",
        type=float,
        default=0.1,
        help="相似度损失权重 (w2)"
    )
    parser.add_argument(
        "--lambda_style",
        type=float,
        default=0.05,
        help="风格损失权重 (w3)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ==================== DDP 初始化 ====================
    local_rank = setup_distributed()

    # 设置设备
    if is_distributed():
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if is_main_process():
        print("=" * 50)
        print("DMTG Training")
        if is_distributed():
            print(f"Mode: DistributedDataParallel ({get_world_size()} GPUs)")
        else:
            print("Mode: Single GPU")
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
        if is_main_process():
            print("Error: No dataset found!")
            print(f"  Checked: {sapimouse_dir}")
            print(f"  Checked: {boun_dir}")
            print(f"  Checked: {open_images_dir}")
        cleanup_distributed()
        return

    if is_main_process():
        print(f"SapiMouse: {sapimouse_path or 'Not found'}")
        print(f"BOUN: {boun_path or 'Not found'}")
        print(f"Open Images: {open_images_path or 'Not found'}")

    # ==================== 创建数据集 ====================
    if is_main_process():
        print("\nLoading dataset...")
        print(f"Max sequence length: {args.max_length} (Padding with Masking)")

    # 先创建完整数据集
    from torch.utils.data import random_split
    dataset = CombinedMouseDataset(
        sapimouse_dir=sapimouse_path,
        boun_dir=boun_path,
        open_images_dir=open_images_path,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    # 分割训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ==================== 创建数据加载器 (支持 DDP) ====================
    if is_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False  # 使用 sampler 时不能 shuffle
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if val_size > 0 else None

    if is_main_process():
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")

    # ==================== 创建模型 ====================
    if is_main_process():
        print("\nCreating model...")

    unet = TrajectoryUNet(
        seq_length=args.max_length,
        input_dim=2,
        base_channels=args.base_channels,
    )

    model = AlphaDDIM(
        model=unet,
        timesteps=args.timesteps,
        seq_length=args.max_length,
        input_dim=2,
    )

    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    # ==================== 创建训练器 ====================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,  # 传入总轮数以正确设置 LR scheduler
        device=device,  # 使用 DDP 设置的设备
        checkpoint_dir=args.checkpoint_dir,
        # 损失权重 (论文 Eq.14)
        lambda_ddim=args.lambda_ddim,
        lambda_sim=args.lambda_sim,
        lambda_style=args.lambda_style,
    )

    # 恢复训练
    if args.resume:
        # 传入 total_epochs 以正确重置 LR scheduler
        trainer.load_checkpoint(args.resume, total_epochs=args.num_epochs)

    # ==================== 开始训练 ====================
    try:
        trainer.train(args.num_epochs)
    finally:
        # 清理 DDP
        cleanup_distributed()


if __name__ == "__main__":
    main()
