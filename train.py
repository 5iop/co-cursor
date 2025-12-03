"""
DMTG训练脚本
训练α-DDIM模型生成人类鼠标轨迹

支持单GPU和多GPU (DDP) 训练:
- 单GPU: python train.py
- 多GPU: torchrun --nproc_per_node=NUM_GPUS train.py
"""
import os
import sys
import signal
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
import threading
import subprocess

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import CombinedMouseDataset, create_dataloader
from src.models.unet import TrajectoryUNet
from src.models.alpha_ddim import AlphaDDIM
from src.models.losses import DMTGLoss
from src.utils.notify import send_training_update, send_notification


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
        # 损失权重 (论文 Eq.14 + 长度预测)
        lambda_ddim: float = 1.0,
        lambda_sim: float = 0.1,
        lambda_style: float = 0.05,
        lambda_length: float = 0.1,  # 长度预测损失权重
        # 自动绘图选项
        plot: bool = False,
        human_data_dir: str = None,
        # 通知选项
        webhook_url: str = None,
        notify_every: int = 5,
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

        # 优化器 - 在 DDP 包装之后创建，确保使用正确的参数
        self.optimizer = optim.AdamW(
            self.model.parameters(),  # 使用 DDP 包装后的模型参数
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

        # 损失函数 (论文 Eq.14 + 长度预测: L = w1·LDDIM + w2·Lsim + w3·Lstyle + w4·Llength)
        self.loss_fn = DMTGLoss(
            lambda_ddim=lambda_ddim,
            lambda_sim=lambda_sim,
            lambda_style=lambda_style,
            lambda_length=lambda_length,
        )

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.learning_rates = []

        # 自动绘图选项
        self.plot = plot
        self.human_data_dir = human_data_dir
        self.training_start_time = datetime.now().strftime("%y%m%d%H%M%S")

        # 通知选项
        self.webhook_url = webhook_url
        self.notify_every = notify_every

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
        loss_components = {'ddim': 0, 'sim': 0, 'style': 0, 'boundary': 0, 'length': 0}

        for batch in pbar:
            trajectory = batch['trajectory'].to(self.device)
            mask = batch['mask'].to(self.device)  # 有效位置掩码
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)
            length = batch['length'].to(self.device)  # 轨迹真实长度 (batch, 1)

            # 构建条件
            condition = torch.cat([start_point, end_point], dim=1)
            batch_size = trajectory.shape[0]

            # 从轨迹计算真实的复杂度α (而非随机采样)
            # 这确保 style loss 学习正确的 α → 复杂度映射
            # α 基于路径长度比率: α = clamp((path_length/straight_dist - 1) / 2, 0, 1)
            # 必须传入mask以忽略padding部分，否则会错误计算终点和路径长度
            # 注意: 训练时 α 保持 [0,1] 尺度，与 style loss 中的 pred_complexity 一致
            # [0.3, 0.8] 约束只在采样时应用
            alpha = self.model_raw.compute_trajectory_alpha(trajectory, mask)

            # 随机采样时间步（使用原始模型引用访问属性）
            t = torch.randint(0, self.model_raw.timesteps, (batch_size,), device=self.device)

            # 前向扩散（使用原始模型引用访问方法）
            noise = torch.randn_like(trajectory)
            x_t, _ = self.model_raw.q_sample(trajectory, t, noise)

            # 预测噪声 - 论文Eq.10: ε_θ(x_t, t, c, α)
            predicted_noise = self.model_raw.model(x_t, t, condition, alpha)

            # 计算预测的x0（用于辅助损失）
            predicted_x0 = self.model_raw.predict_x0_from_noise(x_t, t, predicted_noise)

            # 长度预测 (如果模型支持)
            # 注意: DDP 只同步 forward() 的梯度，predict_length 不经过 forward
            # 但由于 predict_length 内部使用的参数都属于 DDP 包装的模块，梯度会正确同步
            predicted_log_length = None
            unet = self.model_raw.model.module if self.distributed else self.model_raw.model
            if hasattr(unet, 'length_head') and unet.length_head is not None:
                # 不传入 time，让长度预测不依赖时间步（与推理一致）
                predicted_log_length = unet.predict_length(x_t, condition, alpha)

            # 使用DMTGLoss计算完整损失 (论文公式14 + 长度预测)
            # L = w1·LDDIM + w2·Lsim + w3·Lstyle + w4·Llength
            loss_dict = self.loss_fn(
                predicted_noise=predicted_noise,
                target_noise=noise,
                predicted_x0=predicted_x0,
                target_x0=trajectory,  # 人类模板 X̂
                alpha=alpha,           # 目标复杂度 (论文Eq.13)
                t=t,
                timesteps=self.model_raw.timesteps,
                mask=mask,             # 有效位置掩码
                predicted_log_length=predicted_log_length,  # 预测的 log(m+1)
                target_length=length.squeeze(-1),  # 目标长度 m
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
            if 'length_loss' in loss_dict:
                loss_components['length'] += loss_dict['length_loss'].item()
            num_batches += 1
            self.global_step += 1

            postfix = {
                'loss': f"{loss.item():.4f}",
                'ddim': f"{loss_dict['ddim_loss'].item():.4f}",
            }
            if 'length_loss' in loss_dict:
                postfix['len'] = f"{loss_dict['length_loss'].item():.4f}"
            pbar.set_postfix(postfix)

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

        # DDP: 设置 sampler 的 epoch 以确保验证数据一致性
        if self.distributed and hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(self.epoch)

        for batch in tqdm(self.val_loader, desc="Validation", disable=not is_main_process()):
            trajectory = batch['trajectory'].to(self.device)
            mask = batch['mask'].to(self.device)  # 有效位置掩码
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)
            length = batch['length'].to(self.device)  # 轨迹真实长度 (batch, 1)

            condition = torch.cat([start_point, end_point], dim=1)
            batch_size = trajectory.shape[0]

            # 从轨迹计算真实的复杂度α (与训练一致)
            # 必须传入mask以忽略padding部分
            # 训练时 α 保持 [0,1] 尺度
            alpha = self.model_raw.compute_trajectory_alpha(trajectory, mask)
            t = torch.randint(0, self.model_raw.timesteps, (batch_size,), device=self.device)

            noise = torch.randn_like(trajectory)
            x_t, _ = self.model_raw.q_sample(trajectory, t, noise)
            predicted_noise = self.model_raw.model(x_t, t, condition, alpha)

            # 计算预测的x0
            predicted_x0 = self.model_raw.predict_x0_from_noise(x_t, t, predicted_noise)

            # 长度预测 (如果模型支持)
            # 注意: DDP 只同步 forward() 的梯度，predict_length 不经过 forward
            # 但由于 predict_length 内部使用的参数都属于 DDP 包装的模块，梯度会正确同步
            predicted_log_length = None
            unet = self.model_raw.model.module if self.distributed else self.model_raw.model
            if hasattr(unet, 'length_head') and unet.length_head is not None:
                # 不传入 time，让长度预测不依赖时间步（与推理一致）
                predicted_log_length = unet.predict_length(x_t, condition, alpha)

            # 使用DMTGLoss计算完整损失 (论文公式14 + 长度预测)
            loss_dict = self.loss_fn(
                predicted_noise=predicted_noise,
                target_noise=noise,
                predicted_x0=predicted_x0,
                target_x0=trajectory,  # 人类模板 X̂
                alpha=alpha,           # 目标复杂度 (论文Eq.13)
                t=t,
                timesteps=self.model_raw.timesteps,
                mask=mask,             # 有效位置掩码
                predicted_log_length=predicted_log_length,  # 预测的 log(m+1)
                target_length=length.squeeze(-1),  # 目标长度 m
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
        # DDP 模式下需要通过 .module 访问原始模型属性
        unet = self.model_raw.model.module if self.distributed else self.model_raw.model
        model_config = {
            'seq_length': self.model_raw.seq_length,
            'timesteps': self.model_raw.timesteps,
            'input_dim': self.model_raw.input_dim,
            'base_channels': unet.base_channels if hasattr(unet, 'base_channels') else 64,
            'enable_length_prediction': unet.length_head is not None if hasattr(unet, 'length_head') else False,
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
            'learning_rates': self.learning_rates,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def run_tsne_plot_async(self):
        """在后台线程运行 t-SNE 分布图和轨迹生成测试 (只在主进程运行)"""
        if not is_main_process() or not self.plot:
            return

        try:
            from src.utils.notify import run_visualization_scripts

            label = f"{self.training_start_time}_epoch{self.epoch}"
            checkpoint_path = str(self.checkpoint_dir / "best_model.pt")

            run_visualization_scripts(
                checkpoint_path=checkpoint_path,
                label=label,
                webhook_url=self.webhook_url,
                human_data_dir=self.human_data_dir,
                cwd=str(Path(__file__).parent),
            )
        except Exception as e:
            print(f"[Visualization] Failed to start: {e}")

    def load_checkpoint(self, path: str, total_epochs: int = None):
        """
        加载检查点

        Args:
            path: 检查点路径
            total_epochs: 总训练轮数（用于重新计算 LR scheduler）
                         如果为 None，使用 Trainer 初始化时的 num_epochs
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # DDP: checkpoint 保存时去掉了 module. 前缀，加载时需要加回来
        model_state = checkpoint['model_state_dict']
        if self.distributed:
            # 给 UNet 的 keys 加回 module. 前缀
            new_state = {}
            for k, v in model_state.items():
                if k.startswith('model.'):
                    # model.xxx -> model.module.xxx
                    new_key = k.replace('model.', 'model.module.', 1)
                    new_state[new_key] = v
                else:
                    new_state[k] = v
            model_state = new_state

        self.model_raw.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

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

            # 每个epoch开始前打印关键参数
            if is_main_process():
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"\n[Epoch {self.epoch}/{end_epoch}] LR: {current_lr:.2e} | Best: {self.best_loss:.4f}")

            # 训练
            train_loss = self.train_epoch()

            # DDP: 同步所有进程的 loss 取平均
            if self.distributed:
                loss_tensor = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = loss_tensor.item()

            # 验证
            val_loss = None
            if self.val_loader is not None:
                val_loss = self.validate()

                if self.distributed:
                    loss_tensor = torch.tensor([val_loss], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = loss_tensor.item()

            if is_main_process():
                if val_loss is not None:
                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Train Loss: {train_loss:.4f}")

            # 保存最佳模型
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint("best_model.pt")
                self.run_tsne_plot_async()  # 后台运行 t-SNE 绘图
                # 发送 best model 通知 (只在主进程发送)
                if self.webhook_url and is_main_process():
                    send_training_update(
                        epoch=self.epoch,
                        loss=current_loss,
                        best_loss=self.best_loss,
                        is_best=True,
                        webhook_url=self.webhook_url,
                        extra_info="New best model saved!",
                    )

            # 记录当前 epoch 的学习率，然后更新
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
            self.scheduler.step()

            # 周期性训练进度通知（best model 已发送时跳过，只在主进程发送）
            is_best_this_epoch = current_loss == self.best_loss
            if self.webhook_url and is_main_process() and self.epoch % self.notify_every == 0 and not is_best_this_epoch:
                current_loss = val_loss if self.val_loader else train_loss
                send_training_update(
                    epoch=self.epoch,
                    loss=current_loss,
                    best_loss=self.best_loss,
                    is_best=False,
                    webhook_url=self.webhook_url,
                    extra_info=f"LR: {self.scheduler.get_last_lr()[0]:.2e}",
                )

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
            # 发送训练完成通知
            if self.webhook_url:
                send_notification(
                    title="DMTG Training Completed",
                    body=f"Epochs: {self.epoch}\nBest Loss: {self.best_loss:.4f}",
                    webhook_url=self.webhook_url,
                )

    def save_training_log(self):
        """保存训练日志 (只在主进程保存)"""
        if not is_main_process():
            return

        log = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
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
        default=128,  # 方案B: 64→128, 参数量 4.7M→17M
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
        default=4,
        help="数据加载工作进程数"
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="禁用 persistent_workers 以节省内存"
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

    # 损失权重 (论文 Eq.14 + 长度预测)
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
    parser.add_argument(
        "--lambda_length",
        type=float,
        default=0.1,
        help="长度预测损失权重 (w4)"
    )
    parser.add_argument(
        "--length_mode",
        type=str,
        default="shared_encoder",
        choices=["shared_encoder", "disabled"],
        help="长度预测模式: shared_encoder (共享Encoder特征), disabled (禁用)"
    )

    # 自动绘图选项
    parser.add_argument(
        "--plot",
        action="store_true",
        help="保存 best_model 后自动在后台运行 t-SNE 分布图和轨迹生成测试"
    )

    # 通知选项
    parser.add_argument(
        "--webhook",
        type=str,
        default=None,
        help="Webhook URL 用于发送训练通知和图片 (e.g. https://ntfy.jangit.me/notify/notifytg)"
    )
    parser.add_argument(
        "--notify_every",
        type=int,
        default=5,
        help="每N个epoch发送一次训练进度通知 (默认: 5)"
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
    boun_dir = base_dir / args.boun_dir if args.boun_dir else None
    open_images_dir = base_dir / args.open_images_dir if args.open_images_dir else None

    boun_path = str(boun_dir) if boun_dir and boun_dir.exists() else None
    open_images_path = str(open_images_dir) if open_images_dir and open_images_dir.exists() else None

    if boun_path is None and open_images_path is None:
        if is_main_process():
            print("Error: No dataset found!")
            print(f"  Checked: {boun_dir}")
            print(f"  Checked: {open_images_dir}")
        cleanup_distributed()
        return

    if is_main_process():
        print(f"BOUN: {boun_path or 'Not found'}")
        print(f"Open Images: {open_images_path or 'Not found'}")

    # ==================== 创建数据集 ====================
    if is_main_process():
        print("\nLoading dataset...")
        print(f"Max sequence length: {args.max_length} (Padding with Masking)")

    # 先创建完整数据集
    from torch.utils.data import random_split
    dataset = CombinedMouseDataset(
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
        persistent_workers=args.num_workers > 0 and not args.no_persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0 and not args.no_persistent_workers,
    ) if val_size > 0 else None

    if is_main_process():
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")

    # ==================== 创建模型 ====================
    if is_main_process():
        print("\nCreating model...")

    # Shared Encoder 长度预测模式：使用 U-Net encoder bottleneck 特征
    enable_length_pred = args.length_mode != "disabled" if hasattr(args, 'length_mode') else True

    unet = TrajectoryUNet(
        seq_length=args.max_length,
        input_dim=3,  # x, y, dt
        base_channels=args.base_channels,
        enable_length_prediction=enable_length_pred,
    )

    model = AlphaDDIM(
        model=unet,
        timesteps=args.timesteps,
        seq_length=args.max_length,
        input_dim=3,  # x, y, dt
    )

    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        print(f"Length prediction: {'enabled' if enable_length_pred else 'disabled'}")

    # ==================== 创建训练器 ====================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,  # 传入总轮数以正确设置 LR scheduler
        device=device,  # 使用 DDP 设置的设备
        checkpoint_dir=args.checkpoint_dir,
        # 损失权重 (论文 Eq.14 + 长度预测)
        lambda_ddim=args.lambda_ddim,
        lambda_sim=args.lambda_sim,
        lambda_style=args.lambda_style,
        lambda_length=args.lambda_length,
        # 自动绘图选项
        plot=args.plot,
        human_data_dir=boun_path,  # 使用 BOUN 数据作为人类轨迹参考
        # 通知选项
        webhook_url=args.webhook,
        notify_every=args.notify_every,
    )

    # 发送训练开始通知
    if args.webhook and is_main_process():
        send_training_update(
            epoch=0,
            loss=0,
            best_loss=float('inf'),
            webhook_url=args.webhook,
            extra_info=f"Training started!\nEpochs: {args.num_epochs}\nBatch size: {args.batch_size}\nSamples: {len(train_dataset)}",
        )

    # 恢复训练
    if args.resume:
        # 传入 total_epochs 以正确重置 LR scheduler
        trainer.load_checkpoint(args.resume, total_epochs=args.num_epochs)

    # ==================== 信号处理 ====================
    def handle_signal(signum, frame):
        """处理终止信号，发送通知"""
        sig_name = signal.Signals(signum).name
        if is_main_process() and args.webhook:
            send_notification(
                title=f"DMTG Training Interrupted",
                body=f"Signal: {sig_name} ({signum})\nEpoch: {trainer.epoch}\nBest Loss: {trainer.best_loss:.4f}",
                webhook_url=args.webhook,
            )
        cleanup_distributed()
        sys.exit(1)

    # 注册信号处理器（SIGKILL 无法捕获）
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C

    # ==================== 开始训练 ====================
    try:
        trainer.train(args.num_epochs)
    except Exception as e:
        # 捕获异常并发送通知
        if is_main_process() and args.webhook:
            send_notification(
                title=f"DMTG Training Failed",
                body=f"Error: {str(e)[:200]}\nEpoch: {trainer.epoch}\nBest Loss: {trainer.best_loss:.4f}",
                webhook_url=args.webhook,
            )
        raise
    finally:
        # 清理 DDP
        cleanup_distributed()


if __name__ == "__main__":
    main()
