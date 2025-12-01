# DMTG - Diffusion-based Mouse Trajectory Generator

基于扩散模型的鼠标轨迹生成器，使用 α 参数控制轨迹复杂度。论文: arXiv-2410.18233v1

## 快速开始

### 1. 部署数据集（必需）

使用 `aria2c` 下载预处理好的 Parquet 数据集：

```bash
# 安装 aria2（如果没有）
# Ubuntu: sudo apt install aria2
# macOS: brew install aria2
# Windows: scoop install aria2

# 下载数据集到 datasets 目录
cd datasets
aria2c -i download.txt -d . --continue=true
```

`datasets/download.txt` 包含以下数据集：
- `boun_trajectories.parquet` - BOUN 鼠标轨迹数据集
- `open_images_v6-*.parquet` - Localized Narratives Open Images V6 数据集

### 2. 训练模型

**远程服务器训练（带 webhook 通知）：**

```bash
# 使用 webhook 发送训练进度和可视化结果到远程
python train.py --num_epochs 100 --batch_size 64 --plot --webhook "ntfys://ntfy.example.com/topic"

# 多 GPU 训练
torchrun --nproc_per_node=4 train.py --num_epochs 100 --plot --webhook "your_webhook_url"
```

**本地测试训练（仅可视化）：**

```bash
# 本地训练，生成可视化图表但不发送通知
python train.py --num_epochs 10 --batch_size 32 --plot
```

训练时会自动运行以下可视化脚本（保存到 `outputs/` 目录）：
- `plot_tsne_distribution.py` - 空间特征 t-SNE 分布图
- `plot_tsne_temporal.py` - 时间特征 t-SNE 分布图
- `test_generate.py` - 轨迹生成对比图

---

## 脚本说明

### 训练脚本

| 脚本 | 用途 | 示例 |
|------|------|------|
| `train.py` | 模型训练 | `python train.py --num_epochs 100 --plot --webhook URL` |

**参数说明：**
- `--plot`: 启用可视化（保存 best_model 时自动生成图表）
- `--webhook URL`: 发送训练通知和图表到指定 webhook
- `--resume PATH`: 从检查点恢复训练

### 生成脚本

| 脚本 | 用途 | 示例 |
|------|------|------|
| `generate.py` | 单条轨迹生成 | `python generate.py --alpha 1.5 --start 100,100 --end 500,400` |
| `test_generate.py` | 批量生成+可视化 | `python test_generate.py -c checkpoints/best_model.pt --acceleration_dist` |

**test_generate.py 参数：**
- `-c, --checkpoint`: 模型检查点路径
- `--alphas`: Alpha 值列表，默认 `1.0,1.5,2.0,3.0`
- `--acceleration_dist`: 生成加速度方向分布图 (论文 Fig.6)
- `--no_display`: 不显示窗口（服务器模式）
- `--webhook URL`: 发送结果图片

### 可视化脚本

| 脚本 | 用途 | 示例 |
|------|------|------|
| `plot_tsne_distribution.py` | 空间特征 t-SNE | `python plot_tsne_distribution.py -c checkpoints/best_model.pt` |
| `plot_tsne_temporal.py` | 时间特征 t-SNE | `python plot_tsne_temporal.py --checkpoint checkpoints/best_model.pt` |

**共同参数：**
- `--device cpu/cuda`: 运行设备
- `--no_display`: 不显示窗口
- `--webhook URL`: 发送结果图片
- `--label NAME`: 输出文件标签

### 数据处理脚本

| 脚本 | 用途 | 示例 |
|------|------|------|
| `tools/preprocess_boun.py` | 预处理 BOUN 数据集 | `python tools/preprocess_boun.py` |
| `tools/convert_jsonl_to_parquet.py` | JSONL 转 Parquet | `python tools/convert_jsonl_to_parquet.py` |

### 评估脚本

| 脚本 | 用途 | 示例 |
|------|------|------|
| `evaluate.py` | 模型评估 | `python evaluate.py -c checkpoints/best_model.pt` |

---

## Webhook 通知

支持使用 [apprise](https://github.com/caronc/apprise) 发送通知，兼容多种服务：

```bash
# ntfy (推荐)
--webhook "ntfys://ntfy.example.com/topic"

# Telegram
--webhook "tgram://bot_token/chat_id"

# Discord
--webhook "discord://webhook_id/webhook_token"
```

环境变量: `DMTG_WEBHOOK_URL`

---

## 实现说明

### 当前实现概览

| 组件 | 实现方式 | 论文依据 |
|------|----------|----------|
| α 语义 | path_ratio ∈ [1, +∞)，α=1 为直线 | Eq.10 |
| StyleEmb | α' = 1/α → PosEmb(⌊α'·S⌋) | Eq.10 |
| L_style | L1 loss: \|α - ratio(p_a)\| | Eq.13 |
| L_sim | MSE: \|\|p_a - X̂\|\|² | Eq.12 |
| L_length | L1 loss: \|log(m+1) - log(m̂+1)\| | 新增 |
| 双协方差混合 | z_mixed = √(1-a)·z_d + √a·z_n | Eq.4-6 |
| MST 熵 | H ≈ log(L_path) - ½·log(m) + log(2πe) | Eq.8-9 |

### α 参数说明

```
α = path_ratio = 路径长度 / 直线距离
α ≥ 1，α=1 表示完美直线
```

| α 值 | 含义 | 用途 |
|------|------|------|
| 1.0 | 直线 | 最简单轨迹 |
| 1.2-1.5 | 轻微弯曲 | 日常点击 |
| 1.5-2.0 | 中等复杂 | 默认推荐 |
| 2.0-3.0 | 较复杂 | 拖拽操作 |
| 3.0+ | 非常复杂 | 特殊场景 |

**内部转换：**

```python
# StyleEmb 使用 α' = 1/α 映射到 (0, 1]
alpha_prime = 1.0 / alpha.clamp(min=1.0)

# 双协方差混合系数 mixing_coef = 1 - 1/α ∈ [0, 1)
mixing_coef = 1.0 - 1.0 / max(alpha, 1.0)
```

### 损失函数权重

默认配置 (`config.py`):

```python
DMTGLoss(
    lambda_ddim=1.0,    # L_DDIM: 噪声预测 MSE
    lambda_sim=0.1,     # L_sim: 轨迹相似度 MSE
    lambda_style=0.05,  # L_style: 复杂度匹配 L1
    lambda_length=0.1,  # L_length: 长度预测 L1
)
```

### 架构说明

**UNet 嵌入分离：**

```
Encoder: time_emb + condition_emb (不含 α)
Decoder: time_emb + condition_emb + alpha_emb
```

论文要求 encoder 仅由时间步 t 控制，α 仅影响 decoder。

**长度预测：**

使用 Shared Encoder 模式，从 bottleneck 特征预测轨迹长度：

```python
log_length = model.predict_length(x, condition, alpha)
decoded_length = torch.round(torch.exp(log_length) - 1)
```

### 已验证的设计选择

1. **方向协方差矩阵 Σ_X 使用秩1近似**
   - 代码使用 `z_d = σ_d · ε · d̂` 实现方向噪声
   - Σ_X = k_c² · d ⊗ d 本身就是秩1矩阵
   - 代码协方差 E[z_d·z_d^T] = k_c² · d ⊗ d = Σ_X ✓

2. **L_sim 使用 predicted_x0**
   - 使用单步预测 x̂_0 而非完整采样是 diffusion 训练的标准做法
   - 完整采样需 50-1000 步，计算成本极高

3. **混合协方差语义**
   - α=1 → mixing_coef=0 → 纯方向噪声 → 直线 ✓
   - α→∞ → mixing_coef→1 → 纯各向同性 → 复杂 ✓

---

## 项目结构

```
co-cursor/
├── train.py                    # 训练入口
├── generate.py                 # 单条轨迹生成
├── test_generate.py            # 批量生成+可视化
├── evaluate.py                 # 模型评估
├── plot_tsne_distribution.py   # 空间特征 t-SNE
├── plot_tsne_temporal.py       # 时间特征 t-SNE
├── config.py                   # 配置文件
├── src/
│   ├── models/
│   │   ├── alpha_ddim.py       # α-DDIM 扩散采样器
│   │   ├── unet.py             # TrajectoryUNet 模型
│   │   └── losses.py           # 损失函数
│   ├── data/
│   │   └── dataset.py          # 数据集加载
│   └── utils/
│       └── notify.py           # Webhook 通知
├── tools/                      # 数据处理工具
├── datasets/
│   └── download.txt            # 数据集下载链接（必需）
├── checkpoints/                # 模型检查点
├── outputs/                    # 输出图表
└── tests/                      # 单元测试
```

---

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_alpha_ddim_runtime.py -k entropy
```

---

## 相关文档

- `CAUTION.md` - 详细实现说明
- `CLAUDE.md` - Claude Code 指南
- `docs/2410.18233v1.tex` - 论文原文
