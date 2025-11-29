# 共享 Encoder 长度预测方案

## 1. 当前架构 vs 目标架构

### 当前架构（Early Fusion）
```
condition ──→ condition_mlp ──┐
                              ├──→ LengthHead ──→ log(m+1)
alpha ──────→ alpha_embedding ┘

x_t + time + cond + alpha ──→ [Encoder] ──→ [Decoder] ──→ ε_θ
```

**问题**：长度预测只用了浅层嵌入，无法利用 Encoder 的深层特征。

### 目标架构（Shared Encoder）
```
                              ┌──→ LengthHead ──→ log(m+1)
x_proxy + cond + alpha ──→ [Encoder] ──→ bottleneck
                              └──→ [Decoder] ──→ ε_θ (训练时)
```

**关键问题**：长度预测发生在采样之前，没有 x_t，需要一个代理输入。

---

## 2. 设计方案

### 方案 A：直线轨迹代理（推荐）

**核心思想**：用起点到终点的直线轨迹作为代理输入，让 Encoder 提取条件相关的特征。

```python
# 构造代理轨迹
def create_proxy_trajectory(condition, seq_length):
    """
    创建从起点到终点的直线轨迹作为代理输入
    condition: (batch, 4) - [start_x, start_y, end_x, end_y]
    """
    start = condition[:, :2]  # (batch, 2)
    end = condition[:, 2:4]   # (batch, 2)

    # 线性插值
    t = torch.linspace(0, 1, seq_length, device=condition.device)
    t = t.view(1, -1, 1)  # (1, seq_len, 1)

    proxy = start.unsqueeze(1) * (1 - t) + end.unsqueeze(1) * t
    return proxy  # (batch, seq_length, 2)
```

**优点**：
- 直线轨迹包含了起点、终点、距离等几何信息
- Encoder 可以学习到"这条直线在不同 α 下应该变成多长的轨迹"
- 实现简单，复用现有 Encoder

**架构改动**：

```
                                    ┌──→ GlobalPool ──→ LengthHead ──→ log(m+1)
proxy_traj + t=0 + cond + alpha ──→ [Encoder] ──→ bottleneck
                                    └──→ [Decoder] ──→ ε_θ (训练时用真实 x_t)
```

---

## 3. 具体实现

### 3.1 修改 TrajectoryUNet

```python
class TrajectoryUNet(nn.Module):
    def __init__(
        self,
        ...
        enable_length_prediction: bool = True,
        length_prediction_mode: str = "shared_encoder",  # "early_fusion" | "shared_encoder"
    ):
        ...
        self.length_prediction_mode = length_prediction_mode

        if enable_length_prediction:
            if length_prediction_mode == "shared_encoder":
                # 共享 Encoder 模式：从 bottleneck 特征预测
                # bottleneck 维度 = base_channels * channel_mults[-1]
                bottleneck_dim = base_channels * channel_mults[-1]
                self.length_head = SharedEncoderLengthHead(
                    bottleneck_dim=bottleneck_dim,
                    hidden_dim=time_emb_dim,
                )
            else:
                # 早期融合模式（当前实现）
                self.length_head = LengthPredictionHead(
                    input_dim=time_emb_dim * 2,
                    hidden_dim=time_emb_dim,
                )
```

### 3.2 新增 SharedEncoderLengthHead

```python
class SharedEncoderLengthHead(nn.Module):
    """
    共享 Encoder 的长度预测头

    从 Encoder 的 bottleneck 特征预测轨迹长度
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,  # Encoder bottleneck 通道数
        hidden_dim: int = 128,
    ):
        super().__init__()

        # 全局池化后的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bottleneck: Encoder 输出 (batch, channels, seq_len')

        Returns:
            log_length: (batch, 1)
        """
        # 全局平均池化
        pooled = bottleneck.mean(dim=-1)  # (batch, channels)

        # MLP 预测
        return self.mlp(pooled)
```

### 3.3 修改 predict_length 方法

```python
def predict_length(
    self,
    condition: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """预测轨迹长度"""

    if self.length_prediction_mode == "shared_encoder":
        # 1. 创建代理轨迹（直线）
        proxy_traj = self._create_proxy_trajectory(condition)

        # 2. 使用 t=0（代表"干净"状态）
        batch_size = condition.shape[0]
        t = torch.zeros(batch_size, dtype=torch.long, device=condition.device)

        # 3. 通过 Encoder 获取 bottleneck 特征
        bottleneck = self._encode(proxy_traj, t, condition, alpha)

        # 4. 通过 length_head 预测
        return self.length_head(bottleneck)

    else:
        # 早期融合模式（当前实现）
        cond_emb = self.condition_mlp(condition)
        alpha_emb = self.alpha_embedding(alpha)
        return self.length_head(cond_emb, alpha_emb)

def _create_proxy_trajectory(self, condition: torch.Tensor) -> torch.Tensor:
    """创建直线代理轨迹"""
    batch_size = condition.shape[0]
    device = condition.device

    start = condition[:, :2]
    end = condition[:, 2:4]

    t = torch.linspace(0, 1, self.seq_length, device=device)
    t = t.view(1, -1, 1).expand(batch_size, -1, 2)

    proxy = start.unsqueeze(1) * (1 - t) + end.unsqueeze(1) * t
    return proxy

def _encode(
    self,
    x: torch.Tensor,
    time: torch.Tensor,
    condition: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """只运行 Encoder 部分，返回 bottleneck 特征"""

    # 转换为 (batch, channels, seq_len)
    x = x.transpose(1, 2)

    # 嵌入
    time_emb = self.time_mlp(time)
    if condition is not None:
        time_emb = time_emb + self.condition_mlp(condition)
    if alpha is not None:
        time_emb = time_emb + self.alpha_embedding(alpha)

    # 输入投影
    h = self.input_proj(x)

    # 下采样（Encoder）
    for down_block in self.down_blocks:
        h, _ = down_block(h, time_emb)

    # 中间块
    h = self.mid_block1(h, time_emb)
    h = self.mid_attention(h)
    h = self.mid_block2(h, time_emb)

    return h  # bottleneck: (batch, channels, seq_len')
```

---

## 4. 训练策略

### 4.1 联合训练

```python
def get_loss(self, x_0, condition, t, alpha, mask, length):
    """计算总损失"""

    # 1. 噪声预测损失（原有）
    noise = torch.randn_like(x_0)
    x_t, _ = self.q_sample(x_0, t, noise)
    predicted_noise = self.model(x_t, t, condition, alpha)
    ddim_loss = F.mse_loss(predicted_noise, noise)

    # 2. 长度预测损失（使用共享 Encoder）
    if self.model.length_prediction_mode == "shared_encoder":
        # 方式 A：使用代理轨迹（推理一致）
        log_length = self.model.predict_length(condition, alpha)

        # 方式 B：使用真实轨迹 x_0（更强监督，但推理时用代理）
        # bottleneck = self.model._encode(x_0, torch.zeros_like(t), condition, alpha)
        # log_length = self.model.length_head(bottleneck)

    target_log_length = torch.log(length.float() + 1)
    length_loss = F.l1_loss(log_length.squeeze(), target_log_length)

    return {
        'ddim_loss': ddim_loss,
        'length_loss': length_loss,
        'total_loss': ddim_loss + 0.1 * length_loss,
    }
```

### 4.2 可选：双路径训练

训练时可以同时使用代理轨迹和真实轨迹来增强长度预测：

```python
# 代理路径损失（与推理一致）
log_length_proxy = model.predict_length(condition, alpha)  # 用直线
loss_proxy = F.l1_loss(log_length_proxy, target)

# 真实路径损失（更强监督）
bottleneck_real = model._encode(x_0, t_zero, condition, alpha)  # 用真实轨迹
log_length_real = model.length_head(bottleneck_real)
loss_real = F.l1_loss(log_length_real, target)

# 总长度损失
length_loss = 0.5 * loss_proxy + 0.5 * loss_real
```

---

## 5. 实现步骤

### Step 1: 新增 SharedEncoderLengthHead 类
- 文件: `src/models/unet.py`
- 新增全局池化 + MLP 的长度预测头

### Step 2: 修改 TrajectoryUNet
- 添加 `length_prediction_mode` 参数
- 添加 `_create_proxy_trajectory()` 方法
- 添加 `_encode()` 方法（只运行 Encoder）
- 修改 `predict_length()` 支持两种模式

### Step 3: 修改 AlphaDDIM.get_loss()
- 支持新的长度预测模式
- 可选：添加双路径训练

### Step 4: 修改训练脚本
- 添加 `--length_mode` 参数
- 更新 checkpoint 保存/加载

### Step 5: 测试和验证
- 对比两种模式的长度预测精度
- 检查是否影响噪声预测性能

---

## 6. 预期效果

| 指标 | Early Fusion | Shared Encoder |
|------|-------------|----------------|
| 长度预测精度 | 基础 | 预期更好 |
| 参数量 | length_head 独立 | 复用 Encoder |
| 计算量 | 低 | 需要过一遍 Encoder |
| 特征表达 | 浅层嵌入 | 深层几何特征 |

**核心假设**：Encoder 能够学习到"给定起终点和复杂度，轨迹应该有多长"的几何先验，这比简单的嵌入拼接更有表达力。

---

## 7. 备选方案

### 方案 B：可学习的代理嵌入

不使用直线轨迹，而是学习一个"查询嵌入"：

```python
self.length_query = nn.Parameter(torch.randn(1, seq_length, input_dim))

def predict_length(self, condition, alpha):
    # 用可学习的查询代替直线
    query = self.length_query.expand(batch_size, -1, -1)
    bottleneck = self._encode(query, t=0, condition, alpha)
    return self.length_head(bottleneck)
```

### 方案 C：Cross-Attention

使用条件作为 query，Encoder 特征作为 key/value：

```python
# 需要修改架构，增加 cross-attention 层
```

---

## 8. 建议

**推荐先实现方案 A（直线代理）**，原因：
1. 实现最简单，改动最小
2. 直线轨迹是物理上合理的初始状态
3. 容易验证效果，可以和当前方案对比
4. 如果效果不好，再尝试方案 B 或 C
