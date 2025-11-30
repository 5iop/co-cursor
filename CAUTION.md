# CAUTION - 实现说明与注意事项

**更新日期**: 2025-11-30

---

## 当前实现概览

| 组件 | 实现方式 | 论文依据 |
|------|----------|----------|
| α 语义 | path_ratio ∈ [1, +∞)，α=1 为直线 | Eq.10 |
| StyleEmb | α' = 1/α → PosEmb(⌊α'·S⌋) | Eq.10 |
| L_style | L1 loss: \|α - ratio(p_a)\| | Eq.13 |
| L_sim | MSE: \|\|p_a - X̂\|\|² | Eq.12 |
| L_length | L1 loss: \|log(m+1) - log(m̂+1)\| | 新增 |
| 双协方差混合 | z_mixed = √(1-a)·z_d + √a·z_n | Eq.4-6 |
| MST 熵 | H ≈ log(L_path) - ½·log(m) + log(2πe) | Eq.8-9 |

---

## α 参数说明

### 语义

```
α = path_ratio = 路径长度 / 直线距离
α ≥ 1，α=1 表示完美直线
```

### 常用值

| α 值 | 含义 | 用途 |
|------|------|------|
| 1.0 | 直线 | 最简单轨迹 |
| 1.2-1.5 | 轻微弯曲 | 日常点击 |
| 1.5-2.0 | 中等复杂 | 默认推荐 |
| 2.0-3.0 | 较复杂 | 拖拽操作 |
| 3.0+ | 非常复杂 | 特殊场景 |

### 内部转换

```python
# StyleEmb 使用 α' = 1/α 映射到 (0, 1]
alpha_prime = 1.0 / alpha.clamp(min=1.0)

# 双协方差混合系数 mixing_coef = 1 - 1/α ∈ [0, 1)
mixing_coef = 1.0 - 1.0 / max(alpha, 1.0)
```

---

## 损失函数权重

默认配置 (`losses.py`):

```python
DMTGLoss(
    lambda_ddim=1.0,    # L_DDIM: 噪声预测 MSE
    lambda_sim=0.1,     # L_sim: 轨迹相似度 MSE
    lambda_style=0.05,  # L_style: 复杂度匹配 L1
    lambda_length=0.1,  # L_length: 长度预测 L1
)
```

---

## 使用示例

### 生成轨迹

```bash
# 直线轨迹
python generate.py --alpha 1.0 --start 100,100 --end 500,400

# 中等复杂度（默认）
python generate.py --alpha 1.5 --start 100,100 --end 500,400

# 复杂曲线
python generate.py --alpha 3.0 --start 100,100 --end 500,400
```

### 测试生成

```bash
# 生成对比图 + 加速度分布图
python test_generate.py --checkpoint checkpoints/best_model.pt --acceleration_dist

# 不显示窗口（后台运行）
python test_generate.py --checkpoint checkpoints/best_model.pt --no_display
```

### 运行测试

```bash
python -m pytest tests/ -v
```

---

## 架构说明

### UNet 嵌入分离

```
Encoder: time_emb + condition_emb (不含 α)
Decoder: time_emb + condition_emb + alpha_emb
```

论文要求 encoder 仅由时间步 t 控制，α 仅影响 decoder。

### 长度预测

使用 Shared Encoder 模式，从 bottleneck 特征预测轨迹长度：

```python
log_length = model.predict_length(x, condition, alpha)
decoded_length = torch.round(torch.exp(log_length) - 1)
```

---

## 已验证的设计选择

以下经分析确认为正确实现或合理设计选择：

### 1. 方向协方差矩阵 Σ_X 使用秩1近似

代码使用 `z_d = σ_d · ε · d̂` 实现方向噪声，这正是论文 Eq.5 的正确实现：
- Σ_X = k_c² · d ⊗ d 本身就是秩1矩阵
- 代码协方差 E[z_d·z_d^T] = k_c² · d ⊗ d = Σ_X ✓

### 2. L_sim 使用 predicted_x0

使用单步预测 x̂_0 而非完整采样是 diffusion 训练的标准做法：
- 完整采样需 50-1000 步，计算成本极高
- predicted_x0 是对最终轨迹的无偏估计

### 3. 混合协方差语义

代码映射 `mixing_coef = 1 - 1/α` 语义正确：
- α=1 → mixing_coef=0 → 纯方向噪声 → 直线 ✓
- α→∞ → mixing_coef→1 → 纯各向同性 → 复杂 ✓

---

## 相关文档

- `tests/` - 单元测试（92 个测试）
- `docs/2410.18233v1.tex` - 论文原文
