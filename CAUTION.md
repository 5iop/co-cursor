# CAUTION - 已知问题和待验证改动(都实现过发现不对劲然后回滚的问题)

## Style Loss 公式问题

**状态**: 已回滚 (clamp 版本导致梯度消失)

**改动位置**:
- `src/models/losses.py` - `_style_loss` 函数

**问题分析**:

曾尝试将 `pred_complexity` 公式改为:
```python
# 有问题的公式: clamp 导致梯度消失
pred_complexity = torch.clamp((mst_ratio - 1.0) / 2.0, 0.0, 1.0)
```

**为什么 clamp 版本有问题**:

1. 当 `ratio > 3` 时，`(ratio - 1) / 2 > 1.0`
2. `clamp(..., 1.0)` 强制截断为 1.0
3. **在截断区域，clamp 的梯度为 0**
4. 模型无法对高复杂度轨迹 (ratio > 3) 进行优化

**论文参考 (Eq.13)**:

论文直接使用 `ratio = path_length / straight_dist`，**不做 clamp**。

**当前正确实现**:

```python
# 旧公式: (ratio - 1) / ratio = 1 - 1/ratio
# ratio=1 → 0, ratio=2 → 0.5, ratio→∞ → 1
# 梯度: 1/ratio²，永远不为 0
pred_complexity = (mst_ratio - 1.0) / mst_ratio
```

这个公式的优点:
- 渐近趋向 1，但梯度永不为 0
- 对任意 ratio 值都能提供优化方向

---

## 中点偏移初始化问题

**状态**: 已放弃（实测效果不佳）

**改动位置**:
- `src/models/alpha_ddim.py` - `_initialize_with_condition` 函数

**论文建议 (Eq.3)**:

```
X_R = {p_0} ∥ {p_c + ε_i}^m ∥ {p_m} ∥ {0}^(N-m)
```

其中 `p_c = (p_0 + p_m) / 2` 是起终点的中点，中间点应初始化为 `p_c + ε`。

**实际实现**:

```python
# 论文建议: middle_points = midpoint + scaled_noise
# 实际使用: middle_points = scaled_noise (无中点偏移)
middle_points = scaled_noise
```

**放弃原因**:

添加中点偏移后，实测生成轨迹质量下降，t-SNE 分布异常。可能原因：
1. 中点偏移与扩散过程的噪声调度不匹配
2. 边界强制 (boundary inpainting) 机制已隐式处理了起终点约束
3. 初始化时添加中点可能干扰去噪过程的收敛

**如需重新测试**:

```python
midpoint = (start_point + end_point) / 2
midpoint_expanded = midpoint.unsqueeze(1)
middle_points = midpoint_expanded + scaled_noise
```
