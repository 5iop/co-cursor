# CAUTION - 已知问题和待验证改动(都实现过发现不对劲然后回滚的问题)

## Style Loss 公式问题

**状态**: 待验证

**改动位置**:
- `src/models/losses.py` - `_style_loss` 函数
- `tests/test_losses.py` - `test_style_loss_complexity_formula` 测试

**改动内容**:

将 `pred_complexity` 公式从:
```python
# 旧公式: β/(β+1)，其中 β = ratio - 1
pred_complexity = (mst_ratio - 1.0) / mst_ratio
```

改为:
```python
# 新公式: (ratio - 1) / 2，与 compute_trajectory_alpha 一致
pred_complexity = torch.clamp((mst_ratio - 1.0) / 2.0, 0.0, 1.0)
```

**改动原因**:

为了让 `_style_loss` 中的复杂度计算与 `compute_trajectory_alpha` 保持一致:
- `compute_trajectory_alpha`: `alpha = (ratio - 1) / 2`
- 旧 `pred_complexity`: `(ratio - 1) / ratio` (即 `β/(β+1)`)

当 ratio=3 时:
- `compute_trajectory_alpha` 返回 `alpha = 1.0`
- 旧 `pred_complexity` 返回 `0.67`

这种不一致可能导致训练时的目标不匹配。

**潜在问题**:

此改动可能影响模型训练效果，需要验证:
1. t-SNE 分布是否正常
2. 生成轨迹质量是否下降
3. style loss 的收敛情况

**回滚方法**:

如需回滚，将 `src/models/losses.py` 中的公式改回:
```python
pred_complexity = (mst_ratio - 1.0) / mst_ratio
```

并更新 `tests/test_losses.py` 中的测试。

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
