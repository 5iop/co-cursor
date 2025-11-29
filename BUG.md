# Bug 记录

## Bug #1: Per-sample 长度支持导致轨迹异常

**状态**: 已回退

**现象**: 启用 per-sample 长度支持后，生成的轨迹会先经过 (1,1) 点再到达终点，而不是直接从起点到终点。

**影响的函数** (`src/models/alpha_ddim.py`):

### 1. `_initialize_with_condition`
```python
# 回退前 (per-sample 版本):
def _initialize_with_condition(self, batch_size, condition, device,
                                effective_length=None):  # 支持 int 或 Tensor
    # 使用 Tensor lengths 和 advanced indexing
    lengths = effective_length.long().to(device)
    batch_indices = torch.arange(batch_size, device=device)
    end_indices = (lengths - 1).clamp(0, self.seq_length - 1)
    mask[batch_indices, end_indices, :] = 1.0
    x_c[batch_indices, end_indices, :] = end_point

# 回退后 (原版):
def _initialize_with_condition(self, batch_size, condition, device,
                                effective_length: int = None):  # 只支持 int
    m = effective_length if effective_length else self.seq_length
    end_idx = effective_length - 1
    mask[:, end_idx, :] = 1.0
    x_c[:, end_idx, :] = end_point
```

### 2. `_enforce_boundary_inpainting`
```python
# 回退前 (per-sample 版本):
def _enforce_boundary_inpainting(self, x, condition, t, total_timesteps,
                                  effective_length=None):  # 支持 int 或 Tensor
    lengths = effective_length.long().to(device)
    batch_indices = torch.arange(batch_size, device=device)
    end_indices = (lengths - 1).clamp(0, self.seq_length - 1)
    mask[batch_indices, end_indices, :] = 1.0
    x_c[batch_indices, end_indices, :] = end_point

# 回退后 (原版):
def _enforce_boundary_inpainting(self, x, condition, t, total_timesteps,
                                  effective_length: int = None):  # 只支持 int
    end_idx = effective_length - 1 if effective_length else self.seq_length - 1
    mask[:, end_idx, :] = 1.0
    x_c[:, end_idx, :] = end_point
```

### 3. `_apply_boundary_conditions`
```python
# 回退前 (per-sample 版本):
def _apply_boundary_conditions(self, trajectory, condition,
                                effective_length=None):  # 支持 int 或 Tensor
    # 使用 per-sample 权重和 padding mask
    lengths = effective_length.long().to(device)
    seq_indices = torch.arange(self.seq_length, device=device).unsqueeze(0).float()
    weights = seq_indices / (lengths_float - 1).clamp(min=1)
    current_end = trajectory[batch_indices, end_indices, :]
    result = result + correction * padding_mask

# 回退后 (原版):
def _apply_boundary_conditions(self, trajectory, condition,
                                effective_length: int = None):  # 只支持 int
    m = effective_length if effective_length else self.seq_length
    weights = torch.linspace(0, 1, m, device=device)
    traj_valid = trajectory[:, :m, :]
    current_end = traj_valid[:, -1:, :]
    traj_valid = traj_valid + correction
```

### 4. `sample()` 中的 padding 逻辑
```python
# 回退前 (per-sample 版本):
if effective_length is not None:
    if isinstance(effective_length, int):
        # int 处理
    else:
        # Tensor: per-sample padding
        seq_indices = torch.arange(self.seq_length, device=device).unsqueeze(0)
        lengths = effective_length.long().to(device)
        padding_mask = (seq_indices < lengths.unsqueeze(1)).float().unsqueeze(2)

# 回退后 (原版):
if effective_length and effective_length < self.seq_length:
    padding_mask = torch.zeros(batch_size, self.seq_length, 1, device=device)
    padding_mask[:, :effective_length, :] = 1.0
```

**可能原因**:
1. Advanced indexing 在某些情况下行为与预期不符
2. Per-sample 权重计算导致边界修正错误
3. Tensor 类型的 effective_length 在传递过程中出现问题

**临时解决方案**: 使用 `sample_with_auto_length` 时，取预测长度的平均值作为 int 传入。
