# Changelog

## [Unreleased] - 2025-12-01

### Webhook 通知修复

- **notify.py**: 从 apprise URL schema 改为直接 HTTP 请求
  - URL 格式: `https://host/notify/key` (Apprise API 服务器)
  - 使用 `requests.post()` 发送 form data 和文件附件
  - 支持图片附件发送
- **可视化脚本**: 统一使用 `send_image_result()` 发送通知

---

### 重大变更：轨迹维度扩展 2D → 3D

本次更新将模型输入从 **2D (x, y)** 扩展到 **3D (x, y, dt)**，增加了时间间隔信息，使生成的轨迹更加真实。

---

### 模型架构变更

#### config.py
- **移除** `sapimouse_dir` 配置项
- **`base_channels`**: 64 → 128 (模型参数量 4.7M → 17M)
- **`input_dim`**: 2 → 3 (新增 dt 时间维度)

#### src/models/unet.py
- **输入维度**: `input_dim` 默认值 2 → 3
- **LengthPredictionHead 增强**:
  - `hidden_dim`: 128 → 256
  - `num_layers`: 3 → 5 (可配置的 MLP 层数)
- 更新文档注释，说明输入/输出格式为 `(batch, seq_len, 3)`

#### src/models/alpha_ddim.py
- **`input_dim`**: 默认值 2 → 3
- **双协方差混合噪声** (Eq.4-6):
  - 方向约束噪声 `z_d` 仅应用于 x, y 维度
  - dt 维度使用纯各向同性噪声
  - 混合公式: `z_mixed[:, :, :2] = √(1-a)·z_d + √a·z_n[:, :, :2]`
- **边界条件 (inpainting)**:
  - mask 仅对 x, y 维度有效，dt 维度不受约束
  - `x_c` 条件值仅设置 x, y 维度
- **边界修正 `_apply_boundary_condition()`**:
  - 插值修正仅应用于 x, y 维度
  - dt 维度保持不变
- **复杂度计算 `compute_alpha()`**:
  - 路径长度和直线距离仅使用 x, y 维度计算
- **`create_alpha_ddim()` 工厂函数**:
  - `base_channels` 默认值 64 → 128
  - 新增 `input_dim=3` 参数

---

### 数据集重构

#### src/data/dataset.py
- **移除的功能**:
  - SapiMouse 数据集支持 (`_load_sapimouse()`)
  - JSONL 格式支持 (`_load_jsonl()`, `_process_jsonl_file()`, `_process_traces()`)
  - CSV 处理 (`_process_csv_file()`, `_split_into_trajectories()`)
  - 归一化相关参数 (`normalize`, `screen_size`, `_normalize_coords()`)
  - `timestamps` 字段从返回的 dict 中移除

- **新增功能**:
  - **dt 归一化**: `normalize_dt(dt) = log(dt + 1) / log(1001)`
  - **dt 反归一化**: `denormalize_dt(dt_norm) = exp(dt_norm * log(1001)) - 1`
  - `DT_LOG_SCALE = log(1001) ≈ 6.91`，假设大部分 dt < 1000ms

- **轨迹格式变更**:
  - 存储格式: `(N, 2)` → `(N, 3)` 即 `[x, y]` → `[x, y, dt_norm]`
  - Padding 值: `[0, 0]` → `[0, 0, 0]`
  - `start_point` / `end_point` 仍为 `(2,)` (仅 x, y)

- **支持的数据集类型**:
  - `boun_parquet` - BOUN Parquet 格式
  - `open_images_parquet` - Open Images Parquet 格式

- **Parquet 列变更**: `t` (时间戳) → `dt` (时间差)

#### CombinedMouseDataset
- 移除 `sapimouse_dir` 参数
- 移除 `normalize` 参数
- 移除 JSONL 格式回退逻辑

---

### 数据预处理工具

#### tools/preprocess_boun.py
- **输出列变更**:
  - 移除: `trajectory_id`, `test_type`, `session_id`
  - 保留: `x`, `y`, `user_id`
  - 新增: `dt` (时间差，毫秒，dt[0]=0)
- **时间处理**: 从绝对时间戳 `t` 改为时间差 `dt`
  - `dt[i] = t[i] - t[i-1]` (毫秒)
  - `dt[0] = 0`
- **新增参数**: `--min_straight_dist` (默认 0.01)
- **移除功能**:
  - `--no-clean` 参数
  - `--analyze` 参数和 `analyze_dataset()` 函数

#### tools/convert_jsonl_to_parquet.py
- **输出列变更**: `t` → `dt`
- **时间处理**:
  - 原始时间戳单位: 秒
  - 输出 dt 单位: 毫秒
  - `dt[i] = (t[i] - t[i-1]) * 1000`
- **新增参数**: `--min_straight_dist` (默认 0.01)
- **`--merge` 参数改进**: 现在可直接指定输出文件名
- **新增过滤统计**: 输出被过滤的轨迹数量

---

### 训练脚本

#### train.py
- **移除** `--sapimouse_dir` 参数
- **`--base_channels`**: 默认值 64 → 128
- **`input_dim`**: 硬编码为 3
- 移除 SapiMouse 路径检查逻辑

---

### 评估与可视化

#### test_generate.py
- **`load_model()`**: 添加 `input_dim=3` 参数
- **`compute_trajectory_metrics()`**:
  - 支持 `(N, 2)` 或 `(N, 3)` 输入
  - 几何指标仅使用 x, y
  - 新增 dt 相关指标: `total_dt_norm`, `mean_dt_norm`
- **`compute_acceleration_directions()`**: 仅使用 x, y 计算
- **`classify_trajectory_direction()`**: 仅使用 y 坐标判断

#### plot_tsne_distribution.py
- **移除** SapiMouse 相关代码和颜色映射
- **`extract_trajectory_features()`**: 支持 `(N, 3)` 输入，仅使用 x, y 提取特征
- **`load_human_trajectories()`**: 移除 `sapimouse_dir` 参数
- **`create_alpha_ddim()`**: 添加 `input_dim=3`

---

### 迁移指南

1. **重新预处理数据集**:
   ```bash
   python tools/preprocess_boun.py --min_straight_dist 0.01
   python tools/convert_jsonl_to_parquet.py --min_straight_dist 0.01
   ```

2. **更新模型加载代码**:
   ```python
   # 旧代码
   model = create_alpha_ddim(seq_length=500, device=device)

   # 新代码
   model = create_alpha_ddim(seq_length=500, input_dim=3, device=device)
   ```

3. **训练新模型**: 由于输入维度变更，旧的 checkpoint 不兼容，需要重新训练

4. **数据集格式**: Parquet 文件需包含 `dt` 列而非 `t` 列
