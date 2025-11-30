# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DMTG (Diffusion-based Mouse Trajectory Generator) generates human-like mouse cursor trajectories using a conditional diffusion model with entropy control via the α (alpha) parameter. Based on arXiv-2410.18233v1.

## Common Commands

```bash
# Training
python train.py --num_epochs 100 --batch_size 64
torchrun --nproc_per_node=4 train.py  # Multi-GPU DDP

# Generation
python generate.py --checkpoint checkpoints/best_model.pt --alpha 1.5 --start 100,100 --end 500,400
python test_generate.py --checkpoint checkpoints/best_model.pt --acceleration_dist
python test_generate.py --checkpoint checkpoints/best_model.pt --no_display  # Server mode

# Evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt

# Testing
pytest tests/ -v
pytest tests/test_alpha_ddim_runtime.py -k entropy  # Specific test

# Data preprocessing
python tools/preprocess_boun.py
python tools/convert_jsonl_to_parquet.py
```

## Architecture

### Core Components

- **`src/models/alpha_ddim.py`**: α-DDIM diffusion sampler with dual covariance mixing (Eq.4-6) and MST entropy control (Eq.8-9)
- **`src/models/unet.py`**: 1D TrajectoryUNet for noise prediction. α embedding applied ONLY in decoder per paper requirement
- **`src/models/losses.py`**: DMTGLoss combining L_DDIM + L_sim + L_style + L_length (Eq.11-14)
- **`src/data/dataset.py`**: MouseTrajectoryDataset supporting SapiMouse (CSV), BOUN (Parquet), Open Images formats

### α Parameter Semantics

```
α = path_ratio = path_length / straight_distance
α ∈ [1, +∞), where α=1 means perfect straight line

Values:
- α=1.0: Straight line
- α=1.2-1.5: Slight curves (typical clicks)
- α=1.5-2.0: Medium complexity (default=1.5)
- α=2.0-3.0: Complex trajectories
- α=3.0+: Very complex curves

Internal conversions:
- StyleEmb: α' = 1/α → maps to (0, 1]
- Mixing coefficient: a = 1 - 1/α → α=1 gives a=0 (directional), α→∞ gives a→1 (isotropic)
```

### Dual Covariance Mixing (Eq.4-6)

```python
# Direction-constrained noise (straight paths)
z_d = σ_d · ε · d̂  # where d̂ = unit vector from start to end

# Isotropic noise (complex paths)
z_n ~ N(0, σ_n²·I)

# Mixed: z = √(1-a)·z_d + √a·z_n
# where a = 1 - 1/α
```

### Loss Function Weights (config.py)

```python
lambda_ddim=1.0    # Noise prediction MSE (Eq.11)
lambda_sim=0.1     # Trajectory similarity MSE (Eq.12)
lambda_style=0.05  # Complexity matching L1 (Eq.13)
lambda_length=0.1  # Length prediction L1
```

## Key Design Decisions

1. **Rank-1 directional covariance**: `z_d = σ_d · ε · d̂` correctly implements Eq.5
2. **L_sim uses predicted_x0**: Single-step prediction, not full 50-step sampling (standard diffusion practice)
3. **α embedding decoder-only**: Encoder uses time_emb + condition_emb; decoder adds alpha_emb
4. **Boundary inpainting**: Start/end points enforced during denoising, gradually released

## Webhook Notifications

Scripts support `--webhook URL` flag for remote training notifications using apprise:
- `train.py`: Sends training progress, best model alerts
- `test_generate.py`, `plot_tsne_distribution.py`: Send result images

Environment variable: `DMTG_WEBHOOK_URL`
