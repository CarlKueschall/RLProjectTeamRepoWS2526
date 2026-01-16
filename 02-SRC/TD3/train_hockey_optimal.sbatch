#!/bin/bash
#SBATCH --job-name=td3-hockey-optimal
#SBATCH --output=logs/td3_optimal_%j.out
#SBATCH --error=logs/td3_optimal_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# TD3 Hockey Training - Optimal Configuration
# ============================================================================
#
# REWARD SCALING DERIVATION:
# -------------------------
# Sparse rewards: ±10 (win/loss)
# PBRS potential range: ~105 (worst to best state)
#
# With reward_scale=0.1 and pbrs_scale=0.02:
#   - Sparse in training: 10 × 0.1 = ±1.0
#   - Max episode PBRS: 105 × 0.02 = 2.1
#   - Per-step PBRS: typically ±0.02 to ±0.1
#
# Ratio check: 2.1 / 10 = 0.21 (PBRS < sparse) ✓
#
# PBRS COMPONENTS:
# ----------------
# φ_chase: Reward proximity to MOVING puck only (prevents "do nothing" exploit)
# φ_defensive: Triangle defense positioning when puck in own half
#
# NOT ENCODED (let agent discover):
# - Shot direction (bank shots, angles)
# - Puck-to-goal distance
# - Possession rewards
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 42 \
    \
    `# === TD3 HYPERPARAMETERS ===` \
    `# Exploration: Start high, decay to maintain some exploration` \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99992 \
    \
    `# Learning rates (TD3 paper defaults)` \
    --lr_actor 0.001 \
    --lr_critic 0.001 \
    \
    `# Core TD3 parameters` \
    --batch_size 100 \
    --tau 0.005 \
    --gamma 0.99 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    \
    `# === TRAINING SCHEDULE ===` \
    `# Standard TD3: train after each episode` \
    --train_freq -1 \
    --iter_fit 125 \
    --warmup_episodes 2000 \
    --buffer_size 1000000 \
    \
    `# === REWARD CONFIGURATION (CRITICAL) ===` \
    `# reward_scale=0.1: Maps sparse ±10 to ±1 (standard RL range)` \
    `# pbrs_scale=0.02: Ensures max episode PBRS (2.1) < sparse (10)` \
    `# This ratio ensures sparse rewards DOMINATE while PBRS guides exploration` \
    --reward_scale 0.1 \
    --reward_shaping \
    --pbrs_scale 0.02 \
    \
    `# === VF REGULARIZATION (Anti-Lazy) ===` \
    `# Penalizes Q(passive) > Q(active) to encourage active play` \
    --vf_reg_lambda 0.1 \
    \
    `# === Q-VALUE STABILITY ===` \
    `# Soft clipping prevents Q-explosion while allowing smooth gradients` \
    --q_clip 25.0 \
    --q_clip_mode soft \
    \
    `# === NETWORK ARCHITECTURE ===` \
    `# Larger networks for complex game dynamics` \
    --hidden_actor 400 300 \
    --hidden_critic 400 300 128 \
    \
    `# === SELF-PLAY DISABLED ===` \
    `# Focus on mastering weak opponent first` \
    `# Enable self-play after achieving >70% win rate vs weak` \
    --disable_selfplay \
    \
    `# === LOGGING & EVALUATION ===` \
    --log_interval 10 \
    --save_interval 500 \
    --eval_interval 50 \
    --eval_episodes 50 \
    --gif_episodes 3
