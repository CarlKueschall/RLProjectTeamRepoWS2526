#!/bin/bash
#SBATCH --job-name=td3_pbrs_v2
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=20:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# ============================================================================
# TD3 Hockey Training - PBRS V3 Configuration (Balanced Offense/Defense)
# ============================================================================
#
# PBRS V3 DESIGN (Balanced Offense/Defense):
# ------------------------------------------
# Three-component potential function:
#   phi(s) = W_CHASE * phi_chase + W_ATTACK * phi_attack + W_DEFENSE * phi_defense
#          = 0.5 * chase + 0.7 * attack + 0.3 * defense
#
# KEY V3 IMPROVEMENTS:
# 1. ASYMMETRIC phi_attack: Only active in opponent half (no defensive penalty)
# 2. FULL phi_chase in our half: Stationary reduction only in opponent half
# 3. NEW phi_defense: Rewards being between puck and own goal
#
# REWARD MATRIX:
# | Situation                | chase | attack | defense | Net | Result      |
# |--------------------------|-------|--------|---------|-----|-------------|
# | Chase puck (opp half)    | +     | 0      | 0       | +   | Encouraged  |
# | Shoot toward opp goal    | -     | +      | 0       | +   | Encouraged  |
# | Chase puck (OUR half)    | +     | 0      | +       | ++  | STRONGLY!   |
# | Good defensive position  | ~     | 0      | +       | +   | Encouraged  |
# | Ignore puck in our half  | 0     | 0      | -       | -   | Penalized   |
#
# EPSILON DECAY:
#   - Only starts AFTER warmup (not during warmup)
#   - Keeps exploration high during initial buffer filling
#
# ANNEALING TIMELINE (100k episodes):
#   Episode 0-warmup:    epsilon = initial (no decay)
#   Episode warmup+:     epsilon starts decaying
#   Episode 0-5000:      PBRS weight = 1.0 (full guidance)
#   Episode 5000-20000:  PBRS weight = 1.0 -> 0.1 (gradual fade)
#   Episode 20000+:      PBRS weight = 0.1 (minimal guidance retained)
#
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
    --eps_decay 0.9995 \
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
    `# === REWARD CONFIGURATION (PBRS V2) ===` \
    `# reward_scale=0.1: Maps sparse +/-10 to +/-1 (standard RL range)` \
    `# pbrs_scale=0.02: Ensures max episode PBRS (3.0) < sparse (10)` \
    --reward_scale 0.1 \
    --reward_shaping \
    --pbrs_scale 0.02 \
    \
    `# === PBRS ANNEALING (V2 - Slow Fade) ===` \
    `# Start annealing at episode 5000 (after basic skills learned)` \
    `# Anneal over 15000 episodes (very gradual)` \
    `# Min weight 0.1 (never fully disable - retains attack incentive)` \
    --pbrs_anneal_start 5000 \
    --pbrs_anneal_episodes 15000 \
    --pbrs_min_weight 0.1 \
    \
    `# === EPSILON RESET AT ANNEALING ===` \
    `# When reward landscape changes, re-enable exploration` \
    --epsilon_reset_at_anneal \
    --epsilon_anneal_reset_value 0.4 \
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
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    \
    `# === SELF-PLAY CONFIGURATION ===` \
    `# Enable self-play at episode 8000 (after mastering weak opponent)` \
    `# 50% weak ratio maintains baseline performance` \
    `# PFSP variance mode focuses on ~50% win rate opponents` \
    --self_play_start 8000 \
    --self_play_pool_size 25 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.5 \
    --use_pfsp \
    --pfsp_mode variance \
    --episode_block_size 50 \
    \
    `# === EPSILON RESET ON SELF-PLAY ===` \
    `# Re-enable exploration when facing novel self-play opponents` \
    --epsilon_reset_on_selfplay \
    --epsilon_reset_value 0.5 \
    \
    `# === LOGGING & EVALUATION ===` \
    --log_interval 10 \
    --save_interval 500 \
    --eval_interval 1000 \
    --eval_episodes 100 \
    --gif_episodes 3
