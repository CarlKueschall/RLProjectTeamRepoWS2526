#!/bin/bash
#SBATCH --job-name=td3_clean
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=20:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

###############################################################################
# TD3 TRAINING - CLEAN CODEBASE (Pure TD3 + PBRS + VF Regularization)
#
# This configuration follows TD3 paper defaults while adding:
#   - PBRS for dense reward guidance (scaled to not overwhelm sparse rewards)
#   - VF Regularization to prevent lazy/passive agents
#   - Soft Q-clipping for stability
#
# CONFIGURATION RATIONALE:
#
# 1. EXPLORATION SCHEDULE:
#    - eps=1.0 initially (full exploration to fill buffer with diverse data)
#    - Decays to eps_min=0.1 over ~60k episodes
#    - TD3 paper uses constant 0.1, but hockey needs more early exploration
#    - Random policy + low exploration = buffer full of losses = passive agent
#
# 2. TD3 PAPER DEFAULTS:
#    - Learning rates: 1e-3 for both actor and critic
#    - Batch size: 100
#    - tau: 0.005 (soft update coefficient)
#    - policy_freq: 2 (delayed policy updates)
#
# 3. REWARD BALANCE:
#    - reward_scale=0.1 maps sparse ±10 to ±1
#    - pbrs_scale=0.1 reduces PBRS magnitude to match scaled sparse rewards
#    - This ensures sparse rewards (goals) remain the dominant signal
#
# 4. VF REGULARIZATION (Anti-Lazy):
#    - vf_reg_lambda=0.1 penalizes Q(passive) > Q(active)
#    - New W&B metrics track effectiveness:
#      * vf_reg/violation_ratio: should DECREASE over training
#      * vf_reg/q_advantage_mean: should become POSITIVE
#
# 5. Q-VALUE STABILITY:
#    - q_clip=25.0 with soft (tanh) clipping prevents explosion
#    - Allows smooth gradients near boundaries
#
# EXPECTED RUNTIME: ~15-20 hours for 100k episodes
# EXPECTED RESULTS: 80-90%+ win rate vs weak opponent
###############################################################################

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 42 \
    \
    `# === EXPLORATION ===` \
    `# Start with high exploration (eps=1.0) to fill buffer with diverse experiences` \
    `# Decay to eps_min=0.1 over ~60k episodes (0.99995^60000 ≈ 0.05)` \
    `# TD3 paper uses constant 0.1, but hockey needs more early exploration` \
    --eps 1.0 \
    --eps_min 0.1 \
    --eps_decay 0.99995 \
    --lr_actor 0.001 \
    --lr_critic 0.001 \
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
    `# Standard TD3: train after each episode with ~250 gradient steps` \
    --train_freq -1 \
    --iter_fit 250 \
    --warmup_episodes 2000 \
    --buffer_size 1000000 \
    \
    `# === REWARD CONFIGURATION ===` \
    `# reward_scale=0.1 maps sparse ±10 to ±1` \
    `# pbrs_scale=0.1 balances PBRS with scaled sparse rewards` \
    `# Without this, PBRS (~±5) would dominate sparse rewards (±1)` \
    --reward_scale 0.1 \
    --reward_shaping \
    --pbrs_scale 0.1 \
    --pbrs_constant_weight \
    \
    `# === VF REGULARIZATION (Anti-Lazy) ===` \
    `# Penalizes Q(passive) > Q(active) to encourage active play` \
    `# Monitor vf_reg/violation_ratio in W&B - should decrease over training` \
    --vf_reg_lambda 0.1 \
    \
    `# === Q-VALUE STABILITY ===` \
    `# Soft clipping prevents Q-explosion while allowing smooth gradients` \
    --q_clip 25.0 \
    --q_clip_mode soft \
    \
    `# === NETWORK ARCHITECTURE ===` \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    \
    `# === SELF-PLAY DISABLED ===` \
    `# Focus on mastering weak opponent first` \
    --disable_selfplay \
    \
    `# === LOGGING & EVALUATION ===` \
    --log_interval 10 \
    --save_interval 5000 \
    --eval_interval 2000 \
    --eval_episodes 100 \
    --gif_episodes 3

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/TD3/results ~/02-SRC/TD3/

###############################################################################
# W&B METRICS TO MONITOR:
#
# TRAINING HEALTH:
#   - losses/critic_loss: Should be stable, not exploding
#   - losses/actor_loss: Should be negative (maximizing Q)
#   - gradients/critic_grad_norm: Should stay bounded (<10)
#   - gradients/actor_grad_norm: Should stay bounded (<10)
#
# VF REGULARIZATION (NEW):
#   - vf_reg/violation_ratio: % of states where Q(passive) > Q(active)
#     * START: May be 50-80% (agent prefers doing nothing)
#     * GOAL: Should decrease to <10-20% (agent learns active is better)
#   - vf_reg/q_advantage_mean: Mean Q(active) - Q(passive)
#     * START: May be negative (passive valued higher)
#     * GOAL: Should become positive (active valued higher)
#   - vf_reg/reg_loss: Regularization penalty
#     * Should decrease as agent learns active behavior
#
# PERFORMANCE:
#   - performance/cumulative_win_rate: Overall win rate
#   - eval/weak/win_rate: Evaluation win rate vs weak
#     * Target: 80%+ by episode 50k, 90%+ by episode 100k
#
# REWARD SHAPING:
#   - pbrs/avg_per_episode: Average PBRS bonus per episode
#     * Should be small relative to sparse rewards with pbrs_scale=0.1
#
###############################################################################
