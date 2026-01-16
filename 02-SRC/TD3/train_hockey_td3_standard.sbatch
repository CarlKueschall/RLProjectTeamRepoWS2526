#!/bin/bash
#SBATCH --job-name=td3_paper
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=12:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

###############################################################################
# TD3 TRAINING - EXACT PAPER CONFIGURATION
#
# All parameters now match the TD3 paper (Fujimoto et al., 2018):
#   - Gaussian noise N(0, 0.1) instead of OU noise
#   - Learning rates: 10^-3 for both actor and critic
#   - Batch size: 100
#   - No observation normalization
#   - No weight decay / regularization
#   - Constant exploration (no noise decay)
#
# FIXES APPLIED:
#   1. SmoothL1Loss for critic loss (TD3 paper)
#   2. GaussianNoise instead of OUNoise
#   3. Standard Adam optimizer (no custom betas/weight_decay)
#   4. normalize_obs=False by default
#   5. iter_fit=125 (~1 update per env step, matching avg episode length)
#   6. Uniform replay (simple and stable)
#
# EXPECTED SPEED:
#   - Batch size 100, iter_fit=125
#   - Should see ~0.3-0.5 episodes/second (2-3 sec/episode)
#   - 20k episodes in ~15-20 hours
###############################################################################

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# Run training with TD3 paper configuration
# Most parameters use paper defaults, only specify non-defaults and hockey-specific settings
singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 20000 \
    --seed 42 \
    \
    `# === TD3 PAPER DEFAULTS (now automatic) ===` \
    `# --eps 0.1                    # N(0, 0.1) Gaussian noise` \
    `# --eps_decay 1.0              # No decay (constant noise)` \
    `# --lr_actor 0.001             # Paper default` \
    `# --lr_critic 0.001            # Paper default` \
    `# --batch_size 100             # Paper default` \
    `# --tau 0.005                  # Paper default` \
    `# --policy_freq 2              # Paper default` \
    `# --target_noise_std 0.2      # Paper default` \
    `# --target_noise_clip 0.5     # Paper default` \
    \
    `# === TRAINING CONFIGURATION ===` \
    --train_freq -1 \
    --iter_fit 125 \
    --warmup_episodes 1000 \
    --buffer_size 1000000 \
    \
    `# === Q-VALUE STABILITY ===` \
    --q_clip 10.0 \
    --q_clip_mode soft \
    \
    `# === REWARD SHAPING ===` \
    --reward_scale 0.1 \
    --reward_shaping \
    --pbrs_scale 0.5 \
    --pbrs_constant_weight \
    \
    `# === DISABLE SELF-PLAY ===` \
    --disable_selfplay \
    \
    `# === LOGGING ===` \
    --eval_interval 500 \
    --eval_episodes 50 \
    --log_interval 50 \
    --save_interval 1000 \
    --gif_episodes 3

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/TD3/results ~/02-SRC/TD3/

###############################################################################
# PARAMETER SUMMARY (TD3 Paper vs Our Implementation):
#
# | Parameter              | Paper    | Ours     | Match? |
# |------------------------|----------|----------|--------|
# | Exploration            | N(0,0.1) | N(0,0.1) | YES    |
# | Actor LR               | 10^-3    | 10^-3    | YES    |
# | Critic LR              | 10^-3    | 10^-3    | YES    |
# | Batch Size             | 100      | 100      | YES    |
# | tau                    | 0.005    | 0.005    | YES    |
# | Policy Freq (d)        | 2        | 2        | YES    |
# | Target Noise std       | 0.2      | 0.2      | YES    |
# | Target Noise clip      | 0.5      | 0.5      | YES    |
# | Normalized Obs         | False    | False    | YES    |
# | Critic Regularization  | None     | None     | YES    |
# | Optimizer              | Adam     | Adam     | YES    |
#
# HOCKEY-SPECIFIC (not from paper):
#   - iter_fit=125 (~1 update per env step, avg episode ~125 steps)
#   - Uniform replay buffer
#   - PBRS reward shaping
#   - Q-value clipping
###############################################################################
