#!/bin/bash
#SBATCH --job-name=td3_stage2_gamma95
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=23:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

###############################################################################
# STAGE 2: TACTICAL REFINEMENT WITH SHORTER CREDIT ASSIGNMENT
#
# PREREQUISITES (from Stage 1):
#   - Win rate > 55% vs weak opponent
#   - Q_avg is POSITIVE (0.2 to 1.0)
#   - Agent shows active play (puck_touches > 3, time_near_puck > 4)
#
# IF STAGE 1 DID NOT MEET THESE CRITERIA:
#   DO NOT run Stage 2. Instead, debug Stage 1 or extend it.
#
# WHAT STAGE 2 DOES:
#   1. gamma=0.95 provides SHORTER credit assignment
#      - Agent focuses on immediate tactical decisions
#      - Less "long-term planning", more "react to current situation"
#   2. eps=0.15 continues moderate exploration
#      - Agent is now competent, doesn't need full random exploration
#      - Still maintains some exploration for adaptation
#   3. Continues from Stage 1 checkpoint
#      - Preserves learned cause-effect relationships
#      - Refines policy for higher win rate
#
# EXPECTED PROGRESSION:
#   Episodes 30k-50k:   Win rate 60-70%
#   Episodes 50k-100k:  Win rate 70-85%
#   Episodes 100k-200k: Win rate 85-95%
#
# OPTIONAL: OPPONENT PROGRESSION
#   After achieving 80%+ vs weak, consider switching to:
#   --opponent strong (or enable self-play)
#
# ESTIMATED RUNTIME: 20-22 hours for 170k episodes
###############################################################################

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

###############################################################################
# IMPORTANT: Update the checkpoint path below to match your Stage 1 output!
# Expected format: results/checkpoints/TD3_Hockey_NORMAL_weak_30000_seed55.pth
# Or use the best_model.pth if Stage 1 saved one with good performance.
###############################################################################
STAGE1_CHECKPOINT="results/checkpoints/TD3_Hockey_NORMAL_weak_30000_seed55.pth"

# Verify checkpoint exists
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Please update the STAGE1_CHECKPOINT variable with the correct path."
    exit 1
fi

# Run Stage 2 training with tactical gamma
singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 200000 \
    --seed 55 \
    \
    `# === CONTINUE FROM STAGE 1 CHECKPOINT ===` \
    --checkpoint "$STAGE1_CHECKPOINT" \
    \
    `# === EXPLORATION (MODERATE, CONTINUING FROM STAGE 1) ===` \
    `# No warmup needed - buffer preserved from Stage 1` \
    `# eps=0.15 matches where Stage 1 ended` \
    `# Slow decay to eps_min=0.05 over remaining training` \
    --warmup_episodes 0 \
    --eps 0.15 \
    --eps_min 0.05 \
    --eps_decay 0.9999 \
    \
    `# === PRIORITIZED EXPERIENCE REPLAY ===` \
    `# Continue using PER for experience diversity` \
    --use_per \
    --per_alpha 0.6 \
    --per_beta_start 0.4 \
    --per_beta_frames 100000 \
    \
    `# === NETWORK ARCHITECTURE (same as Stage 1) ===` \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    \
    `# === TRAINING DYNAMICS ===` \
    `# gamma=0.95 for TACTICAL refinement` \
    `# Shorter credit assignment = focus on immediate decisions` \
    `# Agent already learned cause-effect in Stage 1` \
    --batch_size 1024 \
    --buffer_size 1000000 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.95 \
    --tau 0.005 \
    --grad_clip 1.0 \
    \
    `# === TD3 SPECIFIC ===` \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    \
    `# === Q-VALUE STABILITY ===` \
    `# q_clip=10.0: Same as Stage 1, realistic for sparse ±10 rewards` \
    `# Soft tanh clipping: gamma=0.95 produces slightly smaller Q-values, still in [-10, +10]` \
    --q_clip 10.0 \
    --q_clip_mode soft \
    \
    `# === REWARD SHAPING ===` \
    `# Can reduce PBRS in Stage 2 since agent is already competent` \
    `# pbrs_scale=0.2 provides light guidance without interference` \
    --reward_scale 1.0 \
    --reward_shaping \
    --pbrs_scale 0.2 \
    --pbrs_constant_weight \
    --no_strategic_rewards \
    --tie_penalty -1.5 \
    \
    `# === SELF-PLAY DISABLED (can enable later) ===` \
    `# Once 80%+ vs weak achieved, consider enabling self-play` \
    `# Or switch to --opponent strong first` \
    --disable_selfplay \
    --no_lr_decay \
    \
    `# === LOGGING ===` \
    --eval_interval 1000 \
    --eval_episodes 100 \
    --log_interval 50 \
    --save_interval 5000 \
    --gif_episodes 3

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/TD3/results ~/02-SRC/TD3/

###############################################################################
# AFTER STAGE 2: OPTIONAL STAGE 3 (STRONG OPPONENT / SELF-PLAY)
#
# If win rate > 85% vs weak, you can progress to stronger training:
#
# Option A: Strong opponent
#   --opponent strong
#   --checkpoint results/checkpoints/best_model.pth
#
# Option B: Self-play (recommended for competition-level performance)
#   --opponent self
#   --checkpoint results/checkpoints/best_model.pth
#   --use_pfsp
#   --performance_gated_selfplay
#
# The successful 94.5% WR run used opponent progression:
#   weak → strong → self-play
###############################################################################
