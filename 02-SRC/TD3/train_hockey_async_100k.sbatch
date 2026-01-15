#!/bin/bash
#SBATCH --job-name=td3_async_100k
#SBATCH --cpus-per-task=6
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

###############################################################################
# TD3 ASYNC HOCKEY TRAINING - 100K EPISODES
#
# ASYNC ARCHITECTURE BENEFITS:
#   - Collectors run continuously in parallel (never block on training)
#   - Trainer runs continuously (GPU never idle)
#   - On Linux: 50K queue size (vs 5K on macOS) = no transition drops
#   - Expected speedup: 2-3x over sequential training
#
# RESOURCE USAGE:
#   - 4 workers = 4 CPU cores for collection
#   - 1 CPU core for buffer filler thread
#   - 1 CPU core for main orchestrator
#   - 1 GPU for continuous training
#
# EXPECTED RUNTIME: ~6-10 hours (vs ~15-20 hours sequential)
###############################################################################

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# Run ASYNC training
singularity exec --nv ~/containers/hockey.sif python3 train_hockey_async.py \
    --mode NORMAL \
    --opponent strong \
    --max_episodes 100000 \
    --seed 55 \
    \
    `# === ASYNC-SPECIFIC SETTINGS ===` \
    `# num_workers: 4 collectors running in parallel` \
    `# weight_sync_interval: sync policy to collectors every 50 training steps` \
    `# warmup_episodes: collect before training starts (faster with 4 workers)` \
    --num_workers 4 \
    --weight_sync_interval 50 \
    --warmup_episodes 500 \
    \
    `# === EXPLORATION ===` \
    `# Same as sequential - eps decays as training progresses` \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99995 \
    \
    `# === PRIORITIZED EXPERIENCE REPLAY ===` \
    `# Async buffer supports PER with thread-safe sampling` \
    --use_per \
    --per_alpha 0.6 \
    --per_beta_start 0.4 \
    --per_beta_frames 100000 \
    \
    `# === NETWORK ARCHITECTURE ===` \
    `# Same as your sequential config` \
    --hidden_actor 1024 1024 \
    --hidden_critic 1024 1024 256 \
    \
    `# === TRAINING DYNAMICS ===` \
    `# Note: No --train_freq in async (trainer runs continuously)` \
    `# batch_size: samples per training step` \
    --batch_size 1024 \
    --buffer_size 1000000 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
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
    --q_clip 25.0 \
    --q_clip_mode soft \
    \
    `# === REWARD SHAPING (PBRS) ===` \
    --reward_scale 1.0 \
    --reward_shaping \
    --pbrs_scale 0.5 \
    --pbrs_constant_weight \
    \
    `# === LOGGING & CHECKPOINTS ===` \
    `# eval_interval: evaluate every N episodes` \
    `# Increased intervals since async is faster` \
    --eval_interval 2000 \
    --eval_episodes 50 \
    --log_interval 100 \
    --save_interval 5000

# Copy results back to home directory
mkdir -p ~/02-SRC/TD3/results
cp -R /scratch/$SLURM_JOB_ID/02-SRC/TD3/results/* ~/02-SRC/TD3/results/

echo "============================================="
echo "ASYNC TRAINING COMPLETE"
echo "Results copied to ~/02-SRC/TD3/results/"
echo "============================================="
