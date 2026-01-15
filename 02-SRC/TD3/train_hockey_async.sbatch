#!/bin/bash
#SBATCH --job-name=td3-async-hockey
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =============================================================================
# Async TD3 Hockey Training on Cluster
# =============================================================================
# This script runs fully asynchronous training with:
# - 4 parallel collector workers (CPU processes)
# - 1 trainer thread (GPU)
# - Continuous data collection and training
#
# Expected speedup: 2-3x over sequential training
# =============================================================================

echo "====================================="
echo "ASYNC TD3 HOCKEY TRAINING"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "====================================="

# Navigate to the TD3 directory
cd ~/workspace/RLProjectHockey/02-SRC/TD3

# Run async training with optimized settings
# Key differences from sequential:
# - num_workers: Number of parallel collector processes
# - weight_sync_interval: How often to sync weights to collectors
# - Larger warmup to ensure buffer has diverse experiences
singularity exec --nv ~/containers/hockey.sif python3 train_hockey_async.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --num_workers 4 \
    --weight_sync_interval 50 \
    --seed 55 \
    --warmup_episodes 500 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99995 \
    --use_per \
    --per_alpha 0.6 \
    --per_beta_start 0.4 \
    --per_beta_frames 100000 \
    --hidden_actor 400 300 \
    --hidden_critic 400 300 128 \
    --batch_size 1024 \
    --buffer_size 1000000 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --grad_clip 1.0 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --q_clip 25.0 \
    --q_clip_mode soft \
    --reward_scale 1.0 \
    --reward_shaping \
    --pbrs_scale 0.5 \
    --pbrs_constant_weight \
    --eval_interval 2000 \
    --eval_episodes 100 \
    --log_interval 100 \
    --save_interval 5000

echo "====================================="
echo "Training complete"
echo "End time: $(date)"
echo "====================================="
