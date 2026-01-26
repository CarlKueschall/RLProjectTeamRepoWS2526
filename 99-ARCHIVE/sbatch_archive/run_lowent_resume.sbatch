#!/bin/bash
#SBATCH --job-name=dreamerv3-lowent
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3

# Copy checkpoint to scratch
CHECKPOINT_NAME="results_checkpoints_DreamerV3-small-weak-seed43_weak_seed43_20260120_122246_50k.pth"
cp ~/02-SRC/DreamerV3/${CHECKPOINT_NAME} /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# ============================================================================
# DreamerV3 Hockey Training - Low Entropy Continuation
# ============================================================================
#
# Resuming from 50k gradient steps checkpoint with reduced entropy to
# encourage exploitation over exploration.
#
# Changes from original run:
#   - entropy_scale: 0.003 -> 0.001 (reduce exploration pressure)
#   - Resuming from checkpoint (skip warmup, continue learning)
#
# Keeping optimal hyperparameters from ablation study:
#   - imagination_horizon: 15
#   - uniform_mix: 0.01
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --mode NORMAL \
    --opponent weak \
    --seed 43 \
    --device cuda \
    \
    --resume "${CHECKPOINT_NAME}" \
    \
    --gradient_steps 1000000000 \
    --replay_ratio 4 \
    --warmup_episodes 50 \
    --interaction_episodes 1 \
    \
    --batch_size 32 \
    --batch_length 32 \
    --imagination_horizon 15 \
    \
    --recurrent_size 256 \
    --latent_length 16 \
    --latent_classes 16 \
    --encoded_obs_size 256 \
    --uniform_mix 0.01 \
    \
    --lr_world 0.0003 \
    --lr_actor 0.0001 \
    --lr_critic 0.0001 \
    \
    --discount 0.997 \
    --lambda_ 0.99 \
    --entropy_scale 0.001 \
    --free_nats 1.0 \
    --gradient_clip 100 \
    \
    --buffer_capacity 500000 \
    \
    --checkpoint_interval 500 \
    --eval_interval 100 \
    --eval_episodes 10 \
    --gif_interval 100 \
    --gif_episodes 3 \
    --log_interval 10 \
    \
    --wandb_project rl-hockey \
    --run_name 'DreamerV3-lowent-resume-seed43'

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/results ~/02-SRC/DreamerV3/results_dreamerv3_lowent_seed43
