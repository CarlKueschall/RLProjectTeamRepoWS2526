#!/bin/bash
#SBATCH --job-name=abl-aux-on
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# ============================================================================
# ABLATION: Auxiliary Tasks ON (Baseline)
# ============================================================================
#
# This is our baseline configuration WITH auxiliary tasks enabled.
# Compare against ablation_auxiliary_off.sbatch to measure auxiliary task impact.
#
# Auxiliary Tasks:
#   1. Goal Prediction: Binary classification - will goal occur in next K steps?
#   2. Distance Prediction: Regression - puck-to-goal distance
#   3. Shot Quality: Regression - offensive opportunity quality
#
# These help the world model learn goal-relevant representations without
# corrupting the primary reward signal.
#
# Expected: Better world model quality, improved final performance
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --seed 42 \
    --device cuda \
    \
    --gradient_steps 150000 \
    --replay_ratio 32 \
    --warmup_episodes 100 \
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
    --lambda_ 0.95 \
    --entropy_scale 0.0003 \
    --free_nats 1.0 \
    --gradient_clip 100 \
    \
    --buffer_capacity 250000 \
    \
    --mixed_opponents \
    --mixed_weak_prob 0.5 \
    \
    --use_dreamsmooth \
    --dreamsmooth_alpha 0.5 \
    \
    --checkpoint_interval 10000 \
    --eval_interval 5000 \
    --eval_episodes 10 \
    --gif_interval 10000 \
    --gif_episodes 3 \
    --log_interval 100 \
    \
    --wandb_project rl-hockey \
    --run_name 'ABLATION-auxiliary-ON'
