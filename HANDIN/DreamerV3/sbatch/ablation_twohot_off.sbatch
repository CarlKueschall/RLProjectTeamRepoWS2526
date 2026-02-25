#!/bin/bash
#SBATCH --job-name=abl-2hot-off
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
# ABLATION: Two-Hot Symlog OFF (Gaussian - NaturalDreamer Baseline)
# ============================================================================
#
# This ablation runs WITH Gaussian reward/value prediction (NaturalDreamer).
# Compare against ablation_twohot_on.sbatch (Two-Hot) to measure impact.
#
# Gaussian (NaturalDreamer baseline):
#   - Outputs Normal(mean, std) distribution for rewards and values
#   - Uses negative log likelihood loss
#   - Assumes unimodal distribution (FAILS for hockey's sparse rewards)
#
# LIMITATION for Hockey:
#   - Hockey rewards are {-10, 0, +10} - highly multi-modal
#   - Gaussian mean gets pulled toward 0 (most common value)
#   - Cannot represent bimodal +10/-10 for goal events
#   - Weak gradients for rare sparse events
#
# Expected: Poor reward prediction, slower/worse learning, lower final win rate
# This demonstrates why Two-Hot Symlog is essential for sparse rewards.
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
    --use_gaussian_heads \
    \
    --checkpoint_interval 10000 \
    --eval_interval 5000 \
    --eval_episodes 10 \
    --gif_interval 10000 \
    --gif_episodes 3 \
    --log_interval 100 \
    \
    --wandb_project rl-hockey \
    --run_name 'ABLATION-twohot-OFF-gaussian'
