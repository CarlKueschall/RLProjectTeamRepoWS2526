#!/bin/bash
#SBATCH --job-name=dreamer-no-aux
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

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# ============================================================================
# Experiment C: DreamSmooth WITHOUT Auxiliary Task Heads
# ============================================================================
#
# Tests whether auxiliary heads help representation learning.
# Compare against exp_dreamsmooth.sbatch to measure aux head impact.
#
# This experiment has:
#   1. Zero initialization for RewardModel
#   2. EMA (slow) critic for stable bootstrap
#   3. DreamSmooth reward smoothing (alpha=0.5)
#   4. NO auxiliary task heads (--no_aux_tasks)
#
# If aux heads help, this should perform worse than exp_dreamsmooth.
# If aux heads don't matter, performance should be similar.
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --mode NORMAL \
    --opponent weak \
    --seed 42 \
    --device cuda \
    \
    --gradient_steps 1000000000 \
    --replay_ratio 16 \
    --warmup_episodes 500 \
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
    --lr_actor 0.0003 \
    --lr_critic 0.0001 \
    \
    --discount 0.997 \
    --lambda_ 0.99 \
    --entropy_scale 0.0003 \
    --free_nats 1.0 \
    --gradient_clip 100 \
    \
    --buffer_capacity 500000 \
    \
    --use_dreamsmooth \
    --dreamsmooth_alpha 0.5 \
    --no_aux_tasks \
    \
    --checkpoint_interval 500 \
    --eval_interval 100 \
    --eval_episodes 10 \
    --gif_interval 100 \
    --gif_episodes 3 \
    --log_interval 10 \
    \
    --wandb_project rl-hockey \
    --run_name 'DreamerV3-no-aux-seed42'
