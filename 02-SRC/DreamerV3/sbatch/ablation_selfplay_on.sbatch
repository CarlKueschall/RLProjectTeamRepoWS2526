#!/bin/bash
#SBATCH --job-name=abl-sp-on
#SBATCH --cpus-per-task=8
#SBATCH --partition=week
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=48:00:00
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
# ABLATION: Self-Play ON (with PFSP, activating at episode 2000)
# ============================================================================
#
# This is our baseline configuration WITH self-play enabled.
# Compare against ablation_selfplay_off.sbatch to measure self-play impact.
#
# Self-Play Configuration:
#   - Activates at episode 2000 (after initial learning against fixed bots)
#   - Pool of 15 past checkpoints (FIFO rotation)
#   - PFSP variance mode: prioritizes opponents with ~50% win rate
#   - Weak ratio 0.3: 30% chance of facing anchor bots vs pool
#   - Save interval 500: new checkpoint added to pool every 500 episodes
#
# Expected: More diverse strategies, better generalization, possibly
# noisier training curve but stronger final agent.
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --seed 42 \
    --device cuda \
    \
    --gradient_steps 100000000000 \
    --replay_ratio 32 \
    --warmup_episodes 200 \
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
    --self_play_start 2000 \
    --self_play_pool_size 15 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.3 \
    --use_pfsp \
    --pfsp_mode variance \
    \
    --checkpoint_interval 10000 \
    --eval_interval 5000 \
    --eval_episodes 10 \
    --gif_interval 10000 \
    --gif_episodes 3 \
    --log_interval 100 \
    \
    --wandb_project rl-hockey \
    --run_name 'ABLATION-selfplay-ON'
