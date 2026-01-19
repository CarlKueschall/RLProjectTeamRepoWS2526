#!/bin/bash
#SBATCH --job-name=dreamerv3-strong
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
# DreamerV3 Hockey Training - Strong Opponent
# ============================================================================
#
# Same architecture as weak training, but against strong opponent.
# Strong opponent is much more aggressive - may need longer training.
#
# EXPECTED PERFORMANCE:
#   vs Strong: ~30-50% win rate after 1M gradient steps
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --mode NORMAL \
    --opponent strong \
    --seed 42 \
    --device cuda \
    \
    --gradient_steps 1000000 \
    --replay_ratio 32 \
    --warmup_episodes 10 \
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
    \
    --lr_world 0.0003 \
    --lr_actor 0.00008 \
    --lr_critic 0.0001 \
    \
    --discount 0.997 \
    --lambda_ 0.95 \
    --entropy_scale 0.003 \
    --free_nats 1.0 \
    --gradient_clip 100 \
    \
    --use_pbrs \
    --pbrs_scale 0.03 \
    --pbrs_w_chase 0.5 \
    --pbrs_w_attack 1.0 \
    \
    --buffer_capacity 100000 \
    \
    --checkpoint_interval 5000 \
    --eval_interval 1000 \
    --eval_episodes 10 \
    --gif_interval 10000 \
    --gif_episodes 3 \
    --log_interval 10 \
    \
    --wandb_project rl-hockey \
    --run_name 'DreamerV3-optimal-strong-seed42'

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/results ~/02-SRC/DreamerV3/results_dreamerv3_strong_seed42
