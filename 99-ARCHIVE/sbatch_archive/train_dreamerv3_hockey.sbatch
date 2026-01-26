#!/bin/bash
#SBATCH --job-name=dreamerv3-hockey
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
# DreamerV3 Hockey Training - Optimal Configuration (Simplified)
# ============================================================================
#
# ARCHITECTURE (based on NaturalDreamer):
#   - Recurrent state (GRU): 256-dim
#   - Categorical latent: 16 vars × 16 classes = 256-dim
#   - Encoded observation: 256-dim
#   - Hidden layers: 256-dim (all MLPs)
#   - Full state: 256 + 256 = 512-dim
#
# TRAINING STRATEGY:
#   - World Model: RSSM with categorical latents
#     - Reconstruction loss (Normal distribution)
#     - Reward prediction (Two-Hot Symlog with inverse frequency weighting)
#     - Continue prediction (Bernoulli)
#     - KL divergence (prior vs posterior with free nats)
#     - Auxiliary tasks (goal prediction, distance, shot quality)
#   - Actor-Critic: Trained entirely in imagination
#     - Lambda returns (TD-lambda) for value targets
#     - Percentile-based value normalization
#     - Entropy regularization for exploration
#
# KEY HYPERPARAMETERS:
#   Batch: 32 sequences × 32 timesteps
#   Imagination horizon: 15 steps
#   Learning rates: world=3e-4, actor=8e-5, critic=1e-4
#   Discount (gamma): 0.997
#   Lambda (TD-lambda): 0.95
#   Entropy scale: 0.003 (prevents policy collapse)
#   Free nats: 1.0 (KL threshold)
#   Replay ratio: 32 (gradient steps per env step)
#
# EXPECTED PERFORMANCE:
#   vs Weak: ~70-80% win rate after 500k gradient steps
#   Training time: ~12-18 hours on RTX 1080 Ti
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --mode NORMAL \
    --opponent weak \
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
    --run_name 'DreamerV3-optimal-weak-seed42'

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/results ~/02-SRC/DreamerV3/results_dreamerv3_seed42
