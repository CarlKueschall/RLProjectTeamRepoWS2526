#!/bin/bash
#SBATCH --job-name=dreamerv3_small_weak
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
# DreamerV3 Hockey Training - Small Model Configuration
# ============================================================================
#
# ARCHITECTURE:
#   Model Size: small (~8M parameters)
#   - Deterministic state (RSSM): 512-dim
#   - Categorical latent: 32 categories × 32 classes = 1024-dim
#   - Hidden layers: 512-dim
#
# TRAINING STRATEGY:
#   - World Model: Learns to predict observations, rewards, episode continues
#   - Actor-Critic: Trained entirely in imagination (latent rollouts)
#   - Imagination Horizon: 15 steps for credit assignment
#   - Return Normalization: Percentile-based (5th-95th) for stable learning
#
# KEY HYPERPARAMETERS (from DreamerV3 paper):
#   Learning Rates: 4e-5 (all components, including AGC)
#   Entropy Scale: 3e-4 (prevents policy collapse)
#   Gradient Clipping: Adaptive (AGC 0.3 factor)
#   Categorical Unimix: 1% uniform mixing (prevents latent collapse)
#   Free Nats: 1.0 (KL loss threshold)
#
# EVALUATION & LOGGING:
#   Eval Frequency: Every 250 training episodes
#   Eval Episodes: 25 per evaluation run
#   GIF Recording: Every 100 episodes (3 episodes per GIF, side-by-side)
#   Checkpoints: Every 250 episodes
#
# EXPECTED PERFORMANCE:
#   vs Weak Opponent: ~70-80% win rate after 1M steps
#   vs Strong Opponent: ~30-40% win rate after 1M steps
#   Training Time: ~24 hours on RTX 1080 Ti
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif bash -c "
  conda activate py310
  python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --model_size small \
    --max_steps 1000000 \
    --seed 43 \
    --device auto \
    --batch_size 16 \
    --batch_length 50 \
    --imagination_horizon 15 \
    --imagine_batch_size 256 \
    --lr_world 4e-5 \
    --lr_actor 4e-5 \
    --lr_critic 4e-5 \
    --entropy_scale 3e-4 \
    --gamma 0.997 \
    --lambda_gae 0.95 \
    --use_agc \
    --agc_clip 0.3 \
    --free_nats 1.0 \
    --unimix 0.01 \
    --buffer_size 1000000 \
    --min_buffer_size 1000 \
    --eval_frequency 250 \
    --eval_episodes 25 \
    --gif_frequency 100 \
    --gif_episodes 3 \
    --save_frequency 250 \
    --self_play_start 0 \
    --wandb_project rl-hockey \
    --run_name 'DreamerV3-small-weak-seed43'
"

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/results ~/02-SRC/DreamerV3/results_seed43
