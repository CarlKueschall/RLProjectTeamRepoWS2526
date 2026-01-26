#!/bin/bash
#SBATCH --job-name=dreamer-mixed-selfplay
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
# Optimal DreamerV3 Training: Mixed Opponents → Self-Play with PFSP
# ============================================================================
#
# Training Strategy (3 phases):
#   Phase 1 (episodes 0-2999): Mixed weak/strong opponents (50/50)
#     - Builds general skills against both playstyles
#     - Prevents overfitting to a single opponent type
#     - DreamSmooth provides denser reward signal for sparse goals
#     - Runs until mixed-opponent performance plateaus
#
#   Phase 2 (episodes 3000+): Self-play with PFSP (variance mode)
#     - Pool of past checkpoints for diverse training opponents
#     - PFSP selects opponents with ~50% win rate (maximum learning signal)
#     - 30% anchor (weak/strong), 70% self-play pool
#     - Pool saves every 300 episodes for smooth difficulty progression
#
#   Phase 3 (continuous): Ongoing self-play refinement
#     - Pool rotates (size 15) keeping recent + diverse opponents
#     - Agent continually improves against its own history
#
# Key improvements over previous runs:
#   1. Mixed opponents prevent overfitting (was 75% vs strong, 40% vs weak)
#   2. Fixed goal prediction threshold (1.0 vs broken 5.0)
#   3. Increased goal prediction horizon (25 steps ≈ 2-3s of gameplay)
#   4. Proper entropy scale (0.0003, paper default - no inflation)
#   5. Self-play with PFSP for continued improvement beyond basic opponents
#   6. DreamSmooth for sparse reward handling
#   7. Slow critic (EMA decay 0.98) for stable bootstrapping
#
# Success criteria:
#   - eval/win_rate_weak: > 70%
#   - eval/win_rate_strong: > 60%
#   - behavior/entropy_mean: stays positive (0.5 - 3.0 range)
#   - world/aux_goal_positive_rate: > 0 (goal prediction working)
#   - diagnostics/return_range_S: growing, not stuck at 1.0
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
    --mean_reg_scale 0.01 \
    --free_nats 1.0 \
    --gradient_clip 100 \
    \
    --buffer_capacity 250000 \
    \
    --use_dreamsmooth \
    --dreamsmooth_alpha 0.5 \
    \
    --mixed_opponents \
    --mixed_weak_prob 0.5 \
    \
    --self_play_start 3000 \
    --self_play_pool_size 15 \
    --self_play_save_interval 300 \
    --self_play_weak_ratio 0.3 \
    --use_pfsp \
    --pfsp_mode variance \
    \
    --checkpoint_interval 500 \
    --eval_interval 100 \
    --eval_episodes 10 \
    --gif_interval 10000 \
    --gif_episodes 3 \
    --log_interval 10 \
    \
    --wandb_project rl-hockey \
    --run_name 'DreamerV3-mixed-selfplay-seed42'
