#!/bin/bash
#SBATCH --job-name=td3_100k_parallel
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

###############################################################################
# TD3 HOCKEY TRAINING - 100K EPISODES WITH PARALLEL DATA COLLECTION
#
# KEY PARAMETERS (based on comprehensive hyperparameter analysis):
#   - Network: [400, 300] actor, [400, 300, 128] critic (analysis-confirmed)
#   - Learning rate: 0.0003 (higher LR=0.0005 caused 40% slowdown + instability)
#   - Gamma: 0.99 (critical for sparse reward credit assignment)
#   - Q-clip: 25.0 (for gamma=0.99 with sparse ±10 rewards)
#   - PBRS scale: 0.5 (guides exploration without dominating)
#   - Parallel envs: 4 (matches cpus-per-task for 2-3x speedup)
#
# EXPECTED RUNTIME: ~12-15 hours with parallel collection (vs ~30+ hours serial)
# EXPECTED OUTCOME: 60-80% win rate vs weak opponent
###############################################################################

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# Run training with optimized hyperparameters
singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 55 \
    \
    `# === PARALLEL DATA COLLECTION ===` \
    `# 4 workers matches cpus-per-task, provides 2-3x speedup` \
    --parallel_envs 4 \
    \
    `# === EXPLORATION ===` \
    `# eps=1.0 ensures diverse buffer from start` \
    `# Slow decay: 1.0 * 0.99995^100000 ≈ 0.007 (effectively greedy by end)` \
    --warmup_episodes 2000 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99995 \
    \
    `# === PRIORITIZED EXPERIENCE REPLAY ===` \
    `# Oversamples high-TD-error transitions (rare wins/losses)` \
    --use_per \
    --per_alpha 0.6 \
    --per_beta_start 0.4 \
    --per_beta_frames 100000 \
    \
    `# === NETWORK ARCHITECTURE (analysis-confirmed optimal) ===` \
    --hidden_actor 400 300 \
    --hidden_critic 400 300 128 \
    \
    `# === TRAINING DYNAMICS ===` \
    `# gamma=0.99: critical for sparse reward credit assignment` \
    `# With gamma=0.99: reward 100 steps away = 3.66 (useful signal)` \
    `# With gamma=0.95: reward 100 steps away = 0.059 (no signal)` \
    --batch_size 1024 \
    --buffer_size 1000000 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --grad_clip 1.0 \
    \
    `# === TD3 SPECIFIC ===` \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    \
    `# === Q-VALUE STABILITY ===` \
    `# q_clip=25.0: realistic Q-range for gamma=0.99 with sparse ±10 rewards` \
    `# Analysis runs used q_clip=25.0 successfully` \
    --q_clip 25.0 \
    --q_clip_mode soft \
    \
    `# === REWARD SHAPING (PBRS only, no strategic rewards) ===` \
    `# PBRS at 0.5x guides exploration without creating local optimum` \
    `# reward_scale=1.0 keeps sparse rewards dominant` \
    --reward_scale 1.0 \
    --reward_shaping \
    --pbrs_scale 0.5 \
    --pbrs_constant_weight \
    \
    `# === TIE PENALTY ===` \
    `# Analysis: -1.5 vs 0.0 had negligible difference (±1.89%)` \
    `# Using 0.0 (disabled) for simplicity` \
    --no_tie_penalty \
    \
    `# === SELF-PLAY DISABLED ===` \
    `# Focus on mastering weak opponent first` \
    `# Enable self-play in stage 2 after reaching 70%+ win rate` \
    --disable_selfplay \
    --no_lr_decay \
    \
    `# === LOGGING ===` \
    --eval_interval 1000 \
    --eval_episodes 100 \
    --log_interval 50 \
    --save_interval 2000 \
    --gif_episodes 3

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/TD3/results ~/02-SRC/TD3/

###############################################################################
# AFTER TRAINING COMPLETES:
#
# Check final metrics:
#   - Win rate should be 60-80% vs weak opponent
#   - Q_avg should be positive and stable (0.5 to 5.0 range)
#   - Behavior: agent should actively engage with puck
#
# If win rate > 70%, consider Stage 2 with self-play:
#   --self_play_start 100001
#   --self_play_pool_size 25
#   --self_play_weak_ratio 0.3
#   --checkpoint results/checkpoints/best_model.pth
###############################################################################
