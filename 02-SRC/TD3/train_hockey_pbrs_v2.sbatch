#!/bin/bash
#SBATCH --job-name=td3_pbrs_v2
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=20:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# ============================================================================
# TD3 Hockey Training - PBRS V2 Configuration
# ============================================================================
#
# PBRS V2 DESIGN (Directional Attack Incentive):
# ----------------------------------------------
# Two-component potential function:
#   phi(s) = W_CHASE * phi_chase + W_ATTACK * phi_attack
#          = 0.5 * (-dist_agent_puck/MAX) + 1.0 * (-dist_puck_goal/MAX)
#
# REWARD MATRIX:
# | Action           | phi_chase | phi_attack | Net F | Result      |
# |------------------|-----------|------------|-------|-------------|
# | Approach puck    | +         | 0          | +     | Encouraged  |
# | Shoot forward    | -         | +          | +     | Encouraged! |
# | Shoot backward   | -         | -          | --    | Penalized!  |
# | Hold puck        | 0         | 0          | 0     | Neutral     |
#
# MATHEMATICAL GUARANTEE:
#   Forward shot:  dF = (D/MAX) * (W_ATTACK - W_CHASE) = +0.5D/MAX
#   Backward shot: dF = -(D/MAX) * (W_ATTACK + W_CHASE) = -1.5D/MAX
#   Margin: +2.0D/MAX (forward always better!)
#
# SCALING:
#   Max potential range: 150 (W_CHASE*1 + W_ATTACK*1 = 1.5, * 100 = 150)
#   Max episode PBRS: 150 * 0.02 = 3.0
#   Sparse reward (win): 10 * 0.1 = 1.0
#   Ratio: 3.0 / 10 = 0.3 < 1 (PBRS guides but doesn't dominate)
#
# ANNEALING TIMELINE (100k episodes):
#   Episode 0-5000:      weight = 1.0 (full guidance)
#   Episode 5000-20000:  weight = 1.0 -> 0.1 (gradual fade)
#   Episode 20000+:      weight = 0.1 (minimal guidance retained)
#
# ============================================================================

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 42 \
    \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.9998 \
    \
    --lr_actor 0.001 \
    --lr_critic 0.001 \
    \
    --batch_size 100 \
    --tau 0.005 \
    --gamma 0.99 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    \
    --train_freq -1 \
    --iter_fit 125 \
    --warmup_episodes 2000 \
    --buffer_size 1000000 \
    --pbrs_clip 2.0 \
    \
    --reward_scale 1.0 \
    --reward_shaping \
    --pbrs_scale 0.003 \
    \
    --pbrs_anneal_start 15000 \
    --pbrs_anneal_episodes 2000 \
    --pbrs_min_weight 0.1 \
    \
    --epsilon_reset_at_anneal \
    --epsilon_anneal_reset_value 0.4 \
    \
    --vf_reg_lambda 0.1 \
    \
    --q_clip 25.0 \
    --q_clip_mode soft \
    \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    \
    --self_play_start 100000 \
    --self_play_pool_size 25 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.5 \
    --use_pfsp \
    --pfsp_mode variance \
    --episode_block_size 50 \
    \
    --epsilon_reset_on_selfplay \
    --epsilon_reset_value 0.5 \
    \
    --log_interval 50 \
    --save_interval 500 \
    --eval_interval 100 \
    --eval_episodes 100 \
    --gif_episodes 3
