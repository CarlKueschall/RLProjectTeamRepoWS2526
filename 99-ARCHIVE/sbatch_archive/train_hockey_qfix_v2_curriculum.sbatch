#!/bin/bash
#SBATCH --job-name=hockey_qfix_v2_curriculum
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

#########################################################
# Q-VALUE FIX V2 - 50k Episodes with Curriculum Learning
#########################################################
# CRITICAL CHANGES FROM V1:
# 1. START WITH WEAK OPPONENT (not strong!) - agent must learn basics first
# 2. Use curriculum: weak (0-10k) → strong (10k-20k) → self-play (20k+)
# 3. PBRS magnitude increased 10x (0.05 vs 0.005) - now provides real guidance
# 4. Critic bias increased to +5.0 (was +1.0) - stronger optimism
# 5. Reward scale increased to 0.2 (was 0.1) - softer punishment for losses
# 6. Self-play gate based on weak opponent performance (75% win rate)
#
# Why V1 Failed:
# - Training from scratch against STRONG opponent → agent never learned basics
# - Q-values collapsed to negative (-0.10) and stayed there
# - Actor loss stuck at +0.10 (positive) = no learning signal
# - PBRS was 200x too small (0.005 vs -1.0 losses) = useless
# - Only achieved 22% win rate vs weak after 24k episodes = total failure
#
# V2 Strategy:
# - Learn basics vs weak (touch puck, score occasionally) first
# - Graduate to strong opponent once competent vs weak
# - Then activate self-play for advanced learning
# - 50k episodes = ~12-16 hours on A4000
#########################################################

cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --opponent_progression \
    --weak_until 10000 \
    --strong_until 20000 \
    --max_episodes 50000 \
    --seed 56 \
    --warmup_episodes 2000 \
    --eps 1.0 \
    --eps_min 0.1 \
    --eps_decay 0.9997 \
    --batch_size 1024 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.95 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 1000000 \
    --reward_scale 0.2 \
    --q_clip 25.0 \
    --q_clip_mode soft \
    --q_warning_threshold 10.0 \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    --reward_shaping \
    --pbrs_scale 1.0 \
    --pbrs_constant_weight \
    --no_strategic_rewards \
    --tie_penalty -3.0 \
    --no_lr_decay \
    --epsilon_reset_on_selfplay \
    --epsilon_reset_value 0.5 \
    --init_critic_bias_positive \
    --episode_block_size 50 \
    --eval_interval 1000 \
    --eval_episodes 100 \
    --log_interval 10 \
    --save_interval 500 \
    --gif_episodes 3 \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.75 \
    --self_play_pool_size 25 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.5 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode hard

