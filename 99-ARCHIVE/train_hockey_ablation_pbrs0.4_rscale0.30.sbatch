#!/bin/bash
#SBATCH --job-name=abl_p0.4_r0.30
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=12:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

#########################################################
# ABLATION: PBRS_SCALE=0.02 (0.4×0.05), REWARD_SCALE=0.30
#########################################################
# Grid position: (3/12)
# PBRS effective: 0.02 (LOW PBRS)
# Reward scale: 0.30 (SOFT punishment)
# Expected: Soft punishment may help, but low PBRS still weak
#########################################################

cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --opponent_progression \
    --weak_until 5000 \
    --strong_until 10000 \
    --max_episodes 10000 \
    --seed 102 \
    --warmup_episodes 1000 \
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
    --reward_scale 0.30 \
    --q_clip 25.0 \
    --q_clip_mode soft \
    --q_warning_threshold 10.0 \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    --reward_shaping \
    --pbrs_scale 0.4 \
    --pbrs_constant_weight \
    --no_strategic_rewards \
    --tie_penalty -3.0 \
    --no_lr_decay \
    --epsilon_reset_on_selfplay \
    --epsilon_reset_value 0.5 \
    --init_critic_bias_positive \
    --episode_block_size 50 \
    --eval_interval 250 \
    --eval_episodes 50 \
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

