#!/bin/bash
#SBATCH --job-name=hockey_td3_optimized_25k
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=12:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent strong \
    --max_episodes 25000 \
    --seed 52 \
    --warmup_episodes 250 \
    --eps 1.0 \
    --eps_min 0.1 \
    --eps_decay 0.9997 \
    --batch_size 1024 \
    --train_freq 10 \
    --lr_actor 0.0008 \
    --lr_critic 0.0008 \
    --gamma 0.99 \
    --tau 0.015 \
    --policy_freq 2 \
    --target_update_freq 1 \
    --target_noise_std 0.25 \
    --target_noise_clip 0.75 \
    --grad_clip 1.0 \
    --buffer_size 500000 \
    --q_clip 20.0 \
    --q_clip_mode hard \
    --q_warning_threshold 10.0 \
    --hidden_actor 1024 1024 \
    --hidden_critic 1024 1024 200 \
    --reward_shaping \
    --tie_penalty -3.0 \
    --lr_decay \
    --lr_min_factor 0.05 \
    --episode_block_size 25 \
    --eval_interval 250 \
    --eval_episodes 50 \
    --log_interval 20 \
    --save_interval 250 \
    --gif_episodes 3 \
    --self_play_start 5000 \
    --self_play_pool_size 25 \
    --self_play_save_interval 250 \
    --self_play_weak_ratio 0.5 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode variance \
    --dynamic_anchor_mixing \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.90 \
    --regression_rollback \
    --regression_threshold 0.15
