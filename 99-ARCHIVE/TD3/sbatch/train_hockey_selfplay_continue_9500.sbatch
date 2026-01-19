#!/bin/bash
#SBATCH --job-name=hockey_td3_selfplay_lowlr
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A4000:1
# Note: Changed from 1080ti to A4000 because 1080ti nodes are all busy
# Alternative: --gres=gpu:2080ti:1 (nodes 37-40 also have idle 2080ti GPUs)
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch (faster access during training)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/

# Run training in container
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key (replace with your actual API key)
# You can find your API key at: https://wandb.ai/authorize
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# Self-play training configuration with LOW learning rate:
# - Trains against strong opponent until self-play activates (episode 25k+)
# - During self-play: 40% anchor episodes (balanced weak/strong) + 60% self-play episodes
# - Anchor buffer maintains balanced weak/strong experiences to prevent forgetting
# - Pool buffer contains evolving self-play experiences
# - Learning rate: 0.0001 (lower than baseline 0.0003)
# - keep_mode=ON (default, matches tournament server)
singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
  --mode NORMAL \
  --opponent strong \
  --max_episodes 100000 \
  --seed 48 \
  --checkpoint /scratch/$SLURM_JOB_ID/02-SRC/TD3/results_checkpoints_TD3_Hockey_NORMAL_strong_9500_seed48.pth \
  --eps 1.0 \
  --eps_min 0.05 \
  --eps_decay 0.999957 \
  --warmup_episodes 0 \
  --batch_size 512 \
  --train_freq 10 \
  --lr_actor 0.0003 \
  --lr_critic 0.0003 \
  --gamma 0.99 \
  --tau 0.005 \
  --policy_freq 2 \
  --target_update_freq 2 \
  --target_noise_std 0.2 \
  --target_noise_clip 0.5 \
  --grad_clip 1.0 \
  --buffer_size 500000 \
  --reward_shaping \
  --q_clip 25.0 \
  --q_clip_mode hard \
  --q_warning_threshold 10.0 \
  --hidden_actor 1024 1024 \
  --hidden_critic 1024 1024 200 \
  --log_interval 100 \
  --save_interval 250 \
  --gif_episodes 3 \
  --eval_interval 250 \
  --self_play_start 0 \
  --self_play_pool_size 12 \
  --self_play_save_interval 400 \
  --self_play_weak_ratio 0.4 \
  --use_dual_buffers \
  --use_pfsp \
  --pfsp_mode variance \
  --dynamic_anchor_mixing \
  --performance_gated_selfplay \
  --selfplay_gate_winrate 0.85 \
  --regression_rollback \
  --regression_threshold 0.1
