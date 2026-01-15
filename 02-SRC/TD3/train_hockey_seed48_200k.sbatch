#!/bin/bash
#SBATCH --job-name=Continue-Hockey-Weak-200k-seed48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/hockey_continue_%j.log
#SBATCH --error=/scratch/%u/logs/hockey_continue_%j.err

# Load modules (adjust to your cluster)
module load python/3.10 cuda/11.8

cd /scratch/1962994/02-SRC/TD3

python3 train_hockey.py \
  --mode NORMAL \
  --opponent weak \
  --max_episodes 200000 \
  --seed 48 \
  --warmup_episodes 0 \
  --checkpoint results_checkpoints_TD3_Hockey_NORMAL_strong_27500_seed48.pth \
  --eps 0.15 \
  --eps_min 0.1 \
  --eps_decay 0.99992 \
  --batch_size 1024 \
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
  --buffer_size 1000000 \
  --reward_scale 1.0 \
  --q_clip 25.0 \
  --q_clip_mode soft \
  --q_warning_threshold 10.0 \
  --hidden_actor 256 256 \
  --hidden_critic 256 256 128 \
  --reward_shaping \
  --pbrs_scale 0.5 \
  --pbrs_constant_weight \
  --no_strategic_rewards \
  --tie_penalty -1.5 \
  --no_lr_decay \
  --init_critic_bias_positive \
  --episode_block_size 50 \
  --eval_interval 500 \
  --eval_episodes 25 \
  --log_interval 50 \
  --save_interval 1000 \
  --gif_episodes 0
