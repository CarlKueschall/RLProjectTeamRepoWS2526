#!/bin/bash
#SBATCH --job-name=hockey_td3
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=20:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch (faster access during training)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/

# Run training in container
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
  --mode NORMAL \
  --opponent weak \
  --max_episodes 50000 \
  --seed 48 \
  --eps 1.0 \
  --eps_min 0.05 \
  --eps_decay 0.99985 \
  --warmup_episodes 500 \
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
  --hidden_actor 400 400 \
  --hidden_critic 400 400 200 \
  --log_interval 100 \
  --save_interval 250 \
  --gif_interval 250 \
  --gif_episodes 3 \
  --eval_interval 250 \
  --self_play_start 4000 \
  --self_play_pool_size 12 \
  --self_play_save_interval 400 \
  --self_play_weak_ratio 0.4 \
  --use_dual_buffers \
  --use_pfsp \
  --pfsp_mode variance \
  --dynamic_anchor_mixing \
  --performance_gated_selfplay \
  --selfplay_gate_winrate 0.8 \
  --selfplay_gate_variance 0.3 \
  --regression_rollback \
  --regression_threshold 0.1 
