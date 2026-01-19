#!/bin/bash
#SBATCH --job-name=hockey_td3_noselfplay_nokeep
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
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

# Training configuration without self-play and with keep_mode disabled:
# - Trains against strong opponent only (no self-play)
# - keep_mode disabled: puck bounces immediately on contact
singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
      --mode NORMAL \
      --opponent strong \
      --max_episodes 100000 \
      --seed 48 \
      \
      # ======== EXPLORATION & WARMUP (research-based for 100k) ========
      --warmup_episodes 2000 \
      --eps 1.0 \
      --eps_min 0.05 \
      --eps_decay 0.999957 \
      \
      # ======== CORE TD3 HYPERPARAMETERS ========
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
      --buffer_size 1000000 \
      --q_clip 25.0 \
      --q_clip_mode hard \
      --q_warning_threshold 10.0 \
      \
      # ======== NETWORK ARCHITECTURE (your proven setup) ========
      --hidden_actor 1024 1024 \
      --hidden_critic 1024 1024 200 \
      \
      # ======== NEW: RESEARCH-BASED IMPROVEMENTS FOR 100K TRAINING ========
      --reward_shaping \
      --tie_penalty -3.0 \
      --lr_decay \
      --lr_min_factor 0.1 \
      --episode_block_size 20 \
      \
      # ======== EVALUATION & LOGGING ========
      --eval_interval 500 \
      --eval_episodes 50 \
      --log_interval 20 \
      --save_interval 500 \
      --gif_episodes 3 \
      \
      # ======== SELF-PLAY: TOURNAMENT ROBUSTNESS ========
      --self_play_start 5000 \
      --self_play_pool_size 25 \
      --self_play_save_interval 500 \
      --self_play_weak_ratio 0.5 \
      --use_dual_buffers \
      --use_pfsp \
      --pfsp_mode variance \
      --dynamic_anchor_mixing \
      --performance_gated_selfplay \
      --selfplay_gate_winrate 0.90 \
      --regression_rollback \
      --regression_threshold 0.15

