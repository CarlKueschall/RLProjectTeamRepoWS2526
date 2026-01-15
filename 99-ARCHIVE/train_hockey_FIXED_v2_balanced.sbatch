#!/bin/bash
#SBATCH --job-name=TD3_FIXv2_balanced
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
# CRITICAL FIXES V2 - BALANCED CONFIGURATION
#########################################################
# Based on analysis of run: Opus-TD3-Hockey-NORMAL-weak-lr0.0003-seed42
#
# Core fixes (already implemented in code):
# 1. Observation normalization (fixes 50x position/velocity scale mismatch) ✓
# 2. PBRS SCALE increased from 0.05 to 1.0 in pbrs.py (20x improvement) ✓
# 3. PBRS velocity component added (rewards shooting, not just positioning) ✓
# 4. Gradient norm logging (monitors learning health) ✓
#
# V2 Parameter Adjustments (based on metrics analysis):
# 5. PBRS scale reduced from 2.0 → 0.5 (total: 1.0 × 0.5 = 0.5)
# 6. Reward scale increased from 0.1 → 1.0 (sparse: ±10 × 1.0 = ±10)
# 7. Ratio: PBRS ~2.9 × 0.5 = 1.45, Sparse ±10 → 14.5% (target range!)
# 8. Self-play gate lowered from 85% → 75% (more achievable)
# 9. Faster epsilon decay to exploit learned policy sooner
#
# Metrics from previous run showed:
# - Critic loss: 0.14 → 0.009 (✓ WORKING - down 94%)
# - Actor loss: -4.6 → -0.7 (✓ WORKING - improving)
# - PBRS contribution: 2.9 per episode (WAS TOO HIGH vs sparse 1.0)
# - Puck touches: 0.5 → 3.5 (✓ 7x improvement)
# - Win rate @ 2800 eps: 10.5% (improving but slower than expected)
#
# Expected outcomes with V2 fixes:
# - Sparse rewards DOMINATE (±10) over PBRS (~1.5)
# - Agent optimizes for GOALS not just PBRS positioning
# - Win rate should reach 70%+ vs weak bot by episode 5000
# - Q-values remain positive and stable (already fixed)
# - Gradient norms healthy (critic 1-10, actor 0.1-1.0)
#########################################################

cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 15000 \
    --seed 55 \
    --warmup_episodes 1000 \
    --eps 1.0 \
    --eps_min 0.1 \
    --eps_decay 0.9995 \
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
    --tie_penalty -3.0 \
    --no_lr_decay \
    --epsilon_reset_on_selfplay \
    --epsilon_reset_value 0.5 \
    --init_critic_bias_positive \
    --episode_block_size 50 \
    --eval_interval 250 \
    --eval_episodes 50 \
    --log_interval 25 \
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

