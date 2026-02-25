#!/bin/bash
#SBATCH --job-name=summit-dreamerv3-24h
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=24:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

set -euo pipefail

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3

# ============================================================================
# Summit Final Push (24h) - Phase A Stabilization
# ============================================================================
#
# Resume from a robust checkpoint, run anchor-heavy self-play continuation with:
# - Lower update aggressiveness (replay_ratio=16, lower actor LR)
# - Frequent checkpoints/evals
# - Probe checkpoint robustness metrics:
#     eval/probe_min_win_rate
#     eval/probe_mean_win_rate
#     eval/robust_score
# - Rolling opponent selection diagnostics:
#     stats/opponent_recent_{weak,strong,selfplay}_frac
#
# Resume checkpoint is assumed to be in this directory (same dir as train_hockey.py).
#
# ============================================================================

# Resume checkpoint (must exist in this directory at runtime)
CHECKPOINT_NAME="357k_0130_193448.pth"

if [ ! -f "${CHECKPOINT_NAME}" ]; then
  echo "ERROR: resume checkpoint not found in $(pwd): ${CHECKPOINT_NAME}"
  exit 1
fi

# Optional: W&B API key (set if needed in your environment)
export WANDB_API_KEY="${WANDB_API_KEY:-}"

RUN_NAME="summit-final24h-phaseA-seed42-${SLURM_JOB_ID}"

singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --config hockey.yml \
    --mode NORMAL \
    --seed 42 \
    --device cuda \
    --run_name "${RUN_NAME}" \
    \
    --resume "${CHECKPOINT_NAME}" \
    \
    --gradient_steps 1000000000 \
    --replay_ratio 16 \
    --warmup_episodes 50 \
    --interaction_episodes 1 \
    \
    --batch_size 32 \
    --batch_length 32 \
    --imagination_horizon 15 \
    \
    --recurrent_size 256 \
    --latent_length 16 \
    --latent_classes 16 \
    --encoded_obs_size 256 \
    --uniform_mix 0.01 \
    \
    --lr_world 0.0002 \
    --lr_actor 0.00005 \
    --lr_critic 0.00007 \
    \
    --discount 0.997 \
    --lambda_ 0.95 \
    --entropy_scale 0.0003 \
    --free_nats 1.0 \
    --gradient_clip 100 \
    \
    --use_dreamsmooth \
    --dreamsmooth_alpha 0.5 \
    --buffer_capacity 250000 \
    \
    --mixed_opponents \
    --mixed_weak_prob 0.5 \
    \
    --self_play_start 1 \
    --self_play_pool_size 30 \
    --self_play_save_interval 250 \
    --self_play_weak_ratio 0.65 \
    --use_pfsp \
    --pfsp_mode variance \
    \
    --probe_checkpoints_file probe_checkpoints_summit_top4.txt \
    --probe_episodes 8 \
    --opponent_window_size 200 \
    \
    --checkpoint_interval 1000 \
    --eval_interval 1000 \
    --eval_episodes 12 \
    --gif_interval 0 \
    --log_interval 5 \
    \
    --wandb_project rl-hockey

# Copy outputs back to home
mkdir -p ~/02-SRC/DreamerV3/results_summit
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/results ~/02-SRC/DreamerV3/results_summit/
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/checkpoints ~/02-SRC/DreamerV3/results_summit/checkpoints_${SLURM_JOB_ID}

echo "Done. Outputs copied to ~/02-SRC/DreamerV3/results_summit/"
