#!/bin/bash
#SBATCH --job-name=summit-dv3-league80
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
# Summit Final Push - 80/10/10 League Continuation
# ============================================================================
# Target sampling:
#   - 80% checkpoint league (self-play pool)
#   - 10% weak anchor
#   - 10% strong anchor
#
# Implemented via:
#   --self_play_weak_ratio 0.2  (20% anchors total -> balanced weak/strong)
#   + bootstrap self-play pool from external checkpoint directory
# ============================================================================

# Resume checkpoint is expected in the same directory as train_hockey.py
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoints/summit-evaluation/opponents/634k.pth}"
OPPONENTS_DIR="${OPPONENTS_DIR:-checkpoints/summit-evaluation/opponents}"
RECOMMENDED_CSV="${RECOMMENDED_CSV:-checkpoints/summit-evaluation/results/pool_select_20260224_145948/recommended_pool.csv}"
LEAGUE_DIR="${LEAGUE_DIR:-checkpoints/summit-evaluation/league-optimal-634k}"
PROBE_FILE="${PROBE_FILE:-${LEAGUE_DIR}/league_checkpoints.txt}"
WEIGHTS_CSV="${WEIGHTS_CSV:-${LEAGUE_DIR}/league_weights.csv}"
RUN_NAME="summit-final24h-league80-seed42-${SLURM_JOB_ID}"

if [ ! -f "${CHECKPOINT_NAME}" ]; then
  echo "ERROR: Resume checkpoint not found: ${CHECKPOINT_NAME}"
  echo "Place it in $(pwd) or override CHECKPOINT_NAME in sbatch env."
  exit 1
fi

if [ ! -d "${OPPONENTS_DIR}" ]; then
  echo "ERROR: Opponents directory not found: ${OPPONENTS_DIR}"
  exit 1
fi

if [ ! -f "${RECOMMENDED_CSV}" ]; then
  echo "ERROR: Recommended pool CSV not found: ${RECOMMENDED_CSV}"
  exit 1
fi

# Build curated league directory (symlinks + weights) from recommended pool.
python3 scripts/prepare_league_pool.py \
  --recommended-csv "${RECOMMENDED_CSV}" \
  --source-dir "${OPPONENTS_DIR}" \
  --out-dir "${LEAGUE_DIR}" \
  --top-k 21 \
  --link-mode symlink

echo "Probe list: ${PROBE_FILE} ($(wc -l < "${PROBE_FILE}") checkpoints)"
echo "Weights CSV: ${WEIGHTS_CSV}"

# Optional: set WANDB_API_KEY before submission if needed.
export WANDB_API_KEY="${WANDB_API_KEY:-}"

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
    --warmup_episodes 0 \
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
    --self_play_start 1 \
    --self_play_pool_size 48 \
    --self_play_save_interval 200 \
    --self_play_weak_ratio 0.2 \
    --self_play_bootstrap_dir "${LEAGUE_DIR}" \
    --self_play_bootstrap_glob "*.pth" \
    --self_play_bootstrap_max 21 \
    --self_play_bootstrap_strategy weighted \
    --self_play_bootstrap_weights_csv "${WEIGHTS_CSV}" \
    --self_play_bootstrap_weight_column blend_score \
    --self_play_prior_alpha 0.45 \
    --use_pfsp \
    --pfsp_mode variance \
    \
    --probe_checkpoints_file "${PROBE_FILE}" \
    --probe_episodes 8 \
    --probe_max_checkpoints 21 \
    --opponent_window_size 200 \
    \
    --checkpoint_interval 500 \
    --eval_interval 500 \
    --eval_episodes 12 \
    --gif_interval 0 \
    --log_interval 5 \
    \
    --wandb_project rl-hockey

# Copy outputs back to home
mkdir -p ~/02-SRC/DreamerV3/results_summit_league80
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/results \
    ~/02-SRC/DreamerV3/results_summit_league80/results_${SLURM_JOB_ID}
cp -R /scratch/$SLURM_JOB_ID/02-SRC/DreamerV3/checkpoints \
    ~/02-SRC/DreamerV3/results_summit_league80/checkpoints_${SLURM_JOB_ID}
cp "job.${SLURM_JOB_ID}.out" ~/02-SRC/DreamerV3/results_summit_league80/ || true
cp "job.${SLURM_JOB_ID}.err" ~/02-SRC/DreamerV3/results_summit_league80/ || true

echo "Done. Outputs copied to ~/02-SRC/DreamerV3/results_summit_league80/"
