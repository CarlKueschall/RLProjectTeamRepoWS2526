#!/bin/zsh
set -euo pipefail

# Activate conda env robustly without relying on interactive ~/.zshrc.
if [[ "${CONDA_DEFAULT_ENV:-}" != "py310" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.zsh hook)"
  elif [[ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found. Activate py310 manually, then rerun."
    exit 1
  fi
  conda activate py310
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
DREAMER_DIR="${PROJECT_ROOT}/02-SRC/DreamerV3"
cd "${DREAMER_DIR}"

# Core run settings (mirrors train_league.sbatch)
CHECKPOINT_NAME="634k.pth"
RUN_NAME="summit-final24h-league80-local-seed42-$(date +%Y%m%d_%H%M%S)"
LEAGUE_DIR="checkpoints/summit-evaluation/league-optimal-634k"
PROBE_FILE="${LEAGUE_DIR}/league_checkpoints.txt"
WEIGHTS_CSV="${LEAGUE_DIR}/league_weights.csv"
RECOMMENDED_CSV="checkpoints/summit-evaluation/results/pool_select_20260224_145948/recommended_pool.csv"
DEVICE="${DEVICE:-mps}" # set DEVICE=cuda on CUDA machines

echo "Running local league continuation:"
echo "  checkpoint: ${CHECKPOINT_NAME}"
echo "  run name:   ${RUN_NAME}"
echo "  device:     ${DEVICE}"
echo "  league dir: ${LEAGUE_DIR}"

# Sanity checks
[[ -f "${CHECKPOINT_NAME}" ]] || { echo "ERROR: Resume checkpoint not found: ${CHECKPOINT_NAME}"; exit 1; }
[[ -f "${RECOMMENDED_CSV}" ]] || { echo "ERROR: Recommended pool CSV not found: ${RECOMMENDED_CSV}"; exit 1; }
[[ -d "checkpoints/summit-evaluation/opponents" ]] || {
  echo "ERROR: Source opponents dir not found: checkpoints/summit-evaluation/opponents"
  exit 1
}

# Rebuild local league directory from recommended pool (same as sbatch)
python3 prepare_league_pool.py \
  --recommended-csv "${RECOMMENDED_CSV}" \
  --source-dir checkpoints/summit-evaluation/opponents \
  --out-dir "${LEAGUE_DIR}" \
  --top-k 21 \
  --link-mode copy

[[ -d "${LEAGUE_DIR}" ]] || { echo "ERROR: League directory not found: ${LEAGUE_DIR}"; exit 1; }
[[ -f "${PROBE_FILE}" ]] || { echo "ERROR: Probe file not found: ${PROBE_FILE}"; exit 1; }
[[ -f "${WEIGHTS_CSV}" ]] || { echo "ERROR: Weights CSV not found: ${WEIGHTS_CSV}"; exit 1; }

python3 train_hockey.py \
  --config hockey.yml \
  --mode NORMAL \
  --seed 42 \
  --device "${DEVICE}" \
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
  --checkpoint_interval 250 \
  --eval_interval 250 \
  --eval_episodes 12 \
  --gif_interval 0 \
  --log_interval 10 \
  \
  --wandb_project rl-hockey

# Archive local outputs
mkdir -p results_local_archive
cp -R results "results_local_archive/results_${RUN_NAME}"
cp -R checkpoints "results_local_archive/checkpoints_${RUN_NAME}"
