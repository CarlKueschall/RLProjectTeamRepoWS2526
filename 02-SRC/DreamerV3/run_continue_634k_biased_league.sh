#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Robust conda activation without sourcing user shell rc files.
if [[ -n "${CONDA_EXE:-}" ]]; then
  eval "$($CONDA_EXE shell.zsh hook)"
elif [[ -f /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh ]]; then
  source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
else
  echo "ERROR: conda initialization script not found."
  exit 1
fi
conda activate py310

RECOMMENDED_CSV="${RECOMMENDED_CSV:-checkpoints/summit-evaluation/results/pool_select_20260224_145948/recommended_pool.csv}"
OPPONENTS_DIR="${OPPONENTS_DIR:-checkpoints/summit-evaluation/opponents}"
LEAGUE_DIR="${LEAGUE_DIR:-checkpoints/summit-evaluation/league-optimal-634k}"
CHECKPOINT="${CHECKPOINT:-checkpoints/summit-evaluation/opponents/634k.pth}"
DEVICE="${DEVICE:-mps}"
RUN_NAME="${RUN_NAME:-summit-continue-634k-biased-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$RECOMMENDED_CSV" ]]; then
  echo "ERROR: recommended pool CSV not found: $RECOMMENDED_CSV"
  exit 1
fi
if [[ ! -d "$OPPONENTS_DIR" ]]; then
  echo "ERROR: opponents directory not found: $OPPONENTS_DIR"
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: resume checkpoint not found: $CHECKPOINT"
  exit 1
fi

python3 prepare_league_pool.py \
  --recommended-csv "$RECOMMENDED_CSV" \
  --source-dir "$OPPONENTS_DIR" \
  --out-dir "$LEAGUE_DIR" \
  --top-k 21 \
  --link-mode symlink

PROBE_FILE="$LEAGUE_DIR/league_checkpoints.txt"
WEIGHTS_CSV="$LEAGUE_DIR/league_weights.csv"

if [[ ! -f "$PROBE_FILE" || ! -f "$WEIGHTS_CSV" ]]; then
  echo "ERROR: league prep outputs missing in $LEAGUE_DIR"
  exit 1
fi

echo "Running continuation training:"
echo "  run:        $RUN_NAME"
echo "  checkpoint: $CHECKPOINT"
echo "  device:     $DEVICE"
echo "  league dir: $LEAGUE_DIR"

python3 train_hockey.py \
  --config hockey.yml \
  --mode NORMAL \
  --seed 42 \
  --device "$DEVICE" \
  --run_name "$RUN_NAME" \
  \
  --resume "$CHECKPOINT" \
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
  --self_play_bootstrap_dir "$LEAGUE_DIR" \
  --self_play_bootstrap_glob "*.pth" \
  --self_play_bootstrap_max 21 \
  --self_play_bootstrap_strategy weighted \
  --self_play_bootstrap_weights_csv "$WEIGHTS_CSV" \
  --self_play_bootstrap_weight_column blend_score \
  --self_play_prior_alpha 0.45 \
  --use_pfsp \
  --pfsp_mode variance \
  \
  --probe_checkpoints_file "$PROBE_FILE" \
  --probe_episodes 8 \
  --probe_max_checkpoints 21 \
  --opponent_window_size 200 \
  \
  --checkpoint_interval 250 \
  --eval_interval 250 \
  --eval_episodes 16 \
  --gif_interval 0 \
  --log_interval 5 \
  \
  --wandb_project rl-hockey \
  "$@"
