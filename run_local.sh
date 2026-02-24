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

cd /Users/carlkueschall/workspace/RLProjectHockey/02-SRC/DreamerV3 || exit 1

RUN_NAME="summit-local-league80-$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_NAME="results_checkpoints__weak_seed42_20260223_031428_474k.pth"
LEAGUE_DIR="checkpoints/summit-evaluation/opponents"
PROBE_FILE="probe_checkpoints_league80_local.txt"
DEVICE="mps"   # set to "cuda" if you're on a CUDA machine

[ -f "$CHECKPOINT_NAME" ] || { echo "Missing checkpoint: $CHECKPOINT_NAME"; exit 1; }
[ -d "$LEAGUE_DIR" ] || { echo "Missing league dir: $LEAGUE_DIR"; exit 1; }

find "$LEAGUE_DIR" -maxdepth 1 -name "*.pth" | sort > "$PROBE_FILE"

python3 train_hockey.py \
  --config hockey.yml \
  --mode NORMAL \
  --seed 42 \
  --device "$DEVICE" \
  --run_name "$RUN_NAME" \
  --resume "$CHECKPOINT_NAME" \
  --gradient_steps 1000000000 \
  --replay_ratio 16 \
  --warmup_episodes 0 \
  --interaction_episodes 1 \
  --batch_size 32 \
  --batch_length 32 \
  --imagination_horizon 15 \
  --recurrent_size 256 \
  --latent_length 16 \
  --latent_classes 16 \
  --encoded_obs_size 256 \
  --uniform_mix 0.01 \
  --lr_world 0.0002 \
  --lr_actor 0.00005 \
  --lr_critic 0.00007 \
  --discount 0.997 \
  --lambda_ 0.95 \
  --entropy_scale 0.0003 \
  --free_nats 1.0 \
  --gradient_clip 100 \
  --buffer_capacity 250000 \
  --self_play_start 1 \
  --self_play_pool_size 48 \
  --self_play_save_interval 200 \
  --self_play_weak_ratio 0.2 \
  --self_play_bootstrap_dir "$LEAGUE_DIR" \
  --self_play_bootstrap_glob "*.pth" \
  --self_play_bootstrap_max 24 \
  --self_play_bootstrap_strategy uniform \
  --use_pfsp \
  --pfsp_mode variance \
  --probe_checkpoints_file "$PROBE_FILE" \
  --probe_episodes 6 \
  --probe_max_checkpoints 12 \
  --opponent_window_size 200 \
  --checkpoint_interval 250 \
  --eval_interval 250 \
  --eval_episodes 12 \
  --gif_interval 0 \
  --log_interval 10 \
  --wandb_project rl-hockey

mkdir -p results_local_archive
cp -R results "results_local_archive/results_${RUN_NAME}"
