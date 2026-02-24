#!/usr/bin/env bash
set -euo pipefail

# Local launcher for Summit Final Push (Phase A style).
# Uses more frequent eval/checkpointing and robustness probe metrics.
#
# Usage examples:
#   ./run_summit_final24h_local.sh
#   RESUME_CKPT=336k_0130_193448.pth RUN_NAME=myrun ./run_summit_final24h_local.sh

ROOT_DIR="/Users/carlkueschall/workspace/RLProjectHockey/02-SRC/DreamerV3"
cd "${ROOT_DIR}"

# Activate conda env
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate py310

# Configurable knobs via env vars
RESUME_CKPT="${RESUME_CKPT:-checkpoints/training/357k_0130_193448.pth}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-summit-final24h-local-seed${SEED}-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-mps}"   # use cuda on Linux GPU boxes

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "ERROR: resume checkpoint not found in ${ROOT_DIR}: ${RESUME_CKPT}"
  exit 1
fi

echo "Running local summit continuation:"
echo "  checkpoint: ${RESUME_CKPT}"
echo "  run name:   ${RUN_NAME}"
echo "  device:     ${DEVICE}"

python train_hockey.py \
    --config hockey.yml \
    --mode NORMAL \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --run_name "${RUN_NAME}" \
    \
    --resume "${RESUME_CKPT}" \
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
    --probe_checkpoints_file checkpoints/probe_checkpoints_summit_top4.txt \
    --probe_episodes 8 \
    --opponent_window_size 200 \
    \
    --checkpoint_interval 500 \
    --eval_interval 250 \
    --eval_episodes 25 \
    --gif_interval 0 \
    --log_interval 10 \
    \
    --wandb_project rl-hockey

