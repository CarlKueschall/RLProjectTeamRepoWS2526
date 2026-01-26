#!/bin/bash
# Ablation Study Scripts for DreamerV3 Hockey Report
#
# These ablations demonstrate the value of our key modifications:
# 1. DreamSmooth (temporal reward smoothing)
# 2. Auxiliary Tasks (goal prediction, distance, shot quality)
#
# Each ablation should run for ~100k gradient steps minimum to show meaningful differences
# Expected runtime: ~10-15 hours per ablation on GPU

# Common base configuration (matches Phase 1 but shorter for ablation)
BASE_CONFIG="--seed 42 \
    --gradient_steps 150000 \
    --replay_ratio 32 \
    --warmup_episodes 100 \
    --lr_world 0.0003 \
    --lr_actor 0.0001 \
    --lr_critic 0.0001 \
    --entropy_scale 0.0003 \
    --mixed_opponents \
    --mixed_weak_prob 0.5 \
    --eval_interval 5000 \
    --checkpoint_interval 10000"

echo "=============================================="
echo "DreamerV3 Ablation Studies for Report"
echo "=============================================="
echo ""
echo "Available ablations:"
echo "  1. DreamSmooth ON vs OFF"
echo "  2. Auxiliary Tasks ON vs OFF"
echo ""
echo "Run each ablation separately or use --all to run sequentially"
echo ""

# Parse arguments
ABLATION=$1

run_dreamsmooth_ablation() {
    echo "=============================================="
    echo "ABLATION 1: DreamSmooth"
    echo "=============================================="

    # Baseline: WITH DreamSmooth (our approach)
    echo "[1/2] Running WITH DreamSmooth..."
    python3 train_hockey.py $BASE_CONFIG \
        --use_dreamsmooth \
        --dreamsmooth_alpha 0.5 \
        --wandb_name "ABLATION-dreamsmooth-ON" \
        --results_dir results/ablation_dreamsmooth_on

    # Ablation: WITHOUT DreamSmooth
    echo "[2/2] Running WITHOUT DreamSmooth..."
    python3 train_hockey.py $BASE_CONFIG \
        --wandb_name "ABLATION-dreamsmooth-OFF" \
        --results_dir results/ablation_dreamsmooth_off

    echo "DreamSmooth ablation complete!"
}

run_auxiliary_ablation() {
    echo "=============================================="
    echo "ABLATION 2: Auxiliary Tasks"
    echo "=============================================="

    # Baseline: WITH Auxiliary Tasks (our approach - default is ON)
    echo "[1/2] Running WITH Auxiliary Tasks..."
    python3 train_hockey.py $BASE_CONFIG \
        --use_dreamsmooth \
        --dreamsmooth_alpha 0.5 \
        --wandb_name "ABLATION-auxiliary-ON" \
        --results_dir results/ablation_auxiliary_on

    # Ablation: WITHOUT Auxiliary Tasks
    echo "[2/2] Running WITHOUT Auxiliary Tasks..."
    python3 train_hockey.py $BASE_CONFIG \
        --use_dreamsmooth \
        --dreamsmooth_alpha 0.5 \
        --no_aux_tasks \
        --wandb_name "ABLATION-auxiliary-OFF" \
        --results_dir results/ablation_auxiliary_off

    echo "Auxiliary tasks ablation complete!"
}

case $ABLATION in
    dreamsmooth)
        run_dreamsmooth_ablation
        ;;
    auxiliary)
        run_auxiliary_ablation
        ;;
    all)
        run_dreamsmooth_ablation
        echo ""
        run_auxiliary_ablation
        ;;
    *)
        echo "Usage: $0 {dreamsmooth|auxiliary|all}"
        echo ""
        echo "Examples:"
        echo "  $0 dreamsmooth    # Run DreamSmooth ablation"
        echo "  $0 auxiliary      # Run Auxiliary Tasks ablation"
        echo "  $0 all            # Run all ablations sequentially"
        exit 1
        ;;
esac
