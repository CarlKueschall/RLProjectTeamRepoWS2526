#!/bin/bash

#########################################################
# PBRS Magnitude × Reward Scaling Ablation Grid Launcher
#########################################################
# Submits all 12 ablation configurations
# Usage: bash launch_ablation_grid.sh
#########################################################

echo "=========================================="
echo "Launching PBRS × Reward Scale Ablation Grid"
echo "=========================================="
echo ""

# Array of all 12 sbatch files
sbatch_files=(
    "train_hockey_ablation_pbrs0.4_rscale0.10.sbatch"
    "train_hockey_ablation_pbrs0.4_rscale0.20.sbatch"
    "train_hockey_ablation_pbrs0.4_rscale0.30.sbatch"
    "train_hockey_ablation_pbrs1.0_rscale0.10.sbatch"
    "train_hockey_ablation_pbrs1.0_rscale0.20.sbatch"
    "train_hockey_ablation_pbrs1.0_rscale0.30.sbatch"
    "train_hockey_ablation_pbrs2.0_rscale0.10.sbatch"
    "train_hockey_ablation_pbrs2.0_rscale0.20.sbatch"
    "train_hockey_ablation_pbrs2.0_rscale0.30.sbatch"
    "train_hockey_ablation_pbrs4.0_rscale0.10.sbatch"
    "train_hockey_ablation_pbrs4.0_rscale0.20.sbatch"
    "train_hockey_ablation_pbrs4.0_rscale0.30.sbatch"
)

# Submit each sbatch file
for i in "${!sbatch_files[@]}"; do
    file="${sbatch_files[$i]}"
    job_num=$((i + 1))

    if [ -f "$file" ]; then
        echo "[${job_num}/12] Submitting: $file"
        sbatch "$file"
        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    else
        echo "[${job_num}/12] ERROR: $file not found!"
    fi
done

echo ""
echo "=========================================="
echo "All 12 jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u stud432"
echo "View run results: https://wandb.ai"
echo ""
