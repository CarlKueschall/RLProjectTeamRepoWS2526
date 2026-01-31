#!/bin/bash
# Launch all ablation experiments

cd ~

echo "Launching ablation experiments..."

sbatch ablation_dreamsmooth_on.sbatch
sbatch ablation_dreamsmooth_off.sbatch
sbatch ablation_auxiliary_on.sbatch
sbatch ablation_auxiliary_off.sbatch
sbatch ablation_twohot_on.sbatch
sbatch ablation_twohot_off.sbatch
sbatch ablation_selfplay_on.sbatch
sbatch ablation_selfplay_off.sbatch

echo "All 8 ablation jobs submitted."
