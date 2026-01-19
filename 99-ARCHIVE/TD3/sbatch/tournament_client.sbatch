#!/bin/bash
#SBATCH --job-name=hockey_tournament
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=24:00:00
#SBATCH --error=tournament.%J.err
#SBATCH --output=tournament.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

# Copy code to scratch
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/

# Navigate to tournament client
cd /scratch/$SLURM_JOB_ID/02-SRC/comprl-hockey-agent

# Set environment variables for cluster
export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=65335
export COMPRL_ACCESS_TOKEN=8a293a11-aeea-483b-8d37-01d81fde656e

# Run tournament client in container (no conda activation needed inside container)
singularity exec --nv ~/containers/hockey.sif python3 run_client.py \
  --args --agent=td3 \
  --model-path=../TD3/results/checkpoints/results_checkpoints_TD3_Hockey_NORMAL_weak_30250_seed48.pth
