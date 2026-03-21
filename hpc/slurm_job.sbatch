#!/bin/bash
#SBATCH --job-name=grid-ppo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/grid-ppo-%j.out
#SBATCH --error=logs/grid-ppo-%j.err

# Example SLURM batch script for running distributed training
# 1. Adjust module/conda commands for your HPC environment
# 2. Ensure ray is installed in the active environment
# 3. Launch the script with: sbatch hpc/slurm_job.sbatch

module load python/3.10 || true
source "$HOME/.venvs/grid/bin/activate"

# Start Ray head on the allocated node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Run training and attach to the local Ray head
python -m hpc.train_distributed \
  --ray-address auto \
  --num-workers 16 \
  --envs-per-worker 4 \
  --num-gpus 1 \
  --data-path data/sample_data.csv \
  --checkpoint-freq 2 \
  --output-dir /scratch/$USER/grid-ppo

# Stop Ray to release resources
ray stop