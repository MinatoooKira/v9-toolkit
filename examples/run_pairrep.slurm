#!/bin/bash
#SBATCH --job-name=v9_pairrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu_5090
#SBATCH --time=08:00:00
#SBATCH --output=logs/pairrep_%j.log
#SBATCH --error=logs/pairrep_%j.err

# IMPORTANT — set up env (edit paths to your install):
source /etc/profile
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate protenix_lynx
module load cuda/12.8

export LAYERNORM_TYPE=torch
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}
mkdir -p $MPLCONFIGDIR logs

# Use full conda env path NOT recommended — `conda activate` sets LD_LIBRARY_PATH
# correctly, which is required for CUDA to find libraries.

CONFIG=${1:-examples/PTEN_HUMAN.yaml}
echo "=== v9 pair-rep extraction | config=$CONFIG | host=$(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -m v9pipeline.cli extract --config "$CONFIG"

echo "=== Done at $(date) ==="
