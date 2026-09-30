#!/usr/bin/env bash
#SBATCH --job-name=mw-v130-graph-check
#SBATCH --partition=gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/psk6950/MiniWorld
#SBATCH --output=runs/v1.3.0/implementation_20260924/graph-check-%j.log
set -euo pipefail
source scripts/bioai_v130_runtime.sh
torchrun --standalone --nproc_per_node=4 scripts/check_distogram_diffusion_graph_v130.py \
  --output-dir "runs/v1.3.0/implementation_20260924/graph-check-${SLURM_JOB_ID}"
