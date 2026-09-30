#!/usr/bin/env bash
#SBATCH --job-name=mw-v130-smoke
#SBATCH --partition=gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/psk6950/MiniWorld
#SBATCH --output=runs/v1.3.0/implementation_20260924/smoke-%j.log
set -euo pipefail
source scripts/bioai_v130_runtime.sh
python -m pytest tests/test_distogram_diffusion_v130.py tests/test_distogram_v110.py -q
torchrun --standalone --nproc_per_node=4 scripts/check_distogram_diffusion_v130.py --compile \
  --output "runs/v1.3.0/implementation_20260924/gpu-smoke-${SLURM_JOB_ID}.json"
