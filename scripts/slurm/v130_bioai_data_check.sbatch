#!/usr/bin/env bash
#SBATCH --job-name=mw-v130-data
#SBATCH --partition=cpu-short
#SBATCH --account=bioai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/psk6950/MiniWorld
#SBATCH --output=runs/v1.3.0/implementation_20260924/data-%j.log
set -euo pipefail
source scripts/bioai_v130_runtime.sh
python scripts/measure_distogram_sigma_data.py \
    --config configs/miniworld/distogram_diffusion_medium_v130_bioai.yaml \
    --num-items 64 --num-workers 4 \
    --output "runs/v1.3.0/implementation_20260924/target-L384-${SLURM_JOB_ID}.json"
python scripts/measure_distogram_sigma_data.py \
    --config configs/miniworld/distogram_diffusion_medium_v130_bioai_L768.yaml \
    --num-items 64 --num-workers 4 \
    --output "runs/v1.3.0/implementation_20260924/target-L768-${SLURM_JOB_ID}.json"
