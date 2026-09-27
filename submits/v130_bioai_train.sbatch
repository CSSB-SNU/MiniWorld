#!/usr/bin/env bash
# New v1.3 run. v1.2 CE checkpoints are NOT optimizer-resume compatible.
#SBATCH --job-name=mw-v130-PF16
#SBATCH --partition=gpu-4farm
#SBATCH --account=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=340G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/psk6950/MiniWorld
#SBATCH --output=runs/v1.3.0/train-%j.log
set -euo pipefail
source scripts/bioai_v130_runtime.sh
v130_config=${V130_CONFIG:-configs/miniworld/distogram_diffusion_medium_v130_bioai.yaml}
v130_resume=()
if [[ -n "${V130_CHECKPOINT:-}" ]]; then
    v130_resume=(--ckpt "$V130_CHECKPOINT")
fi
# The config selects compile + CUDA Graph for fresh starts and resumes alike.
# A standard .../checkpoints/last.pt carries its run directory and W&B identity.
exec torchrun --standalone --nproc_per_node=4 \
    scripts/run_miniworld_distogram_train.py train \
    --config "$v130_config" --job-name v1.3.0-distogram-diffusion-PF16 \
    "${v130_resume[@]}" train.force_trainer=cudagraph train.compile=true
