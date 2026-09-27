#!/bin/bash
#SBATCH -J mw-triattn-bwd-installed
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/installed-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export TORCHINDUCTOR_CACHE_DIR="$R/inductor"
bash "$ENV" python -u -B "$R/check_installed.py"
for L in 384 768 1024; do
  bash "$ENV" python -u -B "$R/paired_installed.py" --length "$L" --output "$R/installed-paired-L$L.json"
done
