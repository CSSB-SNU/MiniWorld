#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=b1-gate-bound
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_b1_gate_bound_20260921/job-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
for length in 384 768; do
 bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_b1_gate_bound_20260921/measure.py --length "$length"
 bash runs/anthropic_adoption_20260919/env.sh ncu --profile-from-start off --set full --cache-control none --clock-control none --kernel-name regex:^b1_gate_only$ --launch-count 1 --force-overwrite -o "runs/trimul_b1_gate_bound_20260921/ncu-L${length}" python -B runs/trimul_b1_gate_bound_20260921/measure.py --length "$length" --profile
 bash runs/anthropic_adoption_20260919/env.sh ncu --import "runs/trimul_b1_gate_bound_20260921/ncu-L${length}.ncu-rep" --page raw --csv --log-file "runs/trimul_b1_gate_bound_20260921/ncu-L${length}-raw.csv"
done
