#!/bin/bash
#SBATCH --job-name=mw_v240_stereo
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/eval_results/foldbench/stereo_v240_2a_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/eval_results/foldbench/stereo_v240_2a_%j.out
# Stereo audit of the v2.4.0 phase-2a epoch-100 FoldBench predictions: OST lDDT with vs without the stereochemistry checks (+ TM-score control),
# 25 random targets per category x (seed0, sample0). CPU only (OST containers), results cached per target in --out.
set -uo pipefail
cd /home/psk6950/MiniWorld
/home/psk6950/data/foldbench/upstream/.venv/bin/python scripts/foldbench_stereo_audit.py \
  --pred-dir runs/foldbench/v2.4.0-2a-from1b-ep100 \
  --out runs/eval_results/foldbench/stereo_audit_v240_2a_ep100 --per-category 25 --workers 30
echo "==== EXIT $? ===="
