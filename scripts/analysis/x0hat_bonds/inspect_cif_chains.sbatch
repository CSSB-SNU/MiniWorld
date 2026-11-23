#!/bin/bash
#SBATCH --job-name=mw_inspect_cif
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_hitm/inspect_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_hitm/inspect_%j.out
set -uo pipefail
cd /home/psk6950/MiniWorld
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/inspect_cif_chains.py \
  /home/psk6950/data/foldbench_v240eval/ground_truth/8q79-assembly1.cif \
  runs/foldbench/v2.4.0-2a-from1b-ep100/8q79-assembly1/8q79-assembly1_seed0_sample0_pred.cif \
  runs/foldbench/v2.4.0-2a-from1b-ep100/7xi5-assembly1/7xi5-assembly1_seed0_sample0_pred.cif
echo "==== EXIT $? ===="
