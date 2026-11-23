#!/bin/bash
#SBATCH --job-name=mw_trajpep
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/trajpep_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/trajpep_%j.out
# peptide C-N bond of the saved x0-hat trajectories (8oxk, 9jst), per solver step. CPU only.
set -uo pipefail
cd /home/psk6950/MiniWorld
R=runs/foldbench/x0hat_traj_ep100
for t in 8oxk 9jst; do
  CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/traj_peptide.py $R/$t-assembly1.npz $R/preds/$t-assembly1/${t}-assembly1_seed0_sample0_pred.cif
done
echo "==== EXIT $? ===="
