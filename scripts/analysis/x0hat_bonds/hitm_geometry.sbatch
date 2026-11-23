#!/bin/bash
#SBATCH --job-name=mw_hitm_geom
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_hitm/hitm_geom_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_hitm/hitm_geom_%j.out
# Peptide / CA-CA / intra bond lengths + heavy-atom clashes of the high-TM low-lDDT targets: GT vs original bf16 predictions vs diffusion-fp32 predictions (CPU),
# then the peptide C-N trajectory of the saved bf16 x0-hat (1 seed x 1 sample) for each target.
set -uo pipefail
cd /home/psk6950/MiniWorld
GT=/home/psk6950/data/foldbench_v240eval/ground_truth
B=runs/foldbench/v2.4.0-2a-from1b-ep100
F=runs/foldbench/v2.4.0-2a-ep100-dfp32-hitm
ARGS=()
for t in 8q79 7xi5 7yhr 8t93 8pak; do
  ARGS+=("${t}_GT=$GT/$t-assembly1.cif" "${t}_bf16_x25=$B/$t-assembly1/*_pred.cif" "${t}_dfp32_x25=$F/$t-assembly1/*_pred.cif")
done
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/cif_geometry.py "${ARGS[@]}"
R=runs/foldbench/x0hat_traj_hitm
for t in 8q79 7xi5 7yhr 8t93 8pak; do
  CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/traj_peptide.py $R/$t-assembly1.npz $R/preds/$t-assembly1/${t}-assembly1_seed0_sample0_pred.cif
done
echo "==== EXIT $? ===="
