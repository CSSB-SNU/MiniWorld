#!/bin/bash
#SBATCH --job-name=mw_cifgeom
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/newfp32geom_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/newfp32geom_%j.out
# peptide C-N / clash check of the new fp32-logit phase-2a ep100 (patch off = bf16 hoist logits, patch on) vs GT; 4 hitm targets (CPU)
set -uo pipefail
cd /home/psk6950/MiniWorld
GT=/home/psk6950/data/foldbench_v240eval/ground_truth
N=/home/psk6950/MiniWorld/runs/foldbench/fusedlogit
ARGS=()
for t in 7xi5:a 7yhr:a 8t93:b 8pak:b; do id=${t%:*}; x=${t#*:}
  ARGS+=("${id}_GT=$GT/${id}-assembly1.cif" "${id}_new_bf16logit=$N/newfp32_p0_$x/${id}-assembly1/*_pred.cif" "${id}_new_fp32patch=$N/newfp32_p1_$x/${id}-assembly1/*_pred.cif" "${id}_oldmodel_fp32patch=$N/on2_$x/${id}-assembly1/*_pred.cif")
  [ "$x" = a ] && ARGS+=("${id}_oldmodel_bf16logit=$N/off_a/${id}-assembly1/*_pred.cif")
done
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/cif_geometry.py "${ARGS[@]}"
echo "==== EXIT $? ===="
