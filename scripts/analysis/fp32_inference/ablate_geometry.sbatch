#!/bin/bash
#SBATCH --job-name=mw_abl_geom
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/ablate/geom_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/ablate/geom_%j.out
# peptide / CA-CA / intra bond lengths and clashes of the dtype-ablation predictions (CPU). VARIANTS env: space-separated list of ablation run names.
set -uo pipefail
cd /home/psk6950/MiniWorld
GT=/home/psk6950/data/foldbench_v240eval/ground_truth
A=runs/foldbench/ablate
ARGS=()
for t in 7xi5 7yhr; do
  ARGS+=("${t}_GT=$GT/$t-assembly1.cif"
         "${t}_ALL_bf16_orig=runs/foldbench/v2.4.0-2a-from1b-ep100/$t-assembly1/*_pred.cif"
         "${t}_ALL_fp32=runs/foldbench/v2.4.0-2a-ep100-dfp32-hitm/$t-assembly1/*_pred.cif")
  for v in $VARIANTS; do ARGS+=("${t}_${v}=$A/$v/$t-assembly1/*_pred.cif"); done
done
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/cif_geometry.py "${ARGS[@]}"
echo "==== EXIT $? ===="
