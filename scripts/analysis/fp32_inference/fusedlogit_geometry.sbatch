#!/bin/bash
#SBATCH --job-name=mw_fl_geom
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/fusedlogit/geom_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/fusedlogit/geom_%j.out
# peptide / CA-CA / intra bond lengths and clashes: GT vs the original (fused bf16) vs diffusion fp32 vs the fused path with fp32 pair-bias logits (CPU).
set -uo pipefail
cd /home/psk6950/MiniWorld
GT=/home/psk6950/data/foldbench_v240eval/ground_truth
ARGS=()
for t in 7xi5 7yhr 8t93 8pak; do
  case $t in 7xi5|7yhr) d=on2_a;; *) d=on2_b;; esac
  ARGS+=("${t}_GT=$GT/$t-assembly1.cif"
         "${t}_orig_bf16=runs/foldbench/v2.4.0-2a-from1b-ep100/$t-assembly1/*_pred.cif"
         "${t}_all_fp32=runs/foldbench/v2.4.0-2a-ep100-dfp32-hitm/$t-assembly1/*_pred.cif"
         "${t}_fused_logit32=runs/foldbench/fusedlogit/$d/$t-assembly1/*_pred.cif")
done
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/cif_geometry.py "${ARGS[@]}"
echo "==== EXIT $? ===="
