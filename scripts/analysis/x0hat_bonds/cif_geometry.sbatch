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
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/cifgeom_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/cifgeom_%j.out
# peptide / CA-CA / intra-residue bond lengths and HEAVY-atom clashes of every FoldBench prediction variant (8oxk, 9jst) vs the ground truth (CPU only).
set -uo pipefail
cd /home/psk6950/MiniWorld
GT=/home/psk6950/data/foldbench_v240eval/ground_truth
P=/home/psk6950/miniworld-engine/foldbench_cifs
N=/home/psk6950/MiniWorld/runs/foldbench
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python scripts/analysis/x0hat_bonds/cif_geometry.py \
  "8oxk_GT=$GT/8oxk-assembly1.cif" \
  "8oxk_engine_bf16_EMA=$P/8oxk-assembly1/engine_bf16_original/*_pred.cif" \
  "8oxk_engine_bf16_RAW=$N/v2.4.0-2a-ep100-noema-8oxk/8oxk-assembly1/*_pred.cif" \
  "8oxk_pytorch_dfp32(SWAeng)=$P/8oxk-assembly1/pytorch_diffusion_fp32/*_pred.cif" \
  "8oxk_pure_pytorch_100pct=$P/8oxk-assembly1/pure_pytorch_100pct/*_pred.cif" \
  "8oxk_pure_pt_oldrope=$P/8oxk-assembly1/pure_pytorch_old_bf16_rope/*_pred.cif" \
  "9jst_GT=$GT/9jst-assembly1.cif" \
  "9jst_engine_bf16=$P/9jst-assembly1/engine_bf16_original/*_pred.cif" \
  "9jst_engine_dfp32=$P/9jst-assembly1/engine_diffusion_fp32/*_pred.cif"
echo "==== EXIT $? ===="
