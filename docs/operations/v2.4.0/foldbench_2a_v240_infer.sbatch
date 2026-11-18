#!/bin/bash
#SBATCH --job-name=mw_v240_2a_fb
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/v2.4.0-2a-from1b/fb_%A_%a.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/v2.4.0-2a-from1b/fb_%A_%a.out
# FoldBench inference of the v2.4.0 phase-2a model (frozen phase-1b trunk, epoch 700 EMA seed + EDM diffusion head), submitted with
# --dependency=afterok:<2a training job>. FULL FoldBench (1,522 targets) x 5 seeds x 5 diffusion samples, 10 recycles, 200 steps (the AF3 /
# Protenix protocol of docs/foldbench-evaluate), EMA weights, sequential sample path (--sample-batch corrupts the CUDA context, docs/known-issues.md).
# 8 array tasks = 8 stride shards of the size-sorted index, 1 GPU each; finished targets are skipped, so a rerun of a task resumes.
# Inputs: ~/data/foldbench_v240eval = original FoldBench + the fixed nucleic inputs. Scoring afterwards: scripts/slurm/phase2b/foldbench_score.sbatch.
set -uo pipefail
cd /home/psk6950/MiniWorld
ROOT=/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/medium_diffusion_L384_from1b
OUT=/home/psk6950/MiniWorld/runs/foldbench/v2.4.0-2a-from1b
mkdir -p "$OUT"
CK=$(ls -t "$ROOT"/*/*/checkpoints/last.pt 2>/dev/null | head -1)
[ -f "$CK" ] || { echo "FATAL: no last.pt under $ROOT"; exit 1; }
echo "=== FoldBench 2a | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node $SLURM_JOB_NODELIST | ckpt $CK"
export LD_LIBRARY_PATH="$HOME/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export MW_COMPILE_TIMEOUT=1 MINIWORLD_TRANSITION_FORCE_SPLIT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${USER}_fb_shared; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 CK="$CK" OUT="$OUT" pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python -c "import torch;d=torch.load(\"$CK\",map_location=\"cpu\",weights_only=False);print(\"  ckpt epoch\",d[\"epoch\"],\"step\",d.get(\"global_step\"))"
  python scripts/run_miniworld_diffusion_inference.py foldbench \
    --config configs/miniworld/phase2a_diffusion_medium_v240.yaml \
    --ckpt "$CK" \
    --index docs/operations/v2.4.0/foldbench_full_index.txt \
    --inputs-root /home/psk6950/data/foldbench_v240eval/inputs \
    --output-dir "$OUT" \
    --shard $SLURM_ARRAY_TASK_ID --n-shards 8 \
    --n-seeds 5 --n-samples 5 --num-recycles 10 --timesteps 200 --msa-subsample --use-ema'
rc=$?
echo "==== task $SLURM_ARRAY_TASK_ID EXIT $rc ===="
exit "$rc"
