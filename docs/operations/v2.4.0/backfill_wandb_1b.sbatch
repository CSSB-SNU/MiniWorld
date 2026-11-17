#!/bin/bash
#SBATCH --job-name=mw_1b_wandb_backfill
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld-scratch/phase1b-v240/backfill_%j.out
cd /home/psk6950/MiniWorld
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=team_gm WANDB_PROJECT=MiniWorld
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 python docs/operations/v2.4.0/backfill_wandb_1b.py \
  runs/v2.4.0/phase1b/medium_distogram_L768/run1/train.log v240-phase1b-medium-L768-backfill
