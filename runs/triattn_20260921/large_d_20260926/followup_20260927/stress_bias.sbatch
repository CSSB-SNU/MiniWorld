#!/bin/bash
#SBATCH -J mw-triattn-wide-bias-stress
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 00:05:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/bias-stress-%j.log
set -euo pipefail
timeout 120s bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/stress_bias.py
