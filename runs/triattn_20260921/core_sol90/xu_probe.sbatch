#!/bin/bash
#SBATCH -J mw-ta-xu
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/xu-probe-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --kernel-name regex:xu_probe --clock-control none --metrics sm__inst_executed_pipe_xu.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum --force-overwrite -o "$R/xu_probe" "$R/xu_probe"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/xu_probe.ncu-rep" --page raw --csv > "$R/xu_probe.csv"
