#!/bin/bash
#SBATCH -J mw-ta-source
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/source-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
PYTHONNOUSERSITE=1 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/ncu --profile-from-start off --kernel-name-base demangled --kernel-name "${SOURCE_KERNEL_FILTER:-regex:ta_sol_${SOL_VARIANT}::triattn_m1_kernel.*1073741824}" --clock-control none --cache-control none --section SourceCounters --section InstructionStats --section WarpStateStats -o "$R/core_sol90/${SOL_VARIANT}-source" --force-overwrite python -u -B "$R/core_sol90/roof.py" --length 768 --output "$R/core_pipeline/roof.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/core_sol90/${SOL_VARIANT}-source.ncu-rep" --page raw --csv > "$R/core_sol90/${SOL_VARIANT}-source.csv"
