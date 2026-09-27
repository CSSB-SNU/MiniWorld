#!/bin/bash
#SBATCH -J mw-ta-tma
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/cuda_tma/test-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/cuda_tma/bench.py" --length 768 --output "$R/cuda_tma/results-L768.json"

bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section SourceCounters -o "$R/cuda_tma/warp-profile" --force-overwrite python -u -B "$R/cuda_tma/roof.py" --length 768 --output "$R/cuda_tma/roof.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/cuda_tma/warp-profile.ncu-rep" --page raw --csv > "$R/cuda_tma/warp-profile.csv"
