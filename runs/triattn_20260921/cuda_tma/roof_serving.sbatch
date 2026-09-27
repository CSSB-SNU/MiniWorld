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
for V in baseline cuda; do
 TMA_VARIANT=$V PYTHONNOUSERSITE=1 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed --section Occupancy --section WarpStateStats -o "$R/cuda_tma/serving-roof-$V" --force-overwrite python -u -B "$R/cuda_tma/roof_serving.py" --length 768 --output "$R/cuda_tma/serving-roof-$V.json"
 env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/cuda_tma/serving-roof-$V.ncu-rep" --page raw --csv > "$R/cuda_tma/serving-roof-$V.csv"
done
