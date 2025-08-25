#!/bin/sh 
#SBATCH -J run_af3
#SBATCH --mem=324g
#SBATCH -p h100
#SBATCH -w node02
#SBATCH -c 64
#SBATCH -o ./logs/test_dataloader.log
#SBATCH -e ./logs/test_dataloader.err

  
export OMP_NUM_THREADS=64
python -u MiniWorld/data/dataloader_multistate.py