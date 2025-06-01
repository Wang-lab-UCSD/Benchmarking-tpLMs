#!/bin/bash
#SBATCH -p gpu -w gpu-3               # job submitted to gpu
#SBATCH -N 1                      # 
###SBATCH --ntasks-per-node=1       # 
###SBATCH --cpus-per-task=4         # 
#SBATCH --mem 50G
#SBATCH -t 10-10:00:00               # 
###SBATCH --array=1
#SBATCH -J ankh
#SBATCH --output=%A.out 

python3 esm2_generation.py
