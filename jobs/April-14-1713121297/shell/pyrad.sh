#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=pyradfeat
#SBATCH --output=pyradfeat.%j
#SBATCH --error=pyradfeat.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
###SBATCH --nodes=1
#SBATCH --cpus-per-task=4
###SBATCH --time=0-9:00:00

module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt

python -u '/scratch/a.bip5/BraTS/scripts/Analysis/pyradiomicfeatures.py' \
