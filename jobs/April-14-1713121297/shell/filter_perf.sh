#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=filter_perf
#SBATCH --output=filter_perf.%j
#SBATCH --error=filter_perf.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
###SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --time=0-12:00:00


module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/Analysis/filter_perf.py' \