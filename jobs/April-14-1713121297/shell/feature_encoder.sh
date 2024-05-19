#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai_dev
#SBATCH --job-name=Encode
#SBATCH --output=Encode.%j
#SBATCH --error=Encode.err.%j
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
###SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/Analysis/encoded_features.py' \