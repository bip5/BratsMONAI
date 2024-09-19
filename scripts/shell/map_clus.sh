#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=ECluster
#SBATCH --output=/scratch/a.bip5/BraTS/training_logs/clusterse.%j
#SBATCH --error=/scratch/a.bip5/BraTS/training_logs/clusterse.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
###SBATCH --nodes=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

conda list >> /scratch/a.bip5/BraTS/training_logs/result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/Analysis/PixelCluster.py' \