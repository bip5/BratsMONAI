#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=resnet_a
#SBATCH --output=/scratch/a.bip5/BraTS/training_logs/resnet.%j
#SBATCH --error=/scratch/a.bip5/BraTS/training_logs/resnet.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
####SBATCH --time=0-12:00:00

module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/Training/training.py'