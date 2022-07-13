#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=elfsim
#SBATCH --output=%j.elfsim
#SBATCH --error=err.%j.elfsim
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u changeLabel.py