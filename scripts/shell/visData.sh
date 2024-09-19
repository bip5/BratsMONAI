#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=visD
#SBATCH --output=visData.out.%j
#SBATCH --error=visData.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00

python -u '/scratch/a.bip5/BraTS 2021/scripts/dataVis.py' \
--train_folder '/scratch/a.bip5/BraTS 2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/*'