#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=nnunet
#SBATCH --output=%j.nnunet
#SBATCH --error=err.%j.n
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00

nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity