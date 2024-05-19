#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=Eval_ensemble
#SBATCH --output=%j.Eval
#SBATCH --error=err.%j.Eval
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/monai_eval.py' \


###########Va1 1 to get dice score for each training sample /scratch/a.bip5/BraTS 2021/ssensemblemodels0922/Evaluation Folder1
