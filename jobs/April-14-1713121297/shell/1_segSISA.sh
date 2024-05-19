#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=resnet_a
#SBATCH --output=resnet.%j
#SBATCH --error=resnet.err.%j
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS 2021/scripts/sisa.py' --model SegResNet --load_save 0 --load_path "SegResNetCV5ms1000rs4A" --fold_num 1 --seed 0  \
--lr 2e-4 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--epochs 120 \
--method A_modifiedLoss \