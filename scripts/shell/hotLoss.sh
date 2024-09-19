#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=loss_ch
#SBATCH --output=loss_ch.out.%j
#SBATCH --error=loss_ch.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./betterLoss.py --model SegResNet --load_save 0 \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 0 \
--fold_num 1 \
--epochs 100 \
--seed 4  \
--method DiceLoss --size_factor 0 --dist_factor 0 \
--save_model 1 \
--comb 0 \

