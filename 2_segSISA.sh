#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=seg_2
#SBATCH --output=seg_2.out.%j
#SBATCH --error=seg_2.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./sisa_200.py --model SegResNet --load_save 0 \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--fold_num 1 \
--epochs 10 \
--seed 1 \
--method B \