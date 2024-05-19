#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=b_unet
#SBATCH --output=unet_b.out.%j
#SBATCH --error=unet_b.err.%j
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS 2021/scripts/sisa_200.py' --model UNet --load_save 1 --load_path 2022-04-28UNetCV5ms1000rs4B --fold_num 5 --seed 0 \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--epochs 300 \
--method B_300 \
--Tmax 20 \