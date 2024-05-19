#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=unet_a
#SBATCH --output=SISA_a.out.%j
#SBATCH --error=SISA_a.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS 2021/scripts/sisa.py' --model UNet --load_save 0 --load_path "saved models/UNetCV5ms1000rs4A" --seed 0 --fold_num 1 \
--lr 1e-2 \
--batch_size 1 \
--max_samples 11 \
--CV_flag 0 \
--epochs 3000 \
--method A_FS1 \
--T_max 3000 \