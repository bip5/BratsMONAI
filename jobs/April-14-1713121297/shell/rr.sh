#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=rr_segres
#SBATCH --output=MoRR.out.%j
#SBATCH --error=MoRR.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS 2021/MoRR.py' --model UNet --load_save 0 \
--lr 1e-3 \
--batch_size 1 \
--max_samples 1000 \
--CV_flag 0 \
--fold_num 1 \
--bunch 10 \
--epochs 5 \
--flush 0 \