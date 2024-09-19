#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=seg_2
#SBATCH --output=seg_2.out.%j
#SBATCH --error=seg_2.err.%j
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS 2021/sisa_200.py' --model SegResNet --load_save 0 --load_path 2022-04-25SegResNetCV1ms1000rs1B_Final --fold_num 1 --seed 1 \
--lr 1e-4 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--epochs 120 \
--method Bredo \