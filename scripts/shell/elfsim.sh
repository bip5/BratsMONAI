#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=elfsim
#SBATCH --output=%j.elfsim
#SBATCH --error=err.%j.elfsim
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./ss_seg.py --model SegResNet --load_save 0 --load_path 2022-04-25SegResNetCV1ms1000rs1B_Final --fold_num 1 --seed 0 \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--epochs 100 \
--method seg2x \
--workers 16 \