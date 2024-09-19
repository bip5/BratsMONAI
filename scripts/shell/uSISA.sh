#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=segres_bt
#SBATCH --output=SISA.out.%j
#SBATCH --error=SISA.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./sisa_1000.py --model UNet --load_save 0 \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--fold_num 5 \
--epochs 100 \