#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=unet_a
#SBATCH --output=SISA_a.out.%j
#SBATCH --error=SISA_a.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./sisa.py --model UNet --load_save 1 --load_name "saved models/UNetCV5ms1000rs4A" --seed 4 --fold_num 5 \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 1 \
--epochs 20 \
--method Aplus \