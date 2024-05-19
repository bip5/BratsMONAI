#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=segres_btA
#SBATCH --output=segres_bt.out.%j
#SBATCH --error=segres_bt.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./Barlow_augment_only.py --load_save 0 \
--load_barlow 1 \
--load_path ./2022-01-22T6best_metric_model.pth \
--save_name barl_Aonly.pth \
--batch_size 2 \
--barlow_flag 0 \
--bar_model_name checkpoint.pth \
--upsample DECONV 
