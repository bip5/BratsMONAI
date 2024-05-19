#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=segres_bt
#SBATCH --output=segres_bt.out.%j
#SBATCH --error=segres_bt.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./MONAI_start.py --load_save 0 \
--load_barlow 1 \
--load_path ./2022-02-04T9barl_3dimAllsampExtraAug.pth \
--save_name barl_3dimAllsampExtraAug.pth \
--batch_size 4 \
--barlow_final 1 \
--bar_model_name all3_dimsfin.pth \
--upsample DECONV \
--max_samples 20