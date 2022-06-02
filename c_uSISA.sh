#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=c_segres_bt
#SBATCH --output=uSISA_c.out.%j
#SBATCH --error=uSISA_c.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./sisa_1000.py --model UNet --load_save 1 --load_path UNetep100rs4C --seed 4  \
--lr 1e-3 \
--batch_size 4 \
--max_samples 1000 \
--CV_flag 0 \
--fold_num 0 \
--epochs 20 \
--method Cpnos \