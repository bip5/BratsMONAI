#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=Eval_ensemble
#SBATCH --output=%j.Eval
#SBATCH --error=err.%j.Eval
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u ./monai_eval.py --model SegResNet --load_save 0 \
--load_name UNetCV3ms1000rs2A_anom \
--batch_size 4 \
--upsample DECONV \
--max_samples 200 \
--ensemble 1 \
--avgmodel 0 \
--plot 1 \
--val 0 \

