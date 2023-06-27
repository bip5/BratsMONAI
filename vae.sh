#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai_dev
#SBATCH --job-name=seg_2
#SBATCH --output=vae.out.%j
#SBATCH --error=seg_2.err.%j
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix
export LOCAL_RANK=0

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
##python -u
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
 '/scratch/a.bip5/BraTS 2021/scripts/vae.py' --model SegResNetVAEx --load_save 0 --load_path 2022-04-25SegResNetCV1ms1000rs1B_Final --fold_num 1 --seed 1 \
--lr 5e-4 \
--batch_size 1 \
--max_samples 1000 \
--CV_flag 0 \
--epochs 30 \
--method segnet \
--workers 8 \