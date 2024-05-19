#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=xrun
#SBATCH --output=%j.xrun
#SBATCH --error=err.%j.xrun
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

export OMP_NUM_THREADS=4

df -h /dev/shm
ipcs -lm

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
--master_addr="localhost" --master_port=1255 '/scratch/a.bip5/BraTS/scripts/Training/training.py'	
