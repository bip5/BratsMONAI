#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=nnunet
#SBATCH --output=%j.nnunet
#SBATCH --error=err.%j.n
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
####SBATCH --time=1-6:00:00


module load anaconda
source activate pix2pix

########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
# python -u nnu.py --batch_size 1 --train_num_workers 16 --val_num_workers 16
python -m torch.distributed.launch \
--nproc_per_node=4 \
--nnodes=1 --node_rank=0 \
--master_addr="localhost" --master_port=1234 \
 nnu.py --root_dir ./nnUNet_raw_data/ --batch_size 4 --multi_gpu True \
-train_num_workers 16 -val_num_workers 16 -interval 1 -num_samples 1 \
-task_id 001