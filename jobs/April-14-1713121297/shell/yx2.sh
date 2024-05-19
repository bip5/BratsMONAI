#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=resnet_a
#SBATCH --output=/scratch/a.bip5/BraTS/jobs/April-14-1713121297/resnet.%j.out
#SBATCH --error=/scratch/a.bip5/BraTS/jobs/April-14-1713121297/resnet.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --time=0-12:00:00

cd /scratch/a.bip5/BraTS/jobs/April-14-1713121297
export PYTHONPATH=/scratch/a.bip5/BraTS/jobs/April-14-1713121297:$PYTHONPATH

module load anaconda
source activate pix2pix
conda list >> result_$SLURM_JOB_ID.txt

python -u Training/training.py
