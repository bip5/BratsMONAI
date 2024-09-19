#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai

#SBATCH --job-name=seg
#SBATCH --output=/scratch/a.bip5/BraTS/training_logs/%x.%j
#SBATCH --error=/scratch/a.bip5/BraTS/training_logs/%x.err.%j
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL


cd __JOB_DIR__
export PYTHONPATH=__JOB_DIR__:$PYTHONPATH

module load anaconda
source activate pix2pix
conda list >> __JOB_DIR__/result_$SLURM_JOB_ID.txt
echo __JOB_DIR__

##ln -s ${__JOB_DIR__} "/scratch/a.bip5/BraTS/jobs/${SLURM_JOB_ID}"

python -u Training/training.py
