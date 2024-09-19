#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai_dev
#SBATCH --job-name=resnet_a
#SBATCH --output=__JOB_DIR__/resnet.%j.out
#SBATCH --error=__JOB_DIR__/resnet.err.%j
#SBATCH --output=/scratch/a.bip5/BraTS/training_logs/%x.%j
#SBATCH --error=/scratch/a.bip5/BraTS/training_logs/%x.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL


cd __JOB_DIR__
export PYTHONPATH=__JOB_DIR__:$PYTHONPATH

module load anaconda
source activate pix2pix
conda list >> result_$SLURM_JOB_ID.txt
echo __JOB_DIR__

##ln -s ${__JOB_DIR__} "/scratch/a.bip5/BraTS/jobs/${SLURM_JOB_ID}"

cd /scratch/a.bip5/BraTS/scripts/Evaluation

pytest -v eval_tests.py
