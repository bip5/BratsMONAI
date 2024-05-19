#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai_dev
#SBATCH --job-name=Eval
#SBATCH --output=/scratch/a.bip5/BraTS/eval_logs/Eval.%j
#SBATCH --error=/scratch/a.bip5/BraTS/eval_logs/Eval.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
###SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --export=ALL


module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/Evaluation/evaluation.py' \