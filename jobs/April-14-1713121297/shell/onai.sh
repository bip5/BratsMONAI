#!/bin/bash --login
#$ -cwd
#SBATCH --account=scw1767
#SBATCH --job-name=seg_monai
#SBATCH --out=%J
#SBATCH --err=%J
#SBATCH --mem-per-cpu=30G
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH -p gpu
#SBATCH --gres=gpu:2

module load anaconda
# echo "before calling source: $PATH"
source activate pix2pix
# echo "after calling source: $PATH"

###source activate /scratch/a.bip5/brats2021  #SBATCH -n 10


python ./printTest.py

###python -u ./MONAI_start.py