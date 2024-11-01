#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=build_seg_image
#SBATCH --output=/scratch/a.bip5/BraTS/build_logs/%x.%j.out
#SBATCH --error=/scratch/a.bip5/BraTS/build_logs/%x.err.%j
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL

# Ensure the build_logs directory exists
mkdir -p /scratch/a.bip5/BraTS/build_logs

# Load the Singularity module
module load singularity

# Define paths
TARBALL_PATH="/scratch/a.bip5/BraTS/singularity/threshold_model.tar.gz"
OUTPUT_IMAGE="/scratch/a.bip5/BraTS/threshold_model.sif"

# Unpack the tarball if necessary (in case it is compressed)
tar -xzf $TARBALL_PATH -C /scratch/a.bip5/BraTS/singularity

# Build the Singularity image from the Docker tarball
singularity build $OUTPUT_IMAGE docker-archive://scratch/a.bip5/BraTS/singularity/threshold_model_tar
