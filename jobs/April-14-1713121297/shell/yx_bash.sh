#!/bin/bash

# Directory where all scripts and configs are originally located
BASE_DIR="/scratch/a.bip5/BraTS/scripts"

# Generate a unique directory for this job
JOB_DIR="/scratch/a.bip5/BraTS/jobs/$(date +%B-%d-%s)"
mkdir -p ${JOB_DIR}

# Copy all scripts and config files to the job-specific directory
cp -r ${BASE_DIR}/* ${JOB_DIR}/

# Prepare the Slurm script, replacing placeholders with actual paths
SLURM_SCRIPT="${JOB_DIR}/shell/yx2.sh"
cp "/scratch/a.bip5/BraTS/scripts/shell/yx2.sh" ${SLURM_SCRIPT}
sed -i "s|__JOB_DIR__|${JOB_DIR}|g" ${SLURM_SCRIPT}  # Replace placeholders with actual paths

# Submit the job
sbatch ${SLURM_SCRIPT}
