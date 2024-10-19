#!/bin/bash

# Directory where all scripts and configs are originally located
BASE_DIR="/scratch/a.bip5/BraTS/scripts"

# Generate a unique directory for this job
JOB_DIR="/scratch/a.bip5/BraTS/jobs/$(date +%B-%d-%s)"
mkdir -p ${JOB_DIR} || { echo "Failed to create job directory ${JOB_DIR}"; exit 1; }

# Copy all scripts and config files to the job-specific directory
cp -r ${BASE_DIR}/* ${JOB_DIR}/ || { echo "Failed to copy files to ${JOB_DIR}"; exit 1; }

# Prepare the Slurm script, replacing placeholders with actual paths
SLURM_SCRIPT="${JOB_DIR}/shell/yx2.sh"
cp "/scratch/a.bip5/BraTS/scripts/shell/yx2.sh" ${SLURM_SCRIPT}
sed -i "s|__JOB_DIR__|${JOB_DIR}|g" ${SLURM_SCRIPT}  # Replace placeholders with actual paths

for arg in "$@"; do
  case $arg in
    --note=*)
      NOTE_FOR_WANDB="${arg#*=}"
      shift
      ;;
    *)
      sbatch_args+=("$arg")
      ;;
  esac
done

# Submit the job and capture the Slurm job ID
##JOB_ID=$(sbatch --export=NOTE_FOR_WANDB="${NOTE_FOR_WANDB}" "$@" ${SLURM_SCRIPT} | awk '{print $4}')
# Submit the job and capture the Slurm job ID

sbatch_output=$(sbatch --export=NOTE_FOR_WANDB="${NOTE_FOR_WANDB}" "${sbatch_args[@]}" ${SLURM_SCRIPT})
echo "sbatch output:"
echo "$sbatch_output"
JOB_ID=$(echo "$sbatch_output" | awk '{print $4}')
echo "Extracted JOB_ID: $JOB_ID"

echo "Submitted job with ID: $JOB_ID"

# Commit the current codebase to Git and tag with the job ID
git add ${BASE_DIR}
git commit -m "Code snapshot for job $JOB_ID" || echo "No changes to commit."
git tag "$JOB_ID"

# Push the commit and tag to GitHub
git push origin master || { echo "Failed to push to GitHub"; exit 1; }
git push origin --tags || { echo "Failed to push tags to GitHub"; exit 1; }