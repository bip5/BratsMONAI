#!/bin/bash

# Directory where all scripts and configs are originally located
BASE_DIR="/scratch/a.bip5/BraTS/scripts"

# Generate a unique directory for this job
JOB_DIR="/scratch/a.bip5/BraTS/jobs_eval/$(date +%B-%d-%s)"
mkdir -p ${JOB_DIR}

# Copy all scripts and config files to the job-specific directory
cp -r ${BASE_DIR}/* ${JOB_DIR}/

# Prepare the Slurm script, replacing placeholders with actual paths
SLURM_SCRIPT="${JOB_DIR}/shell/Eval_seg2.sh"
cp "/scratch/a.bip5/BraTS/scripts/shell/Eval_seg2.sh" ${SLURM_SCRIPT}
sed -i "s|__JOB_DIR__|${JOB_DIR}|g" ${SLURM_SCRIPT}  # Replace placeholders with actual paths

function check_pending_jobs {
    # Get the count of pending jobs submitted by the current user
    pending_jobs=$(squeue --noheader -t PD -u $USER | wc -l)
    echo $pending_jobs
}

function get_used_nodes {
    # Get a unique list of nodes used by the running jobs of the current user
    used_nodes=$(squeue --noheader -t R -o "%N" -u $USER | sort | uniq | tr '\n' ',')
    echo ${used_nodes%,}  # Remove the trailing comma
}

# Main loop to monitor jobs
while true; do
    pending_jobs=$(check_pending_jobs)
    
    if [[ "$pending_jobs" -eq "0" ]]; then
        # No pending jobs, collect nodes and prepare to submit new job
        nodes_to_exclude=$(get_used_nodes)
        echo "Submitting new job excluding nodes: $nodes_to_exclude"
        
        # Submit a new job excluding these nodes
        sbatch --exclude=$nodes_to_exclude ${SLURM_SCRIPT}
        break  # Exit the loop after submitting the job
    else
        echo "There are still $pending_jobs jobs pending. Checking again in 60 seconds..."
        sleep 300  # Wait for 60 seconds before checking again
    fi
done
# Submit the job
##sbatch "$@" ${SLURM_SCRIPT}