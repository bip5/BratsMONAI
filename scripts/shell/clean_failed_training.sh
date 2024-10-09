#!/bin/bash

# Set the default path if not provided as a command-line argument
DIR_PATH=${1:-/scratch/a.bip5/BraTS/training_logs}

# Loop through all files in the specified directory
for file in "$DIR_PATH"/*; do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Get the last occurrence of the line containing "best mean dice"
        line=$(grep "best mean dice:" "$file" | tail -n 1)
        
        # Check if the line exists and extract the number after "best mean dice:"
        if [[ -n "$line" ]]; then
            # Extract the number after "best mean dice:"
            best_mean_dice=$(echo "$line" | sed -n 's/.*best mean dice: \([0-9.]*\).*/\1/p')

            # Check if the number is less than 0.2
            if (( $(echo "$best_mean_dice < 0.2" | bc -l) )); then
                # Delete the file if the condition is met
                echo "Deleting file: $file with best mean dice: $best_mean_dice"
                rm "$file"
            fi
        fi
    fi
done
