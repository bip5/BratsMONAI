#!/bin/bash

# Check if a directory path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory-path>"
    exit 1
fi

# Assign the provided directory path to a variable
dir_path=$1

# Check if the provided path is a directory
if [ ! -d "$dir_path" ]; then
    echo "Error: Provided path is not a directory"
    exit 1
fi

# Loop through each subdirectory in the provided directory path
find "$dir_path" -mindepth 1 -maxdepth 1 -type d | while read subdir; do
    # Check for Excel and CSV files in the subdirectory
    if [ -z "$(find "$subdir" -maxdepth 1 -type f \( -name '*.xlsx' -o -name '*.csv' \))" ]; then
        echo "Deleting $subdir as it does not contain any Excel or CSV files"
        rm -rf "$subdir"
    else
        echo "$subdir contains Excel or CSV files and will not be deleted"
    fi
done

echo "Cleanup completed."
