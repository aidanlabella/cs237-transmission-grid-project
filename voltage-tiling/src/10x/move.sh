#!/bin/bash

# Function to move files up from subdirectories to parent directory
move_files_up() {
    local directory="$1"

    # Move files up for each subdirectory
    for subdir in "$directory"/*/; do
        echo "Moving files up from $subdir"
        mv "$subdir"/* "$directory"
        rm -r "$subdir"
    done
}

# Replace 'directory_path' with the path to the directory containing subdirectories
directory_path="/Users/ilyonsg/Documents/nrel/data/sfo/base_timeseries/"

move_files_up "$directory_path"