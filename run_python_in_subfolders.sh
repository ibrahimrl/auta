#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_path> <python_script>"
    exit 1
fi

folder_path=$1
python_script=$2

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder not found."
    exit 1
fi

# Loop through subfolders and run the Python script
for subfolder in "$folder_path"/*; do
    if [ -d "$subfolder" ]; then
        echo "Running $python_script for subfolder: $subfolder"
        python3 "$python_script" --weights yolov5s.pt --source "$subfolder"
    fi
done
