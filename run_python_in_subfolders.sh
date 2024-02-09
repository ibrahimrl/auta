#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <folder_path> <python_script> <anonymization_flag>"
    exit 1
fi

folder_path=$1
python_script=$2
anonymization_flag=$3

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder not found."
    exit 1
fi

# Loop through subfolders and run the Python script
for subfolder in "$folder_path"/*; do
    if [ -d "$subfolder" ]; then
        echo "Running $python_script for subfolder: $subfolder"
        
        if [ "$anonymization_flag" == "True" ]; then
            python3 "$python_script" --weights yolov5s.pt --source "$subfolder" --anonymization
        else
            python3 "$python_script" --weights yolov5s.pt --source "$subfolder"
        fi
    fi
done
