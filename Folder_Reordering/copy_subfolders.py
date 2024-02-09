import os
import shutil


def copy_subfolders(source_folder):
    # Define the destination folder path
    merged_folder_path = os.path.join(os.path.dirname(source_folder), "Merged")

    # Create the merged folder if it doesn't exist
    if not os.path.exists(merged_folder_path):
        os.makedirs(merged_folder_path)

    # Walk through the source folder and copy subfolders
    for root, dirs, files in os.walk(source_folder):
        for directory in dirs:
            # Construct source and destination paths
            source_path = os.path.join(root, directory)

            # Iterate through the subfolders and copy them to the merged folder
            for subfolder in os.listdir(source_path):
                subfolder_path = os.path.join(source_path, subfolder)
                if os.path.isdir(subfolder_path):
                    destination_subfolder_path = os.path.join(
                        merged_folder_path, subfolder)
                    shutil.copytree(subfolder_path, destination_subfolder_path)
                    print(
                        f"Copied '{subfolder_path}' to '{destination_subfolder_path}'")


# Example usage:
source_folder_path = "/content/drive/MyDrive/Project/JEM207_Project/grouped"

copy_subfolders(source_folder_path)
