import os
import shutil

def merge_folders_and_create_new(folder1_path, folder2_path, new_folder_name):
    # Get the parent directory of folder1 and folder2
    parent_directory = os.path.dirname(folder1_path)

    # Create a new folder with the given name in the parent directory
    new_folder_path = os.path.join(parent_directory, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    # Merge the contents of the two given folders into the new folder
    for folder_path in [folder1_path, folder2_path]:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                shutil.copy(file_path, new_folder_path)

    # Delete the given folders
    shutil.rmtree(folder1_path)
    shutil.rmtree(folder2_path)

    print(f"Folders '{folder1_path}' and '{folder2_path}' merged into '{new_folder_name}' and deleted successfully.")

# Example usage
folder1_path = "/content/drive/MyDrive/Project/JEM207_Project/Merged/vvv"
folder2_path = "/content/drive/MyDrive/Project/JEM207_Project/Merged/V"
new_folder_name = 'Vito'

merge_folders_and_create_new(folder1_path, folder2_path, new_folder_name)