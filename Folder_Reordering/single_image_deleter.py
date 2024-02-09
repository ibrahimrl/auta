import os
from PIL import Image


def find_folders_with_single_image(folder_path):
    folders_with_single_image = []

    # Iterate through each subfolder in the given folder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # List all files in the subfolder
            files = os.listdir(subfolder_path)

            # Filter out only image files
            image_files = [file for file in files if file.endswith(
                ('.jpg', '.jpeg', '.png', '.gif'))]

            # Check if there's only one image file in the subfolder
            if len(image_files) == 1:
                folders_with_single_image.append(subfolder_path)

    return folders_with_single_image


def delete_folders(folders):
    for folder in folders:
        try:
            # Remove all files within the folder
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                os.remove(file_path)

            # Remove the folder itself
            os.rmdir(folder)
            print(f"Folder '{folder}' deleted successfully.")
        except OSError as e:
            print(f"Error: {folder} : {e.strerror}")


# Example usage
folder_path = "/content/drive/MyDrive/Project/JEM207_Project/Merged"
single_image_folders = find_folders_with_single_image(folder_path)
print("Folders containing only a single image:")
for folder in single_image_folders:
    print(folder)

delete_folders(single_image_folders)
