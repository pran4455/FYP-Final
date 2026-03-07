import os

def list_files_and_folders(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Print the current directory
        print(f"Folder: {dirpath}")
        # Print all files in the current directory
        for filename in filenames:
            print(f"  File: {os.path.join(dirpath, filename)}")
        # Print all subdirectories in the current directory
        for dirname in dirnames:
            print(f"  Folder: {os.path.join(dirpath, dirname)}")

# Use '.' to refer to the current directory (root where the script is placed)
list_files_and_folders('.')
