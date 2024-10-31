import os
import shutil

def delete_subdirectories_without_wav(root_dir):
    """
    Deletes subdirectories within the specified root directory that do not contain any .wav files.
    
    Args:
        root_dir (str): The path to the root directory to search through.
    """
    # Walk through the directory tree from bottom to top
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Check if any .wav files are in the current directory
        contains_wav = any(file.endswith('.wav') for file in filenames)
        
        # Delete the directory if it doesn't contain .wav files and is not the root directory
        if not contains_wav and dirpath != root_dir:
            shutil.rmtree(dirpath)

# Specify paths to directories where empty subdirectories should be removed
root_directory = '/path/to/directory'
delete_subdirectories_without_wav(root_directory)

root_directory = 'path/to/dataset'
delete_subdirectories_without_wav(root_directory)
