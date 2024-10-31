"""
    This script reorganizes the LibriSpeech dataset structure to facilitate training a speaker identification model.
    
    " The corpus is split into several parts to enable users to selectively download
        subsets of it, according to their needs. The subsets with "clean" in their name
        are supposedly "cleaner"(at least on average), than the rest of the audio and
        US English accented. That classification was obtained using very crude automated 
        means, and should not be considered completely reliable. The subsets are
        disjoint, i.e. the audio of each speaker is assigned to exactly one subset. "

        <corpus root>
    |
    .- README.TXT
    |
    .- READERS.TXT
    |
    .- CHAPTERS.TXT
    |
    .- BOOKS.TXT
    |
    .- train-clean-100/
                   |
                   .- 19/
                       |
                       .- 198/
                       |    |
                       |    .- 19-198.trans.txt
                       |    |    
                       |    .- 19-198-0001.flac
                       |    |
                       |    .- 14-208-0002.flac
                       |    |
                       |    ...
                       |
                       .- 227/
                            | ...

        , where 19 is the ID of the reader, and 198 and 227 are the IDs of the chapters
        read by this speaker. The *.trans.txt files contain the transcripts for each
        of the utterances, derived from the respective chapter and the FLAC files contain
        the audio itself.
    
    The script consolidates all audio files for each speaker into a single folder under their ID,
    removing the original subfolders where the files were stored.

    Note:
        - LibriSpeech audio files are in FLAC format.
        - To use this script, specify the path to the LibriSpeech dataset in "dir_path".
"""

import os
import shutil

def reorganize(root_dir):
    """
    Consolidates all audio files for each speaker into their primary directory and removes subdirectories.
    
    Args:
        root_dir (str): Path to the root directory of the LibriSpeech dataset.
    """
    # Iterate over each speaker directory
    for speaker_id in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_id)
        
        if os.path.isdir(speaker_path):
            # Process each chapter directory within the speaker's directory
            for chapter_id in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_id)
                
                if os.path.isdir(chapter_path):
                    # Move each .flac audio file to the main speaker directory
                    for file_name in os.listdir(chapter_path):
                        file_path = os.path.join(chapter_path, file_name)
                        if file_name.endswith('.flac'):  # Adjust the file extension if dataset format differs
                            shutil.move(file_path, speaker_path)
                    
                    # Remove the empty chapter directory after moving files
                    if not os.listdir(chapter_path):
                        os.rmdir(chapter_path)

# Specify the path to the LibriSpeech dataset
dir_path = '/path-to-LibriSpeech-dataset'
reorganize(dir_path)
print("Reorganization complete.")
