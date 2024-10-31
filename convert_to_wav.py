"""
    This script converts .flac files in the LibriSpeech dataset from FLAC to WAV format
    and deletes the original FLAC files afterward.
    
    To run this program, the "pydub" library must be installed, and the path to the
    LibriSpeech dataset should be specified in "input_dir".
"""

from pydub import AudioSegment
import os

# Specify the directory containing the LibriSpeech dataset
input_dir = "/path-to-LibriSpeech"

# Traverse the directory structure to find all .flac files
for root, dirs, files in os.walk(input_dir):
    for file in files:
        # Process only files with a .flac extension
        if file.endswith(".flac"):
            flac_path = os.path.join(root, file)  # Original FLAC file path
            wav_path = os.path.splitext(flac_path)[0] + ".wav"  # Destination WAV file path
            
            # Convert FLAC to WAV and save to the specified path
            print(f"Converting {flac_path} to {wav_path}")
            audio = AudioSegment.from_file(flac_path, format="flac")
            audio.export(wav_path, format="wav")
            
            # Delete the original FLAC file after conversion
            os.remove(flac_path)
            print(f"Removed original file {flac_path}")
