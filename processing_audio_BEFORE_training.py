"""
    This program extracts MFCCs from an audio dataset, normalizes the data, maps each speaker, 
    assigns labels to each speaker, and creates a JSON file containing the MFCCs and labels.

    Notes:
        - This code is based on an earlier project and includes enhancements:
            - Audio files are truncated or padded to ensure uniform length.
            - MFCCs are normalized for consistency.
    
    Requirements:
        - Python Version: 3.9.6
        - Librosa Version: 0.10.2

    To run this program, specify:
        - The dataset path in DATASET_PATH
        - The JSON output path in JSON_PATH 
"""

import json
import os
import librosa
import numpy as np

# Define paths and sample rate
DATASET_PATH = "path/to/dataset"
JSON_PATH = "path/to/output/json"
SAMPLE_RATE = 16000 

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, max_len=None):
    """
    Extracts MFCCs from an audio dataset and saves them to a JSON file along with speaker labels.

    Args:
        dataset_path (str): Path to the dataset.
        json_path (str): Path to save the JSON file with MFCCs and labels.
        num_mfcc (int): Number of MFCC coefficients to extract.
        n_fft (int): Number of samples in each FFT window.
        hop_length (int): Step size between consecutive FFT windows.
        max_len (int, optional): Maximum MFCC sequence length for padding/truncating.

    Returns:
        None
    """
    # Dictionary to store mappings, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    # Create a mapping dictionary for speaker labels
    label_map = {}
    current_label = 0

    # Traverse each sub-directory in the dataset path
    for dirpath, dirnames, filenames in os.walk(dataset_path):

        # Process each speaker directory
        if dirpath != dataset_path and not os.path.basename(dirpath).startswith('.'):
            # Assign a new label if the speaker is not already mapped
            semantic_label = os.path.basename(dirpath)
            if semantic_label not in label_map:
                label_map[semantic_label] = current_label
                data["mapping"].append(semantic_label)
                current_label += 1

            label = label_map[semantic_label]
            print("\nProcessing: {}, Label: {}".format(semantic_label, label))

            # Process each .wav audio file in the speaker directory
            valid_files = [f for f in filenames if f.endswith('.wav')]
            for f in valid_files:
                # Load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T

                # Apply padding or truncation to enforce uniform length
                if max_len:
                    if len(mfcc) > max_len:
                        mfcc = mfcc[:max_len]
                    else:
                        pad_width = max_len - len(mfcc)
                        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

                # Normalize MFCCs (zero mean, unit variance)
                mfcc_mean = np.mean(mfcc, axis=0)
                mfcc_std = np.std(mfcc, axis=0)
                mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-10)

                # Store the MFCC and label in the data dictionary
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(label)
                print("{}, MFCC shape: {}".format(file_path, mfcc.shape))

    # Save the MFCCs and labels to a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        print("\nData successfully saved to JSON file")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, max_len=1300)
