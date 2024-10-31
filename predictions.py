"""
    This script is used for making predictions with a pre-trained speaker identification model.
    
    Important:
        The input data for predictions must match the characteristics of the data the model was trained on, including:
            - Audio file:
                - Format: WAV
                - Channels: Stereo
                - Sample rate: 16kHz
            - Audio preprocessing parameters:
                - SAMPLE_RATE = 16000 
                - NUM_MFCC = 13
                - N_FFT = 2048
                - HOP_LENGTH = 512
                - MAX_LEN = 1300 

    Requirements:
        - Python Version: 3.9.6
        - Librosa Version: 0.10.2
        - TensorFlow Version: 2.16.2

    Specify the path to the model in the global variable "MODEL_PATH" and the path to the audio file to be predicted.
"""

import librosa
import numpy as np
import tensorflow.keras as keras

# Path to the trained model
MODEL_PATH = "path-to-model"
# Audio preprocessing parameters
SAMPLE_RATE = 16000 
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 1300 

def preprocess_audio(file_path):
    """
    Preprocesses the audio file to prepare it for prediction:
        - Extracts MFCCs
        - Applies padding if necessary
        - Normalizes the MFCCs
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        np.ndarray: Preprocessed MFCCs ready for prediction.
    """
    # Load audio file with specified sample rate
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T

    # Apply padding or truncation to maintain consistent length
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

    # Normalize MFCCs (zero mean and unit variance)
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-10) 
    mfcc = mfcc[np.newaxis, ...]

    return mfcc

def predict_speaker(model, file_path):
    """
    Predicts the speaker label for a given audio file.
    
    Args:
        model (keras.Model): The loaded model for prediction.
        file_path (str): Path to the audio file.
    
    Returns:
        int: The predicted label of the speaker.
    """
    # Preprocess the audio file
    mfcc = preprocess_audio(file_path)
    # Perform prediction with the model
    prediction = model.predict(mfcc)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label

if __name__ == "__main__":
    # Load the trained speaker identification model
    model = keras.models.load_model(MODEL_PATH)
    # Path to the audio file for prediction
    audio_file_path = "path-to-audio-file-for-prediction"
    # Predict the label and print the result
    label = predict_speaker(model, audio_file_path)
    print(f"Predicted label: {label}")
