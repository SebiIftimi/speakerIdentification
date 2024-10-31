"""
    This program creates, trains, and saves a speaker identification model.

    Python Version: 3.9.6
    TensorFlow Version: 2.16.2
    Scikit-learn Version: 1.5.1 
    Matplotlib Version: 3.9.2

    To run this program, specify the path to the dataset JSON file in "DATA_PATH".
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the path to the dataset JSON file
DATA_PATH = "path-to-dataset-json"

def load_data(data_path):
    """Loads the training dataset from a JSON file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])  # Extracts MFCC features
    y = np.array(data["labels"])  # Extracts labels
    return X, y

def prepare_datasets(test_size, validation_size):
    """
    Prepares train, validation, and test sets.

    Args:
        test_size (float): Proportion of data to allocate for testing.
        validation_size (float): Proportion of training data to allocate for validation.

    Returns:
        Tuple of numpy arrays: Training, validation, and testing data splits.
    """
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def plot_history(history):
    """
    Generates two plots: model accuracy over epochs and model loss over epochs.
    
    Args:
        history: Training history object from model fitting.
    """
    fig, axs = plt.subplots(2)

    # Plot accuracy
    axs[0].plot(history.history["accuracy"], label="train_accuracy")
    axs[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Model Accuracy")

    # Plot loss
    axs[1].plot(history.history["loss"], label="train_loss")
    axs[1].plot(history.history["val_loss"], label="val_loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Model Loss")

    plt.show()

def build_model(input_shape):
    """
    Builds and returns an LSTM-based model with batch normalization and dropout for speaker identification.

    - The model consists of three LSTM layers followed by dense layers.
    - BatchNormalization is applied to stabilize and speed up training.
    - Dropout is applied in the last two dense layers to reduce overfitting.
    - The output layer has 12 neurons, each representing a speaker category.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.Sequential: The built model.
    """
    model = keras.Sequential()

    model.add(keras.layers.LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.LSTM(128, return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(12, activation='softmax'))

    return model

if __name__ == "__main__":

    # Prepare training, validation, and test datasets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # Build the model using input shape derived from the training set
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # Compile the model with Adam optimizer and sparse categorical cross-entropy loss
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # Configure callbacks for learning rate reduction and early stopping
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     patience=5, 
                                                     factor=0.5, 
                                                     min_lr=1e-6)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=10, 
                                                   restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), 
                        batch_size=64, epochs=50, callbacks=[lr_reduction, early_stopping])

    # Evaluate model performance on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print("Test set accuracy: {:.2f}%".format(test_accuracy * 100))

    # Plot training history and save the trained model
    plot_history(history)
    model.save("speaker_identification_model.h5")
