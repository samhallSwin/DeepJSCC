import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def main():
    metadata_file = "output_images/metadata.csv"
    image_size = (64, 64)
    batch_size = 32
    input_shape = image_size + (3,)
    preTrained = True
    epochs = 20
    snr_range = (-10, 20)

    snr_labls = ['SNR_0', 'SNR_5','SNR_10','SNR_-10','SNR_15']

    # Prepare datasets
    train_dir = "dataset/processedEurosat/train"
    test_dir = "dataset/processedEurosat/test"
    model_dir = 'models/channelState_models/'

    train_ds = prepare_snr_dataset(train_dir, image_size=image_size, batch_size=batch_size, snr_range=snr_range)
    test_ds = prepare_snr_dataset(test_dir, image_size=image_size, batch_size=batch_size, snr_range=snr_range)

    if preTrained == False:
        # Build and train the model
        model = build_snr_regression_model(input_shape=input_shape)
        model.fit(train_ds, validation_data=test_ds, epochs=epochs)

        # Save the trained model
        
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir,f"snr_estimation_model_reg_{epochs}.h5"))

        loss, mae = model.evaluate(test_ds)
        print(f"Test MAE: {mae}")
    else:
        model = tf.keras.models.load_model(f"{model_dir}snr_estimation_model_reg_20.h5")

        for lab in snr_labls:
            dir_path = f'dataset/processedEurosat/test/{lab}/'
            image_files = [file for file in os.listdir(dir_path) if file.endswith('.png')]
            if image_files:  # Ensure there are files in the directory
                # Select a random image
                image_path = os.path.join(dir_path, random.choice(image_files))
                estimated_snr = estimate_snr_single_image(model, image_path, image_size=(64, 64))
                print(f"Estimated SNR for the image {lab}: {estimated_snr:.2f} dB")
            else:
                print(f"No images found for label {lab}.")
                
        # Plot predicted vs. actual SNR values
        plot_predicted_vs_actual(model, test_ds, num_samples=100, snr_range=snr_range)

def prepare_snr_dataset(data_dir, image_size=(64, 64), batch_size=32, snr_range=(-10, 20)):
    """
    Prepare a dataset filtered by a specific SNR range.

    :param data_dir: Directory containing the dataset.
    :param image_size: Target size of images.
    :param batch_size: Batch size for training/testing.
    :param snr_range: Tuple specifying the range of SNR values to include.
    :return: A TensorFlow dataset filtered by the SNR range.
    """

    snr_min, snr_max = snr_range

    # Extract numeric labels from folder names
    def extract_label_from_path(file_path):
        snr_label = tf.strings.regex_replace(tf.strings.split(file_path, os.sep)[-2], "SNR_", "")
        return tf.strings.to_number(snr_label, out_type=tf.float32)

    # Preprocessing function with filtering by SNR range
    def preprocess_image_with_label(file_path):
        label = extract_label_from_path(file_path)
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size) / 255.0
        # Normalize SNR labels to range [0, 1]
        label = (label - snr_min) / (snr_max - snr_min) 
        return image, label

    # Filter function to include only SNR values within the range
    def filter_by_snr(file_path):
        label = extract_label_from_path(file_path)
        return tf.logical_and(label >= snr_range[0], label <= snr_range[1])

    # Load dataset from directory and apply filtering
    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, "*/*.png"), shuffle=True)
    dataset = dataset.filter(filter_by_snr)  # Apply the SNR range filter
    dataset = dataset.map(preprocess_image_with_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset



def build_snr_regression_model(input_shape=(64, 64, 3)):
    """
    Enhanced CNN model for SNR regression with Batch Normalization and Dropout.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(1)  # Single output for regression
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse', metrics=['mae'])
    
    model.summary()
    return model


def estimate_snr_single_image(model, image_path, image_size=(64, 64)):
    """
    Estimate the SNR of a single image using a trained model.
    
    :param model: Trained SNR regression model.
    :param image_path: Path to the image file to process.
    :param image_size: Expected size of the input image (height, width).
    :return: Predicted SNR value.
    """
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Decode the PNG image
    image = tf.image.resize(image, image_size)  # Resize to match model input size
    image = image / 255.0  # Normalize to [0, 1]
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the SNR
    predicted_snr = model.predict(image)
    
    return predicted_snr[0][0]  # Return the scalar SNR value

def plot_predicted_vs_actual(model, dataset, num_samples=100, snr_range=(-10, 20)):
    """
    Plot predicted vs. actual SNR values for a test dataset.
    :param model: Trained regression model.
    :param dataset: TensorFlow dataset containing test images and their labels.
    :param num_samples: Number of samples to visualize.
    """
    actual_snr = []
    predicted_snr = []

    # Collect predictions and labels
    for images, labels in dataset:
        predictions = model.predict(images)
        actual_snr.extend(rescale_snr(labels.numpy(), snr_range))
        predicted_snr.extend(rescale_snr(predictions.flatten(), snr_range))
        if len(actual_snr) >= num_samples:  # Limit to specified number of samples
            break

    actual_snr = np.array(actual_snr[:num_samples])
    predicted_snr = np.array(predicted_snr[:num_samples])

    output_path = "outputs/channel_state_est/channelStatePlot.png"

    # Plot results
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_snr, predicted_snr, alpha=0.7, label="Predictions")
    plt.plot([actual_snr.min(), actual_snr.max()], 
             [actual_snr.min(), actual_snr.max()], 
             'r--', label="Ideal Fit (y=x)")
    plt.xlabel("Actual SNR")
    plt.ylabel("Predicted SNR")
    plt.title("Predicted vs. Actual SNR")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")

def rescale_snr(predicted_snr, snr_range=(-10, 20)):
    snr_min, snr_max = snr_range
    return predicted_snr * (snr_max - snr_min) + snr_min

if __name__ == "__main__":
    main()