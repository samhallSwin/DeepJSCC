import time
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import contextlib

def evaluate_inference_time(model, input_shape, num_iterations=100):
    """
    Measure the average inference time for the model, suppressing terminal output.
    
    Args:
        model: Trained TensorFlow/Keras model.
        input_shape: Tuple indicating the shape of the input (excluding batch size).
        num_iterations: Number of iterations to average over.
        
    Returns:
        Average inference time in milliseconds.
    """
    # Create dummy input data
    dummy_input = np.random.random((1, *input_shape)).astype(np.float32)

    # Suppress output
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        # Warm-up iterations (to avoid initialization overhead)
        for _ in range(10):
            _ = model.predict(dummy_input)

    # Measure inference time without suppressing
    start_time = time.time()
    for _ in range(num_iterations):
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            _ = model.predict(dummy_input)
    end_time = time.time()

    # Calculate average time per inference
    total_time = end_time - start_time
    avg_time_per_inference = (total_time / num_iterations) * 1000  # Convert to ms
    
    return avg_time_per_inference


@tf.function
def model_inference(model, input_data):
    return model(input_data, training=False)

def evaluate_inference_time_tf(model, input_shape, num_iterations=100):
    """
    Measure the average inference time using TensorFlow graph execution.
    
    Args:
        model: Trained TensorFlow/Keras model.
        input_shape: Tuple indicating the shape of the input (excluding batch size).
        num_iterations: Number of iterations to average over.
        
    Returns:
        Average inference time in milliseconds.
    """
    # Create dummy input data
    dummy_input = tf.random.uniform((1, *input_shape))

    # Warm-up iterations
    for _ in range(10):
        _ = model_inference(model, dummy_input)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model_inference(model, dummy_input)
    end_time = time.time()

    # Calculate average time per inference
    total_time = end_time - start_time
    avg_time_per_inference = (total_time / num_iterations) * 1000  # Convert to ms
    
    return avg_time_per_inference

def latent_vis(model, test_ds):
    print("Extracting and visualizing latent space...")
    for images, _ in test_ds.take(1):  # Use one batch from the test dataset
        latent_features = model.get_latent_features(images)  # Extract latent features
        latent_features = tf.reshape(latent_features, (latent_features.shape[0], -1))  # Flatten for PCA/t-SNE

        # Apply t-SNE for dimensionality reduction
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(latent_features.numpy())

        # Visualize the reduced latent space
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=5, alpha=0.7)
        plt.title("Latent Space Visualization")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.savefig('example_images/latent_space.png')