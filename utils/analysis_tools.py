import time
import numpy as np
import tensorflow as tf
import os
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