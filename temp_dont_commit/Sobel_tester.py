import tensorflow as tf
import numpy as np

def compute_edge_map(image):
    """
    Computes an edge map using the Sobel operator.

    Parameters:
        image (tf.Tensor): Input image tensor with shape (H, W, C) or (H, W).
                           Pixel values should be normalized to [0, 1] or [0, 255].

    Returns:
        tf.Tensor: Edge map with the same spatial dimensions as the input image.
    """
    # Ensure the image is in float32 format
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # If the image has multiple channels (e.g., RGB), convert it to grayscale
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)

    # Sobel filters
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)

    sobel_x = sobel_x[..., tf.newaxis, tf.newaxis]  # Shape: (3, 3, 1, 1)
    sobel_y = sobel_y[..., tf.newaxis, tf.newaxis]  # Shape: (3, 3, 1, 1)

    # Apply Sobel filters to the image
    grad_x = tf.nn.conv2d(image[tf.newaxis, ..., tf.newaxis], sobel_x, strides=[1, 1, 1, 1], padding="SAME")
    grad_y = tf.nn.conv2d(image[tf.newaxis, ..., tf.newaxis], sobel_y, strides=[1, 1, 1, 1], padding="SAME")

    # Compute gradient magnitude
    grad_magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))

    # Squeeze to remove batch and channel dimensions
    edge_map = tf.squeeze(grad_magnitude)

    # Normalize edge map to range [0, 1]
    edge_map = edge_map / tf.reduce_max(edge_map)

    return edge_map

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example image: Load a sample grayscale or RGB image as a NumPy array
    sample_image = np.random.rand(256, 256, 3).astype(np.float32)  # Replace with an actual image

    # Compute edge map
    edge_map = compute_edge_map(sample_image)

    # Plot the original image and the edge map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(sample_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Edge Map")
    plt.imshow(edge_map, cmap="gray")
    plt.axis("off")

    plt.show()