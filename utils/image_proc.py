import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to save a single image
def save_image(image, filepath):
    img = Image.fromarray(image)
    img.save(filepath)

# Function to process images, save them, and store file sizes
def process_and_save_images(dataset, model, output_dir, num_images=8):
    os.makedirs(output_dir, exist_ok=True)

    original_images = []
    processed_images = []
    original_sizes = []
    processed_sizes = []
    images_saved = 0

    for batch in dataset:
        images, _ = batch  # Ignore labels
        images = images.numpy()  # Convert TensorFlow tensors to NumPy arrays

        for img in images:
            if images_saved < num_images:
                original_images.append(img)

                # Save the original image and record its file size
                original_filepath = os.path.join(output_dir, f"example_{images_saved + 1}.png")
                save_image((img * 255).astype(np.uint8), original_filepath)
                original_sizes.append(os.path.getsize(original_filepath))

                # Process the image through the model
                img_tensor = tf.expand_dims(img, axis=0)  # Add batch dimension
                processed_image = model(img_tensor)
                processed_image = tf.squeeze(processed_image).numpy()  # Remove batch dimension
                processed_images.append(processed_image)

                # Save the processed image and record its file size
                processed_filepath = os.path.join(output_dir, f"example_{images_saved + 1}_processed.png")
                save_image((processed_image * 255).astype(np.uint8), processed_filepath)
                processed_sizes.append(os.path.getsize(processed_filepath))

                images_saved += 1
            else:
                return original_images, processed_images, original_sizes, processed_sizes

    return original_images, processed_images, original_sizes, processed_sizes



import matplotlib.pyplot as plt

# Function to visualize images and print file size percentage change
def visualize_and_save_images(original_images, processed_images, original_sizes, processed_sizes, output_path, num_images=8):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))

    for i in range(num_images):
        # Calculate percentage change in file size
        size_change = ((processed_sizes[i] - original_sizes[i]) / original_sizes[i]) * 100

        # Display original image
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title(f"Original\nSize: {original_sizes[i]} bytes")
        axes[i, 0].axis("off")

        # Display processed image
        axes[i, 1].imshow(processed_images[i])
        axes[i, 1].set_title(f"Processed\nSize: {processed_sizes[i]} bytes\nChange: {size_change:.2f}%")
        axes[i, 1].axis("off")

        # Print percentage change
        print(f"Image {i+1}: Original Size = {original_sizes[i]} bytes, "
              f"Processed Size = {processed_sizes[i]} bytes, "
              f"Percentage Change = {size_change:.2f}%")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()



def denormalise_image(image):
    #denormalise and save example image to disk
    
    example_image = image * 255.0
    example_image = example_image.numpy().astype(np.uint8)  # Convert to numpy array
    img = Image.fromarray(example_image)  # Create an Image object
    filename = 'example_images/in_image.png' 
    img.save(filename)  # Save the image to disk
            
