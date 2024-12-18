import os
from utils import image_proc
from utils import analysis_tools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.model import simulate_channel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image
import pandas as pd
import random

def validate_model(model, test_ds):
    print("Validating the model...")
    loss = model.evaluate(test_ds)
    print(loss)
    return True

def visualise_latent():
    #analysis_tools.latent_vis(model, test_ds)
    #attmaps.att_mapper(test_ds, model, config)

    #awgn_channel = AWGNChannel(snrdB=15)
    #train_ds, _ = prepare_dataset(config)
#
    #original_images, noisy_images, avg_psnr, avg_ssim = analysis_tools.run_awgn_on_images(train_ds, awgn_channel, config)

   # print(f"Average PSNR: {avg_psnr:.2f} dB")
   # print(f"Average SSIM: {avg_ssim:.4f}")
   # output_dir = "outputs/awgn_images"
   # os.makedirs(output_dir, exist_ok=True)
   # for idx, (orig, noisy) in enumerate(zip(original_images[0], noisy_images[0])):
   #     Image.fromarray(orig.astype(np.uint8)).save(f"{output_dir}/original_{idx}.png")
   #     Image.fromarray(noisy.astype(np.uint8)).save(f"{output_dir}/noisy_{idx}.png")
    return True

def save_latent(model, test_ds, config):
    print('Processing images and saving latent representations and original images to disk')

    output_dir = 'outputs/lantent_analysis/'
    os.makedirs(output_dir, exist_ok=True)
    
    images_count = 1
    for images, _ in test_ds.shuffle(1).take(1):  # Shuffle and take one batch
        num_images = min(images_count, images.shape[0])  # Ensure there are at least 8 images
        selected_images = images[:num_images]  # Take the first 8 images
        break  # Exit after processing one batch

    latent_path = f"{output_dir}latent_batch.npy"
    channel_output_path = f"{output_dir}channel_output_batch.npy"

    # Save the original images for comparison
    original_images_np = (selected_images.numpy() * 255).astype(np.uint8)
    for i, original_image_np in enumerate(original_images_np):
        original_img = Image.fromarray(original_image_np)
        original_img.save(f"{output_dir}original_image_{i}.png")

    # Save encoder's output for the selected images
    model.save_latent_representation(selected_images, latent_path)

    # Simulate the channel for the batch
    simulate_channel(latent_path, channel_output_path, channel=config.channel_type, snr_db=config.train_snrdB)

    # Decode the channel output for the batch
    output_images = model.load_and_decode(channel_output_path)

    # Save each reconstructed image
    output_images_np = (output_images.numpy() * 255).astype(np.uint8)
    for i, output_image_np in enumerate(output_images_np):
        output_img = Image.fromarray(output_image_np)
        output_img.save(f"{output_dir}reconstructed_image_{i}.png")

    print("Processing completed. Original images, latent representations, and reconstructed images saved.")

    latent = np.load(f"{output_dir}latent_batch.npy")
    channel_output = np.load(f"{output_dir}channel_output_batch.npy")

    mse = np.mean((latent - channel_output) ** 2)
    print(f"Mean Squared Error (MSE): {mse}")

    image_proc.create_plots_from_latent(latent, channel_output, output_dir)

def process_images_through_channel(model, test_ds, config, num_images=8, snr_range=(-10, 20)):
    """
    Pass random images directly through the channel layer at random SNR values.
    Visualize the original and channel-modified outputs.
    :param model: Trained model with a channel layer.
    :param test_ds: Test dataset.
    :param config: Config object containing settings.
    :param num_images: Number of images to process and visualize.
    :param snr_range: Tuple specifying the range of SNR values.
    """
    print('Processing images directly through the channel layer')

    output_dir = "outputs/channel_processed/"
    os.makedirs(output_dir, exist_ok=True)

    # Select random images from the dataset
    # Select random images from the dataset
    vis_images, _ = next(iter(test_ds))  # Take the first batch
    vis_images = vis_images[:num_images]  # Take the first `num_images` images from the batch

    # Generate a random SNR for each image
    snr_values = [random.uniform(*snr_range) for _ in range(num_images)]

    # Prepare for visualization
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))  # 2 columns: original and channel-modified
    for i, image in enumerate(vis_images):
        snr = snr_values[i]
        print(f"Processing image {i + 1} with SNR={snr:.2f}dB")

        # Adjust the SNR value in the model
        model.channel.set_snr(snr)

        # Prepare the image for the channel
        image_np = image.numpy()  # Original image in NumPy format
        flat_image = np.reshape(image_np, (-1))  # Flatten the image to a 1D vector
        symbols = np.stack(np.split(flat_image, 2), axis=-1)  # Split into I/Q components
        symbols_tensor = tf.convert_to_tensor([symbols], dtype=tf.float32)  # Add batch dimension

        # Pass the reshaped input through the channel
        channel_output = model.channel(symbols_tensor)  # Pass through channel
        channel_output_np = channel_output.numpy()[0]  # Remove batch dimension

        # Reconstruct the image shape from channel output
        reconstructed_flat = np.concatenate(np.split(channel_output_np, 2, axis=-1), axis=0).flatten()
        channel_image = np.reshape(reconstructed_flat, image_np.shape)  # Reshape to original image shape

        # Scale and convert for visualization
        original_image = (image_np * 255.0).astype(np.uint8)
        channel_image = (channel_image * 255.0).astype(np.uint8)

        # Save the channel-modified image
        save_path = os.path.join(output_dir, f"image_{i + 1}_snr_{snr:.2f}.png")
        plt.imsave(save_path, channel_image)

        # Visualize the results
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(channel_image)
        axes[i, 1].set_title(f"Channel Output\nSNR={snr:.2f}dB")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "channel_visualizations.png"))
    plt.show()

    print(f"Channel-processed images saved to {output_dir}")

def process_All_SNR(model, test_ds, train_ds, config, N=3):
    """
    Process each batch N times, assigning SNR values evenly across all integer values in the range.
    :param model: The trained model with a channel layer.
    :param test_ds: Test dataset.
    :param train_ds: Train dataset.
    :param config: Config object containing the SNR range and other settings.
    :param N: Number of times each batch is processed at different SNRs.
    """
    snr_range = config.snr_range  # e.g., (-10, 20)
    print('Processing images with balanced SNR values across batches')

    output_dir = "dataset/processedEurosat/"
    vis_output_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)

    datasets = {"train": train_ds, "test": test_ds}
    metadata = []

    # Generate a list of all integer SNR values in the range
    all_snrs = list(range(snr_range[0], snr_range[1] + 1))

    # Ensure we process N repeats for each batch
    for dataset_name, dataset in datasets.items():
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        for idx, (images, _) in enumerate(dataset):
            # Shuffle SNRs to avoid sequential structure, and take only N values for this batch
            random.shuffle(all_snrs)  # Shuffle the SNR list
            snr_values_for_batch = all_snrs[:N]  # Take the first N shuffled SNRs

            for repeat, snr in enumerate(snr_values_for_batch):
                print(f"  Processing batch {idx + 1}, repeat {repeat + 1} with SNR={snr}dB for {dataset_name} dataset...")

                # Adjust the SNR value in the model
                model.channel.set_snr(snr)

                snr_recon_dir = os.path.join(dataset_dir, f"SNR_{snr}")
                os.makedirs(snr_recon_dir, exist_ok=True)

                # Encode images to latent space
                latent = model.encoder(images)

                # Pass latents through the channel
                noisy_latent = model.channel(latent)

                # Decode latents to reconstruct images
                reconstructed_images = model.decoder(noisy_latent)

                # Save each reconstructed image
                for i in range(images.shape[0]):
                    recon_image = (reconstructed_images[i].numpy() * 255.0).astype(np.uint8)
                    recon_filename = f"{dataset_name}_recon_batch{idx + 1}_repeat{repeat + 1}_{i}.png"
                    recon_filepath = os.path.join(snr_recon_dir, recon_filename)
                    Image.fromarray(recon_image).save(recon_filepath)

                    # Append metadata
                    metadata.append({
                        "snr": snr,
                        "reconstructed_file": recon_filepath
                    })

                # Visualize the selected images for this SNR
                if idx == 0 and repeat == 0:  # Only visualize for the first batch and first repeat
                    vis_images, _ = next(iter(test_ds))  # Take a batch from test_ds
                    vis_images = vis_images[:8]  # Take the first 8 images
                    vis_latents = model.encoder(vis_images)
                    vis_noisy_latents = model.channel(vis_latents)
                    vis_reconstructed = model.decoder(vis_noisy_latents)

                    # Save and display visualizations for 8 images
                    fig, axes = plt.subplots(8, 3, figsize=(15, 20))  # 3 columns: original, reconstructed, metrics
                    for i in range(8):
                        original_image = (vis_images[i].numpy() * 255.0).astype(np.uint8)  # Original image
                        reconstructed_image = (vis_reconstructed[i].numpy() * 255.0).astype(np.uint8)  # Reconstructed image

                        # Compute PSNR metric
                        psnr_value = psnr(original_image, reconstructed_image, data_range=255)

                        # Display original image
                        axes[i, 0].imshow(original_image)
                        axes[i, 0].set_title("Original")
                        axes[i, 0].axis("off")

                        # Display reconstructed image
                        axes[i, 1].imshow(reconstructed_image)
                        axes[i, 1].set_title(f"Reconstructed\nSNR={snr}dB")
                        axes[i, 1].axis("off")

                        # Display metrics
                        axes[i, 2].text(0.1, 0.5, f"PSNR: {psnr_value:.2f}",
                                        fontsize=12, va='center', ha='left', wrap=True)
                        axes[i, 2].axis("off")

                    plt.tight_layout()
                    vis_filename = f"{dataset_name}_SNR_{snr}_visualization.png"
                    vis_filepath = os.path.join(vis_output_dir, vis_filename)
                    plt.savefig(vis_filepath)  # Save the visualization

    metadata_file = os.path.join(output_dir, "metadata.csv")
    pd.DataFrame(metadata).to_csv(metadata_file, index=False)
    print(f"Processing complete. Metadata saved to {metadata_file}. Visualizations saved to {vis_output_dir}.")


def process_random_image_at_snrs(model, test_ds, num_images=5, snr_range=(-20, 20), step=5, save_dir="outputs/channel_state_est/"):
    """
    Process multiple random images from the test set through the model at different SNRs
    and display results in a single plot for each image.
    :param model: The trained model with a channel layer.
    :param test_ds: Test dataset.
    :param num_images: Number of random images to process.
    :param snr_range: Tuple specifying the SNR range (min, max).
    :param step: Step size for SNR values.
    :param save_dir: Directory to save the visualizations.
    """
    print("Processing multiple random images at different SNR values...")
    os.makedirs(save_dir, exist_ok=True)

    # Generate SNR values from the specified range
    snr_values = list(range(snr_range[0], snr_range[1] + 1, step))

    # Extract one batch from the dataset
    test_images, _ = next(iter(test_ds))  # Get a batch of test images
    batch_size = test_images.shape[0]

    if num_images > batch_size:
        raise ValueError(f"num_images ({num_images}) exceeds the batch size ({batch_size}). Reduce num_images or increase the test batch size.")

    # Randomly select indices from the batch
    selected_indices = random.sample(range(batch_size), num_images)  # Randomly choose indices
    selected_images = tf.gather(test_images, selected_indices)  # Use tf.gather to extract the selected images

    for img_idx, image in enumerate(selected_images):
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        original_image_np = (image.numpy()[0] * 255.0).astype(np.uint8)  # Prepare for visualization

        # Prepare for visualization
        fig, axes = plt.subplots(1, len(snr_values) + 1, figsize=(20, 5))  # One row for this image
        axes[0].imshow(original_image_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Process the image at each SNR value
        for i, snr in enumerate(snr_values):
            print(f"Processing image {img_idx + 1} at SNR={snr}dB")

            # Set the SNR value in the model
            model.channel.set_snr(snr)

            # Encode, pass through channel, and decode
            latent = model.encoder(image)
            noisy_latent = model.channel(latent)
            reconstructed_image = model.decoder(noisy_latent)

            # Convert the reconstructed image to NumPy format for metrics and display
            reconstructed_image_np = (reconstructed_image.numpy()[0] * 255.0).astype(np.uint8)

            # Compute PSNR and SSIM metrics
            psnr_value = compute_psnr(original_image_np, reconstructed_image_np, data_range=255)
            ssim_value = compute_ssim(original_image_np, reconstructed_image_np, win_size=7, channel_axis=-1, data_range=255)

            # Display the reconstructed image
            axes[i + 1].imshow(reconstructed_image_np)
            axes[i + 1].set_title(f"SNR={snr}dB\nPSNR: {psnr_value:.2f}\n SSIM: {ssim_value:.3f}")
            axes[i + 1].axis("off")

        # Save the plot for this image
        save_path = os.path.join(save_dir, f"random_image_{img_idx + 1}_snr_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

        print(f"Visualization for image {img_idx + 1} saved to {save_path}")


def time_analysis(model):
    print('Running inference time analysis')
    input_shape = (64, 64, 3)  # Example input shape
    avg_time = analysis_tools.evaluate_inference_time(model, input_shape)
    print(f"Average Inference Time: {avg_time:.2f} ms")

    input_shape = (64, 64, 3)  # Example input shape
    avg_time_tf = analysis_tools.evaluate_inference_time_tf(model, input_shape)
    print(f"Average Inference Time (TF Graph): {avg_time_tf:.2f} ms")
    return True

def compare_to_JPEG2000(model, test_ds, train_ds, config):
    print('Comparing with JPEG2000 and generating example images...')
    output_dir = "outputs/example_images/JPEG2000"
    num_images = 8
    os.makedirs(output_dir, exist_ok=True)

    # Process and save images with BPG as the comparison format
    results = image_proc.process_and_save_images(
        test_ds, model, output_dir, config, config.bw_ratio, config.train_snrdB, config.mcs, 
        comparison_format="JPEG2000", num_images=num_images, LDPCon=config.LDPCon
    )

    (original_images, processed_images, bpg_ldpc_images,
     psnr_model_processed, psnr_bpg_ldpc,
     ssim_model_processed, ssim_bpg_ldpc) = results
    
    avg_psnr_model = np.mean(psnr_model_processed)
    avg_psnr_bpg = np.mean(psnr_bpg_ldpc)
    avg_ssim_model = np.mean(ssim_model_processed)
    avg_ssim_bpg = np.mean(ssim_bpg_ldpc)

    print(f"Average PSNR (Model Processed): {avg_psnr_model:.2f} dB")
    print(f"Average PSNR (JPEG2000): {avg_psnr_bpg:.2f} dB")
    print(f"Average SSIM (Model Processed): {avg_ssim_model:.4f}")
    print(f"Average SSIM (JPEG2000: {avg_ssim_bpg:.4f}")

    # Visualize and save example images
    output_path = "outputs/example_images/JPEG2000/visualization.png"
    image_proc.visualize_and_save_images_with_comparison(
    original_images=original_images, 
    processed_images=processed_images, 
    comparison_images=bpg_ldpc_images, 
    psnr_model_processed=psnr_model_processed, 
    psnr_comparison=psnr_bpg_ldpc, 
    ssim_model_processed=ssim_model_processed, 
    ssim_comparison=ssim_bpg_ldpc, 
    output_path=output_path, 
    comparison_type="JPEG2000",  
    num_images=num_images  
)

    print(f'8 random images captured to /{output_dir}')
    return True

def compare_to_BPG_LDPC(model, test_ds, train_ds, config):
    print("Comparing with BPG + LDPC and generating example images...")
    output_dir = "outputs/example_images/BPG_LDPC"
    num_images = 8
    os.makedirs(output_dir, exist_ok=True)

    # Process and save images with BPG as the comparison format
    results = image_proc.process_and_save_images(
        test_ds, model, output_dir, config, config.bw_ratio, config.train_snrdB, config.mcs, 
        comparison_format="BPG", num_images=num_images, LDPCon=config.LDPCon
    )

    (original_images, processed_images, bpg_ldpc_images,
     psnr_model_processed, psnr_bpg_ldpc,
     ssim_model_processed, ssim_bpg_ldpc) = results
    
    avg_psnr_model = np.mean(psnr_model_processed)
    avg_psnr_bpg = np.mean(psnr_bpg_ldpc)
    avg_ssim_model = np.mean(ssim_model_processed)
    avg_ssim_bpg = np.mean(ssim_bpg_ldpc)

    print(f"Average PSNR (Model Processed): {avg_psnr_model:.2f} dB")
    print(f"Average PSNR (BPG+LDPC): {avg_psnr_bpg:.2f} dB")
    print(f"Average SSIM (Model Processed): {avg_ssim_model:.4f}")
    print(f"Average SSIM (BPG+LDPC): {avg_ssim_bpg:.4f}")

    # Visualize and save example images
    output_path = "outputs/example_images/BPG_LDPC/visualization.png"
    image_proc.visualize_and_save_images_with_comparison(
    original_images=original_images, 
    processed_images=processed_images, 
    comparison_images=bpg_ldpc_images, 
    psnr_model_processed=psnr_model_processed, 
    psnr_comparison=psnr_bpg_ldpc, 
    ssim_model_processed=ssim_model_processed, 
    ssim_comparison=ssim_bpg_ldpc, 
    output_path=output_path, 
    comparison_type="BPG+LDPC",  
    num_images=num_images  
)

    print(f'8 random images captured to /{output_dir}')
    return True