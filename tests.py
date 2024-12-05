import os
from utils import image_proc
from utils import analysis_tools
import tensorflow as tf
import numpy as np
from models.model import simulate_channel
from PIL import Image

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