#NOTE: This file is a mess and half of it is a graveyard for old functions that are probably broken. One day I'll clean it up.

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .analysis_tools import run_single_image_BPGplusLDPC
from .analysis_tools import run_single_image_JPEG2000
from .analysis_tools import calculate_psnr
from .analysis_tools import calculate_ssim
from skimage.metrics import structural_similarity as ssim

# Function to save a single image
def save_image(image, filepath):
    img = Image.fromarray(image)
    img.save(filepath)

def save_image(image, filepath):
    """
    Save a NumPy array as an image file.

    Args:
        image: NumPy array of the image.
        filepath: Path to save the image.
    """
    # Ensure image is in the range [0, 1] and convert to uint8
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1)  # Clip values to [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255]

    # Convert grayscale images to 2D
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=-1)

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


def process_and_save_images(dataset, model, output_dir, config, bw_ratio, snrs, mcs, comparison_format="BPG", num_images=8, LDPCon=True):
    """
    Process images through a model and the specified comparison pipeline (BPG or JPEG2000), save them, and compute PSNR and SSIM.

    Args:
        dataset: TensorFlow dataset containing the images to process.
        model: Trained TensorFlow/Keras model to process the images.
        output_dir: Directory to save the images.
        config: Configuration object containing image properties and parameters.
        bw_ratio: Bandwidth ratio for the compression pipeline.
        snrs: Signal-to-noise ratio for the compression pipeline.
        mcs: Modulation and coding scheme for the compression pipeline.
        comparison_format: "BPG" or "JPEG2000", specifies the compression format for comparison.
        num_images: Number of images to process.
        LDPCon: Flag to enable or disable LDPC in the compression pipeline.

    Returns:
        Tuple containing lists of original images, model-processed images, and compression pipeline processed images,
        along with their PSNR and SSIM scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_images = []
    processed_images = []
    comparison_images = []

    psnr_model_processed = []
    psnr_comparison = []
    ssim_model_processed = []
    ssim_comparison = []

    images_saved = 0
    max_bytes = 0

    for batch in dataset:
        images, _ = batch  # Ignore labels
        images = images.numpy()  # Convert TensorFlow tensors to NumPy arrays

        for img in images:
            if images_saved < num_images:
                original_images.append(img)

                # Save the original image
                original_filepath = os.path.join(output_dir, f"example_{images_saved + 1}.png")
                save_image((img * 255).astype(np.uint8), original_filepath)

                # Process the image through the model
                img_tensor = tf.expand_dims(img, axis=0)  # Add batch dimension
                processed_image = model(img_tensor)
                processed_image = tf.squeeze(processed_image).numpy()  # Remove batch dimension
                processed_images.append(processed_image)

                # Save the model-processed image
                processed_filepath = os.path.join(output_dir, f"example_{images_saved + 1}_processed.png")
                save_image((processed_image * 255).astype(np.uint8), processed_filepath)

                # Process the image through the specified compression pipeline
                if comparison_format == "BPG":
                    comparison_image, max_bytes = run_single_image_BPGplusLDPC(
                        (img * 255).astype(np.uint8), config, bw_ratio, snrs, mcs, save_path=output_dir, LDPCon=LDPCon
                    )
                elif comparison_format == "JPEG2000":
                    comparison_image = run_single_image_JPEG2000(
                        (img * 255).astype(np.uint8), config, save_path=output_dir
                    )
                else:
                    raise ValueError("Invalid comparison_format. Must be 'BPG' or 'JPEG2000'.")

                comparison_images.append(comparison_image)

                # Save the comparison pipeline processed image
                comparison_filepath = os.path.join(output_dir, f"example_{images_saved + 1}_{comparison_format.lower()}.png")
                save_image(comparison_image, comparison_filepath)

                # Compute PSNR for model-processed images
                psnr_model = calculate_psnr((img * 255).astype(np.uint8), (processed_image * 255).astype(np.uint8))
                psnr_model_processed.append(psnr_model)

                # Compute PSNR for comparison pipeline processed images
                psnr_comp = calculate_psnr((img * 255).astype(np.uint8), comparison_image)
                psnr_comparison.append(psnr_comp)

                # Compute SSIM for model-processed images
                ssim_model = calculate_ssim((img * 255).astype(np.uint8), (processed_image * 255).astype(np.uint8))
                ssim_model_processed.append(ssim_model)

                # Compute SSIM for comparison pipeline processed images
                ssim_comp = calculate_ssim((img * 255).astype(np.uint8), comparison_image)
                ssim_comparison.append(ssim_comp)

                images_saved += 1
            else:
                print(f'Max bytes = {max_bytes}')
                return (original_images, processed_images, comparison_images,
                        psnr_model_processed, psnr_comparison,
                        ssim_model_processed, ssim_comparison)

    return (original_images, processed_images, comparison_images,
            psnr_model_processed, psnr_comparison,
            ssim_model_processed, ssim_comparison)

#Old version for reference
def process_and_save_images_with_bpg(dataset, model, output_dir, config, bw_ratio, snrs, mcs, num_images=8, LDPCon = True):
    """
    Process images through a model and the BPG+LDPC pipeline, save them, and store their file sizes.

    Args:
        dataset: TensorFlow dataset containing the images to process.
        model: Trained TensorFlow/Keras model to process the images.
        output_dir: Directory to save the images.
        config: Configuration object containing image properties and parameters.
        bw_ratio: Bandwidth ratio for the BPG+LDPC pipeline.
        snrs: Signal-to-noise ratio for the BPG+LDPC pipeline.
        mcs: Modulation and coding scheme for the BPG+LDPC pipeline.
        num_images: Number of images to process.

    Returns:
        Tuple containing lists of original images, model-processed images, 
        BPG+LDPC processed images, and their corresponding file sizes.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_images = []
    processed_images = []
    bpg_ldpc_images = []
    original_sizes = []
    processed_sizes = []
    bpg_ldpc_sizes = []
    images_saved = 0

    psnr_model_processed = []
    psnr_bpg_ldpc = []
    ssim_model_processed = []
    ssim_bpg_ldpc = []

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

                # Save the model-processed image and record its file size
                processed_filepath = os.path.join(output_dir, f"example_{images_saved + 1}_processed.png")
                save_image((processed_image * 255).astype(np.uint8), processed_filepath)
                processed_sizes.append(os.path.getsize(processed_filepath))

                # Process the image through the BPG+LDPC pipeline
                bpg_ldpc_image, max_bytes = run_single_image_BPGplusLDPC(
                    (img * 255).astype(np.uint8), config, bw_ratio, snrs, mcs, save_path=output_dir, LDPCon = LDPCon
                )
                bpg_ldpc_images.append(bpg_ldpc_image)

                # Save the BPG+LDPC processed image and record its file size
                bpg_ldpc_filepath = os.path.join(output_dir, f"example_{images_saved + 1}_bpg_ldpc.png")
                save_image(bpg_ldpc_image, bpg_ldpc_filepath)
                bpg_ldpc_sizes.append(os.path.getsize(bpg_ldpc_filepath))

                # Compute PSNR for model-processed images
                psnr_model = calculate_psnr((img * 255).astype(np.uint8), (processed_image * 255).astype(np.uint8))
                psnr_model_processed.append(psnr_model)

                # Compute PSNR for BPG+LDPC-processed images
                psnr_bpg = calculate_psnr((img * 255).astype(np.uint8), bpg_ldpc_image)
                psnr_bpg_ldpc.append(psnr_bpg)

                # Compute SSIM for model-processed images
                ssim_model = calculate_ssim((img * 255).astype(np.uint8), (processed_image * 255).astype(np.uint8))
                ssim_model_processed.append(ssim_model)

                # Compute SSIM for BPG+LDPC-processed images
                ssim_bpg = calculate_ssim((img * 255).astype(np.uint8), bpg_ldpc_image)
                ssim_bpg_ldpc.append(ssim_bpg)

                images_saved += 1
            else:
                print(f'Max bytes = {max_bytes}')
                return (original_images, processed_images, bpg_ldpc_images,
                        original_sizes, processed_sizes, bpg_ldpc_sizes,
                        psnr_model_processed, psnr_bpg_ldpc,
                        ssim_model_processed, ssim_bpg_ldpc)

    return (original_images, processed_images, bpg_ldpc_images,
            original_sizes, processed_sizes, bpg_ldpc_sizes,
            psnr_model_processed, psnr_bpg_ldpc,
            ssim_model_processed, ssim_bpg_ldpc)

def visualize_and_save_images_with_bpg(
    original_images,
    processed_images,
    bpg_ldpc_images,
    psnr_model_processed, 
    psnr_bpg_ldpc,
    ssim_model_processed, 
    ssim_bpg_ldpc, 
    output_path,
    num_images=8,
):
    """
    Visualize and save images processed through a model and the BPG+LDPC pipeline.

    Args:
        original_images: List of original images as numpy arrays.
        processed_images: List of model-processed images as numpy arrays.
        bpg_ldpc_images: List of images processed through BPG+LDPC as numpy arrays.
        original_sizes: List of file sizes for original images.
        processed_sizes: List of file sizes for model-processed images.
        bpg_ldpc_sizes: List of file sizes for BPG+LDPC processed images.
        output_path: File path to save the visualization.
        num_images: Number of images to visualize.
    """
    fig, axes = plt.subplots(num_images, 3, figsize=(5, num_images * 2))

    for i in range(num_images):

        # Display original image
        axes[i, 0].imshow(original_images[i])
        if i==0:
            axes[i, 0].set_title("Original image")
        axes[i, 0].axis("off")

        # Display model-processed image
        axes[i, 1].imshow(processed_images[i])
        if i==0:
            axes[i, 1].set_title(f"Proposed technique\nPSNR = {psnr_model_processed[i]:.2f}\nSSIM = {ssim_model_processed[i]:.4f}")
        else:
            axes[i, 1].set_title(f"PSNR = {psnr_model_processed[i]:.2f}\nSSIM = {ssim_model_processed[i]:.4f}")
        axes[i, 1].axis("off")

        # Display BPG+LDPC processed image
        axes[i, 2].imshow(bpg_ldpc_images[i])
        if i==0:
            axes[i, 2].set_title(f"BPG+LDPC\nPSNR = {psnr_bpg_ldpc[i]:.2f}\nSSIM = {ssim_bpg_ldpc[i]:.4f}")
        else:
            axes[i, 2].set_title(f"PSNR = {psnr_bpg_ldpc[i]:.2f}\nSSIM = {ssim_bpg_ldpc[i]:.4f}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")

def visualize_and_save_images_with_comparison(
    original_images,
    processed_images,
    comparison_images,
    psnr_model_processed, 
    psnr_comparison,
    ssim_model_processed, 
    ssim_comparison, 
    output_path,
    comparison_type="BPG+LDPC",
    num_images=8,
):
    """
    Visualize and save images processed through a model and a comparison pipeline (e.g., BPG+LDPC or JPEG2000).

    Args:
        original_images: List of original images as numpy arrays.
        processed_images: List of model-processed images as numpy arrays.
        comparison_images: List of images processed through the comparison pipeline as numpy arrays.
        psnr_model_processed: List of PSNR values for model-processed images.
        psnr_comparison: List of PSNR values for comparison pipeline images.
        ssim_model_processed: List of SSIM values for model-processed images.
        ssim_comparison: List of SSIM values for comparison pipeline images.
        output_path: File path to save the visualization.
        comparison_type: String indicating the type of comparison (e.g., "BPG+LDPC" or "JPEG2000").
        num_images: Number of images to visualize.
    """
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

    for i in range(num_images):

        # Display original image
        axes[i, 0].imshow(original_images[i])
        if i == 0:
            axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        # Display model-processed image
        axes[i, 1].imshow(processed_images[i])
        if i == 0:
            axes[i, 1].set_title(f"Proposed Technique\nPSNR = {psnr_model_processed[i]:.2f}\nSSIM = {ssim_model_processed[i]:.4f}")
        else:
            axes[i, 1].set_title(f"PSNR = {psnr_model_processed[i]:.2f}\nSSIM = {ssim_model_processed[i]:.4f}")
        axes[i, 1].axis("off")

        # Display comparison processed image
        axes[i, 2].imshow(comparison_images[i])
        if i == 0:
            axes[i, 2].set_title(f"{comparison_type}\nPSNR = {psnr_comparison[i]:.2f}\nSSIM = {ssim_comparison[i]:.4f}")
        else:
            axes[i, 2].set_title(f"PSNR = {psnr_comparison[i]:.2f}\nSSIM = {ssim_comparison[i]:.4f}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

# Function to visualize images and print file size percentage change
def visualize_and_save_images(original_images, processed_images, processed_sizes, original_sizes, output_path, num_images=8):
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

def visualize_and_save_images_with_comparison_and_clip(
    original_images,
    processed_images,
    comparison_images,
    psnr_model_processed, 
    psnr_comparison,
    ssim_model_processed, 
    ssim_comparison, 
    clip_model_sims,           # 1D array-like, per image
    clip_comparison_sims,      # 1D array-like, per image
    output_path,
    comparison_type="BPG+LDPC",
    num_images=8,
):
    """
    Visualize and save triplets (Original, Model, Comparison) and print PSNR/SSIM/CLIP on the two right columns.

    All images can be float in [0,1] or uint8; we display them directly.
    """
    # Safety: pick N that’s actually available
    N = min(
        num_images,
        len(original_images),
        len(processed_images),
        len(comparison_images),
        len(psnr_model_processed),
        len(psnr_comparison),
        len(ssim_model_processed),
        len(ssim_comparison),
        len(clip_model_sims),
        len(clip_comparison_sims),
    )
    if N == 0:
        raise RuntimeError("No images available for visualization.")

    fig, axes = plt.subplots(N, 3, figsize=(10, N * 3))
    if N == 1:
        axes = np.expand_dims(axes, axis=0)  # make it indexable by [i, j]

    for i in range(N):
        # Col 0: original
        axes[i, 0].imshow(original_images[i])
        if i == 0:
            axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Col 1: model
        axes[i, 1].imshow(processed_images[i])
        title_model = f"PSNR={psnr_model_processed[i]:.2f}\nSSIM={ssim_model_processed[i]:.4f}\nCLIP={clip_model_sims[i]:.4f}"
        if i == 0:
            axes[i, 1].set_title(f"Proposed\n{title_model}")
        else:
            axes[i, 1].set_title(title_model)
        axes[i, 1].axis("off")

        # Col 2: comparison (e.g., BPG+LDPC)
        axes[i, 2].imshow(comparison_images[i])
        title_comp = f"PSNR={psnr_comparison[i]:.2f}\nSSIM={ssim_comparison[i]:.4f}\nCLIP={clip_comparison_sims[i]:.4f}"
        if i == 0:
            axes[i, 2].set_title(f"{comparison_type}\n{title_comp}")
        else:
            axes[i, 2].set_title(title_comp)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def visualize_strip_with_titles_no_scores(
    original_images,
    processed_images,
    comparison_images,
    output_path,
    row_labels=("Original image", "Proposed Technique", "BPG + LDPC"),
    num_images=8,
    dpi=200,
):
    import numpy as np, matplotlib.pyplot as plt

    N = min(num_images, len(original_images), len(processed_images), len(comparison_images))
    if N == 0:
        raise RuntimeError("No images available for visualization.")

    fig_w = max(6.0, 1.2 * N)
    fig_h = 3.8
    fig, axes = plt.subplots(3, N, figsize=(fig_w, fig_h), dpi=dpi)
    if N == 1:
        axes = np.expand_dims(axes, axis=1)

    # Make room on the left for labels; pack tiles tightly
    plt.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.03, wspace=0.02, hspace=0.02)

    def _show(ax, img):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        # hide spines but keep labels alive
        for s in ax.spines.values():
            s.set_visible(False)

    for j in range(N):
        _show(axes[0, j], original_images[j])
        _show(axes[1, j], processed_images[j])
        _show(axes[2, j], comparison_images[j])

    # Row labels on the first column only
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(
            label, rotation=0, labelpad=45, ha="right", va="center",
            fontsize=12, fontweight="bold"
        )

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_latex_metrics_table_with_clip(
    filepath,
    psnr_model, ssim_model, clip_model,
    psnr_comp,  ssim_comp,  clip_comp,
    comparison_title="BPG + LDPC",
    proposed_title="Proposed technique",
    col_width="1cm",
    num_images=None,
):
    """
    Write a LaTeX tabular with rows for PSNR/SSIM/CLIP for both methods.
    Values are formatted like your example. Column count matches N images.
    """
    # Determine N and slice consistently
    arrays = [psnr_model, ssim_model, clip_model, psnr_comp, ssim_comp, clip_comp]
    N = min(len(a) for a in arrays)
    if num_images is not None:
        N = min(N, num_images)
    if N == 0:
        raise RuntimeError("No metrics to export.")
    psnr_model, ssim_model, clip_model = psnr_model[:N], ssim_model[:N], clip_model[:N]
    psnr_comp,  ssim_comp,  clip_comp  = psnr_comp[:N],  ssim_comp[:N],  clip_comp[:N]

    # Column spec: first column is 'c', then N times p{<width>}
    colspec = "|c|" + "|".join([f"p{{{col_width}}}"] * N) + "|"

    def fmt_row(name, vals, fmt=".2f"):
        if name.upper() == "SSIM" or name.upper() == "CLIP":
            fmt = ".4f"
        vals_str = " & ".join([f"{v:{fmt}}" for v in vals])
        return f"                 {name} & {vals_str}\\\\"

    lines = []
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("        \\hline")
    lines.append(f"                \\multicolumn{{{N+1}}}{{|l|}}{{\\textbf{{{proposed_title} }}}}\\\\")
    lines.append("                 \\hline")
    lines.append(fmt_row("PSNR", psnr_model, ".2f"))
    lines.append("                 \\hline")
    lines.append(fmt_row("SSIM", ssim_model, ".4f"))
    lines.append("                 \\hline")
    lines.append(fmt_row("CLIP", clip_model, ".4f"))
    lines.append("                 \\hline")
    lines.append(f"                \\multicolumn{{{N+1}}}{{|l|}}{{\\textbf{{ {comparison_title}}}}}\\\\")
    lines.append("                 \\hline")
    lines.append(fmt_row("PSNR", psnr_comp, ".2f"))
    lines.append("                 \\hline")
    lines.append(fmt_row("SSIM", ssim_comp, ".4f"))
    lines.append("                 \\hline")
    lines.append(fmt_row("CLIP", clip_comp, ".4f"))
    lines.append("                 \\hline")
    lines.append("    \\end{tabular}\n")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

def denormalise_image(image):
    #denormalise and save example image to disk
    
    example_image = image * 255.0
    example_image = example_image.numpy().astype(np.uint8)  # Convert to numpy array
    img = Image.fromarray(example_image)  # Create an Image object
    filename = 'outputs/example_images/in_image.png' 
    img.save(filename)  # Save the image to disk
            
def create_plots_from_latent(latent, channel_output, output_folder):
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim

    # Histogram comparison
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(latent.flatten(), bins=50, alpha=0.7, label="Latent")
    plt.title("Histogram of Latent Features")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(channel_output.flatten(), bins=50, alpha=0.7, label="Channel Output")
    plt.title("Histogram of Channel Processed Features")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    output_path = f'{output_folder}histogram.png'
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    # Visualize first latent and channel output as images
    latent_image = latent[0]
    channel_image = channel_output[0]

    # Dynamically compute the reshape size if needed
    size = int(np.sqrt(latent_image.size))
    if size * size == latent_image.size:
        latent_image = latent_image.reshape(size, size)
        channel_image = channel_image.reshape(size, size)
    else:
        print(f"Latent features cannot be reshaped to a square grid. Skipping grid visualization.")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(latent_image, cmap="viridis")
    plt.title("Original Latent Features")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(channel_image, cmap="viridis")
    plt.title("Channel Processed Features")
    plt.colorbar()

    plt.tight_layout()
    output_path = f'{output_folder}firstLatent.png'
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    # Compute SSIM for the noise analysis
    if latent_image.shape == channel_image.shape:
        ssim_index = ssim(latent_image, channel_image, data_range=channel_image.max() - channel_image.min(),channel_axis=-1)
        print(f"Structural Similarity Index (SSIM): {ssim_index}")
    else:
        print(f"Cannot compute SSIM due to shape mismatch: {latent_image.shape} vs {channel_image.shape}")

    # Noise visualization
    noise = channel_output - latent

    # Reshape noise[0] if it represents a flattened grid
    latent_size = noise[0].size  # Total size of the feature vector
    height = int(np.sqrt(latent_size))  # Compute height (assume square grid)
    width = latent_size // height       # Compute width (ensure consistency)

    if height * width == latent_size:  # Ensure it's a perfect grid
        reshaped_noise = noise[0].reshape(height, width)
    else:
        print(f"Warning: Cannot reshape noise to 2D. Size is not a perfect square: {latent_size}")
        reshaped_noise = noise[0]  # Use original 1D representation if reshaping is not possible

    print(f"Noise min: {noise.min()}, max: {noise.max()}, mean: {noise.mean()}")

    plt.figure(figsize=(6, 6))
    plt.imshow(reshaped_noise, cmap="seismic", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Noise Introduced by the Channel")
    output_path = f'{output_folder}noise.png'
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    normalized_noise = (reshaped_noise - reshaped_noise.min()) / (reshaped_noise.max() - reshaped_noise.min())
    plt.figure(figsize=(6, 6))
    plt.imshow(normalized_noise, cmap="seismic", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Normalized Noise Introduced by the Channel")
    output_path = f'{output_folder}noise_normalized.png'
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
