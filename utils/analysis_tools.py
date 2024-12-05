#NOTE: This file is a mess and half of it is a graveyard for old functions that are probably broken. One day I'll clean it up.


import time
import numpy as np
import tensorflow as tf
import os
import io
import cv2
from pyldpc import make_ldpc, encode, decode, get_message
import tempfile
import matplotlib.pyplot as plt
import contextlib
from skimage import img_as_ubyte, img_as_float32
from skimage.util import img_as_float32, img_as_ubyte
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as compare_ssim

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import ebnodb2no
from sionna.channel import AWGN, FlatFadingChannel
from models.channellayer import RayleighChannel, AWGNChannel, RicianChannel

from PIL import Image
import math
from tqdm.auto import tqdm


import numpy as np

def save_image(image, filepath):
    """
    Save a NumPy array as an image file.

    Args:
        image: NumPy array of the image.
        filepath: Path to save the image.
    """
    # Ensure the image is in the range [0, 1] and convert to uint8
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1)  # Clip values to [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
    elif np.issubdtype(image.dtype, np.integer):
        image = np.clip(image, 0, 255).astype(np.uint8)  # Clip integer values to [0, 255]

    # Ensure the shape is valid
    if image.ndim == 3 and image.shape[2] == 1:  # Single-channel 3D -> 2D
        image = np.squeeze(image, axis=-1)
    elif image.ndim == 3 and image.shape[0] == 1 and image.shape[1] == 1:
        image = image.squeeze(axis=(0, 1))  # Squeeze unnecessary dimensions

    # Convert the array to a PIL Image and save it
    img = Image.fromarray(image)
    img.save(filepath)

def calculate_psnr(original, processed):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        original: Original image as a numpy array.
        processed: Processed image as a numpy array.

    Returns:
        PSNR value in dB.
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    max_pixel = 255.0  # Assuming 8-bit images
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(original, processed):
    """
    Calculate the SSIM (Structural Similarity Index) between two images.

    Args:
        original: Original image as a numpy array.
        processed: Processed image as a numpy array.

    Returns:
        SSIM value (float between -1 and 1, where 1 means identical images).
    """
    original_gray = original if len(original.shape) == 2 else np.mean(original, axis=-1)  # Convert to grayscale
    processed_gray = processed if len(processed.shape) == 2 else np.mean(processed, axis=-1)  # Convert to grayscale
    
    ssim, _ = compare_ssim(original_gray, processed_gray, full=True, data_range=255)
    return ssim

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
        plt.savefig('outputs/example_images/latent_space.png')

def run_awgn_on_images(dataset, awgn_channel, config):
    """
    Run the original images through the AWGNChannel and compare them to their noisy versions.
    """
    original_images = []
    noisy_images = []
    psnr_values = []
    ssim_values = [1]

    for batch_images, _ in dataset.take(1):  # Take one batch from the dataset
        # Normalize the images if needed
        batch_images = batch_images.numpy() * 255.0

        # Convert images to (batch, num_features, 2) for AWGNChannel
        batch_size, height, width, channels = batch_images.shape
        num_features = height * width * channels // 2  # Ensure correct feature size
        reshaped_images = tf.reshape(batch_images, (batch_size, num_features, 2))

        # Pass through AWGNChannel
        noisy_output = awgn_channel(reshaped_images)

        # Reshape back to image format
        noisy_output_reshaped = tf.reshape(noisy_output, (batch_size, height, width, channels)).numpy()

        # Calculate PSNR and SSIM
        for orig, noisy in zip(batch_images, noisy_output_reshaped):
            psnr_values.append(tf.image.psnr(orig / 255.0, noisy / 255.0, max_val=1.0).numpy())
            # Explicitly set win_size and channel_axis for SSIM
           # ssim_values.append(
           #     sk_ssim(orig, noisy, multichannel=True, win_size=min(height, width, 7))
           # )

        original_images.append(batch_images)
        noisy_images.append(noisy_output_reshaped)

    # Aggregate results
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return original_images, noisy_images, avg_psnr, avg_ssim

class BPGEncoder:
    def __init__(self, working_directory='./analysis/temp'):
        """
        Initialize the BPGEncoder.

        Args:
            working_directory: Directory to save temporary files.
        """
        self.working_directory = working_directory

    def run_bpgenc(self, qp, input_dir, output_dir='temp.bpg'):
        """
        Run the BPG encoding process.

        Args:
            qp: Quality parameter for encoding.
            input_dir: Path to the input image file.
            output_dir: Path to save the output encoded file.

        Returns:
            The size of the output file in bytes.
        """
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgenc {input_dir} -q {qp} -o {output_dir} -f 444')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1

    def get_qp(self, input_dir, byte_threshold, output_dir='temp.bpg'):
        """
        Find the quality parameter (QP) that meets the byte size constraint.

        Args:
            input_dir: Path to the input image file.
            byte_threshold: Maximum allowed size in bytes.
            output_dir: Path to save the output encoded file.

        Returns:
            The determined quality parameter (QP).
        """
        quality_max = 51
        quality_min = 0
        quality = (quality_max - quality_min) // 2

        while True:
            qp = 51 - quality
            bytes = self.run_bpgenc(qp, input_dir, output_dir)
            if quality == 0 or quality == quality_min or quality == quality_max:
                break
            elif bytes > byte_threshold and quality_min != quality - 1:
                quality_max = quality
                quality -= (quality - quality_min) // 2
            elif bytes > byte_threshold and quality_min == quality - 1:
                quality_max = quality
                quality -= 1
            elif bytes < byte_threshold and quality_max > quality:
                quality_min = quality
                quality += (quality_max - quality) // 2
            else:
                break

        return qp

    def encode(self, image_array, max_bytes, header_bytes=22):
        """
        Encode an image array into a BPG file.

        Args:
            image_array: uint8 Numpy array with shape (h, w, c).
            max_bytes: Maximum size of the encoded file in bytes (excluding header).
            header_bytes: Header size to exclude in byte size calculation.

        Returns:
            Encoded bit array as a Numpy array of floats.
        """
        input_dir = f'{self.working_directory}/temp_enc.png'
        output_dir = f'{self.working_directory}/temp_enc.bpg'

        # Save image to file
        im = Image.fromarray(image_array, 'RGB')
        im.save(input_dir)

        # Calculate quality parameter (QP) and encode
        qp = self.get_qp(input_dir, max_bytes + header_bytes, output_dir)
        if self.run_bpgenc(qp, input_dir, output_dir) < 0:
            raise RuntimeError("BPG encoding failed")

        # Read the encoded file and convert to a binary array
        return np.unpackbits(np.fromfile(output_dir, dtype=np.uint8)).astype(np.float32)
    
class LDPCTransmitter():
    '''
    Transmits given bits (float array of '0' and '1') with LDPC.
    '''
    def __init__(self, k, n, m, esno_db, channel='AWGN'):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM)
        esno_db: channel SNR
        channel: 'AWGN' or 'Rayleigh'
        '''
        self.k = k
        self.n = n
        self.num_bits_per_symbol = round(math.log2(m))

        constellation_type = 'qam' if m != 2 else 'pam'
        self.constellation = Constellation(constellation_type, num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper('app', constellation=self.constellation)
        self.channel = AWGN() if channel == 'AWGN' else FlatFadingChannel
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
        self.esno_db = esno_db
    

    def send(self, source_bits):
        '''
        source_bits: float np array of '0' and '1', whose total # of bits is divisible with k
        '''
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        no = ebnodb2no(self.esno_db, num_bits_per_symbol=1, coderate=1)
        c = self.encoder(u)
        x = self.mapper(c)
        y = self.channel([x, no])
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len(source_bits)]
    
class ImageProcessor:
    def __init__(self, image_shape):
        """
        Initialize the image processor with the specified shape.
        
        Args:
            image_shape: Tuple indicating the shape of the images (height, width, channels).
        """
        self.image_shape = image_shape
        self.mean_pixel_values = {
            (32, 32, 3): np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]) * 255,  # CIFAR-10
            (64, 64, 3): np.array([0.5, 0.5, 0.5]) * 255,  # Default mean for 64x64 images
        }

    def get_mean_image(self):
        """
        Get the mean image for the specified shape.
        
        Returns:
            A mean image array reshaped for broadcasting.
        """
        if self.image_shape not in self.mean_pixel_values:
            raise ValueError(f"Mean pixel values not defined for shape {self.image_shape}")
        
        mean_pixel = self.mean_pixel_values[self.image_shape]
        return np.reshape(mean_pixel, [1] * (len(self.image_shape) - 1) + [3]).astype(np.uint8)

class BPGDecoder:
    def __init__(self, working_directory='./analysis/temp', image_shape=(32, 32, 3)):
        """
        Initialize the BPGDecoder with a specified working directory and image shape.
        
        Args:
            working_directory: Directory to save temporary files.
            image_shape: Shape of the images to decode (default: CIFAR-10).
        """
        self.working_directory = working_directory
        self.image_processor = ImageProcessor(image_shape)


    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgdec {input_dir} -o {output_dir}')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1

    def decode(self, bit_array, image_shape, filename):
        """
        Decode a bit array back into an image, or return a mean image if decoding fails.
        
        Args:
            bit_array: Numpy array representing the encoded bit stream.
            image_shape: Shape of the output image (e.g., (32, 32, 3)).
        
        Returns:
            The decoded image or a mean image if decoding fails.
        """
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/z_{filename}.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        with open(input_dir, "wb") as binary_file:
            binary_file.write(byte_array.tobytes())

        # Use the mean image from ImageProcessor as a fallback
        mean_image = self.image_processor.get_mean_image()
        if self.run_bpgdec(input_dir, output_dir) < 0:
            return 0 * np.ones(image_shape) + mean_image
        else:
            decoded_image = np.array(Image.open(output_dir).convert('RGB'))
            if decoded_image.shape != image_shape:
                return 0 * np.ones(image_shape) + mean_image
            return decoded_image

def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1
    
    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image


def runBPGplusLDPC(config):

    from utils.datasets import dataset_generator
    test_ds = dataset_generator('./dataset/EuroSAT_RGB_split/test/', config)

    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()

    bw_ratio = config.bw_ratio
    snrs = config.snrs
    mcs = config.mcs
    '''
    (3072, 6144), (3072, 4608), (1536, 4608)
    BPSK, 4-QAM, 16-QAM, 64-QAM
    '''

    for esno_db in snrs:
        for bw in bw_ratio:
            for k, n, m in mcs:
                i = 0
                psnr = 0
                ssim = 0
                total_images = 0
                ldpctransmitter = LDPCTransmitter(k, n, m, esno_db, 'AWGN')
                image_count = 0
                for image, _ in tqdm(test_ds):

                    b, _, _, _ = image.shape
                    image = tf.cast(imBatchtoImage(image), tf.uint8)
                    max_bytes = b * config.image_width * config.image_height * config.image_channels * bw * math.log2(m) * k / n / 8
                    src_bits = bpgencoder.encode(image.numpy(), max_bytes)
                    rcv_bits = ldpctransmitter.send(src_bits)
                    

                    filename = f'SNR={esno_db},bw={bw},k={k},n={n},m={m},PSNR={psnr:.2f},SSIM={ssim:.2f}'
                    decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape, filename)
                    total_images += b
                    psnr = (total_images - b) / (total_images) * psnr + float(b * tf.image.psnr(decoded_image, image, max_val=255)) / (total_images)
                    ssim = (total_images - b) / (total_images) * ssim + float(b * tf.image.ssim(tf.cast(decoded_image, dtype=tf.float32), tf.cast(image, dtype=tf.float32), max_val=255)) / (total_images)

                    image_count += 1 
                print(f'SNR={esno_db},bw={bw},k={k},n={n},m={m},PSNR={psnr:.2f},SSIM={ssim:.2f}')

def run_single_image_JPEG2000_OLD(image, config, save_path=None):
    """
    Compress an image using JPEG2000, pass it through an AWGN channel,
    and decompress it for comparison.

    Args:
        image: Input image as a NumPy array.
        config: Configuration object with channel parameters like SNR.
        save_path: Optional directory to save intermediate files.

    Returns:
        decompressed_image: The decompressed image after passing through the AWGN channel.
    """
    # Normalize the image to [0, 1]
    image = img_as_float32(image)

    # Encode the image using JPEG2000
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as temp_file:
        temp_filepath = temp_file.name
        imsave(temp_filepath, img_as_ubyte(image), plugin="imageio", format_str="jp2")

    # Read back the compressed JPEG2000 image for channel simulation
    compressed_image = imread(temp_filepath, plugin="imageio")

    # Add AWGN channel noise to the compressed image
    compressed_image = compressed_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    compressed_flatten = compressed_image.flatten()
    iq_components = np.stack((compressed_flatten, np.zeros_like(compressed_flatten)), axis=-1)  # Add I/Q components
    awgn_input = np.expand_dims(iq_components, axis=0)  # Add batch dimension

    # Pass through AWGNChannel
    awgn_channel = AWGNChannel(snrdB=config.train_snrdB)
    noised_flatten = awgn_channel(tf.convert_to_tensor(awgn_input)).numpy().squeeze()

    # Extract only the in-phase component (I) for further decoding
    noised_image = noised_flatten[:, 0].reshape(compressed_image.shape)

    # Normalize back to [0, 1]
    noised_image = np.clip(noised_image, 0, 1)

    # Decode the JPEG2000 image from the noised data
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as temp_output_file:
        temp_output_filepath = temp_output_file.name
        save_image(noised_image, temp_output_filepath)  # Use updated save_image function

    decompressed_image = imread(temp_output_filepath, plugin="imageio").astype(np.float32) / 255.0

    # Optionally save the decompressed image
    if save_path:
        output_filepath = f"{save_path}/jpeg2000_processed.png"
        save_image(decompressed_image, output_filepath)  # Use updated save_image function

    return decompressed_image

def run_single_image_JPEG2000(image, config, save_path=None):
    # Normalize the image to [0, 1]
    image = img_as_float32(image)

    # Encode the image using JPEG2000
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as temp_file:
        temp_filepath = temp_file.name
        imsave(temp_filepath, img_as_ubyte(image), plugin="imageio", format_str="jp2")

    # Read back the compressed JPEG2000 image
    compressed_image = imread(temp_filepath, plugin="imageio")
    compressed_image = compressed_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Flatten the image for LDPC encoding
    bitstream = compressed_image.flatten()

    # LDPC Coding Parameters
    n, d_v, d_c = 2000, 3, 4
    H, G = make_ldpc(n, d_v, d_c, seed=42)
    k = G.shape[1]  # Input size for LDPC

    # Split bitstream into blocks
    num_blocks = math.ceil(bitstream.size / k)
    padded_bitstream = np.pad(bitstream, (0, num_blocks * k - bitstream.size), mode='constant')
    blocks = padded_bitstream.reshape(num_blocks, k)

    # Encode, pass through channel, and decode each block
    recovered_blocks = []
    for block in blocks:
        encoded_bits = encode(G, block, snr=config.train_snrdB)

        # Simulate AWGN Channel
        iq_components = np.stack((encoded_bits, np.zeros_like(encoded_bits)), axis=-1)
        awgn_input = np.expand_dims(iq_components, axis=0)
        awgn_channel = AWGNChannel(snrdB=config.train_snrdB)
        noised_iq = awgn_channel(tf.convert_to_tensor(awgn_input)).numpy().squeeze()

        # Extract and normalize the in-phase component (I)
        noised_bits = noised_iq[:, 0]
        noised_bits = np.clip(noised_bits, -1, 1)
        noised_bits = noised_bits.astype(np.float64)

        # Decode the LDPC encoded message
        decoded_bits = decode(H, noised_bits, maxiter=50, snr=config.train_snrdB)
        recovered_message = get_message(G, decoded_bits)

        recovered_blocks.append(recovered_message)

    # Reconstruct the full bitstream
    recovered_bitstream = np.concatenate(recovered_blocks)[:bitstream.size]

    # Reshape the recovered bitstream into the original compressed image shape
    recovered_image = recovered_bitstream.reshape(compressed_image.shape)

    # Clip values to [0, 1]
    recovered_image = np.clip(recovered_image, 0, 1)

    # De-normalize recovered image to [0, 255]
    recovered_image = (recovered_image * 255).astype(np.uint8)

    # Save the de-normalized image
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as temp_output_file:
        temp_output_filepath = temp_output_file.name
        save_image(recovered_image, temp_output_filepath)

    decompressed_image = imread(temp_output_filepath, plugin="imageio").astype(np.float32) / 255.0

    # Optionally save the decompressed image
    if save_path:
        output_filepath = f"{save_path}/jpeg2000_processed.png"
        save_image(decompressed_image, output_filepath)

    return decompressed_image

def run_single_image_BPGplusLDPC(image, config, bw_ratio, snrs, mcs, save_path='./outputs', LDPCon=True):
    """
    Process a single image through the BPG + LDPC pipeline and save both original and processed images.

    Args:
        image: Input image as a numpy array (uint8, shape: (h, w, c)).
        config: Configuration object containing image properties and working parameters.
        bw_ratio: Bandwidth ratio for processing.
        snrs: Signal-to-noise ratio (SNR) in dB.
        mcs: Modulation and coding scheme as a tuple (k, n, m).
        save_path: Directory path to save the original and processed images.

    Returns:
        Processed image as a numpy array.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the original image for comparison
    original_image_path = os.path.join(save_path, 'original_image.png')
    Image.fromarray(image).save(original_image_path)

    # Unpack modulation and coding scheme
    k, n, m = mcs

    # Initialize encoder, transmitter, and decoder
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    ldpctransmitter = LDPCTransmitter(k, n, m, snrs, 'AWGN')

    # Calculate maximum bytes for encoding based on bandwidth ratio
    h, w, c = image.shape

    if LDPCon:
        max_bytes = h * w * c * bw_ratio * math.log2(m) * k / n / 8
    else:
        max_bytes = h * w * c * bw_ratio * math.log2(1 + 10 ** (snrs/10)) / 8
        
    
    # BPG encode the image
    src_bits = bpgencoder.encode(image, max_bytes)

    # Transmit the encoded bits through the LDPC transmitter
    if LDPCon:
        rcv_bits = ldpctransmitter.send(src_bits)
        decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape, 'processed_image')
    else:
        decoded_image = bpgdecoder.decode(src_bits, image.shape, 'processed_image')

    # Decode the received bits back into an image
    

    # Save the processed image
    processed_image_path = os.path.join(save_path, 'processed_image.png')
    Image.fromarray(decoded_image).save(processed_image_path)

    # Return the processed image
    return decoded_image, max_bytes

def compress_with_jpeg2000(image, bit_rate):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[bit_rate])
    compressed_data = buffer.getvalue()
    return compressed_data

def decompress_with_jpeg2000(compressed_data):
    buffer = io.BytesIO(compressed_data)
    image = Image.open(buffer)
    return image

def simulate_awgn_channel(data, snr_dB):
    snr_linear = 10 ** (snr_dB / 10)
    signal_power = np.mean(np.abs(data) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise