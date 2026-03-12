import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from pyldpc import make_ldpc, encode, decode
from scipy.special import expit as sigmoid
import random  # <-- Add this line


# Parameters
DATASET_PATH = '../dataset/EuroSAT_Uncompressed_BMP/'
OUTPUT_IMAGE = '../outputs/PipelineTesting/transmission_result.png'
NUM_IMAGES = 1
SNR_dB = 25  # signal-to-noise ratio in dB

def jpeg_compress(image, quality=75):
    _, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded

def jpeg_decompress(encoded):
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def bpsk_modulate(bits):
    return 1 - 2*bits  # 0 → 1, 1 → -1

def awgn(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    noise_power = 1 / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def process_image(image, H, G):
    # Flatten the image to a byte array
    original_shape = image.shape
    image_bytes = image.astype(np.uint8).flatten()
    bitstream = np.unpackbits(image_bytes)

    # LDPC encode
    padded_len = (len(bitstream) + G.shape[1] - 1) // G.shape[1] * G.shape[1]
    padded_bits = np.zeros(padded_len, dtype=np.uint8)
    padded_bits[:len(bitstream)] = bitstream
    bit_chunks = padded_bits.reshape(-1, G.shape[1])
    encoded_chunks = np.array([encode(G, chunk, snr=SNR_dB) for chunk in bit_chunks])
    encoded_bits = encoded_chunks.flatten()

    # BPSK modulation
    modulated_signal = bpsk_modulate(encoded_bits)

    # AWGN
    received_signal = awgn(modulated_signal, SNR_dB)

    # Demodulation to LLR
    llr = 2 * received_signal
    decoded_chunks = np.array([decode(H, llr[i:i+H.shape[1]], snr=SNR_dB) for i in range(0, len(llr), H.shape[1])])
    decoded_bits = decoded_chunks.flatten()[:len(bitstream)]

    # Reconstruct the image
    try:
        expected_size = np.prod(original_shape)
        decoded_bytes = np.packbits(decoded_bits)[:expected_size]
        reconstructed = decoded_bytes.reshape(original_shape).astype(np.uint8)
    except:
        reconstructed = np.zeros_like(image)  # fallback in case of reshaping failure
    return reconstructed

def main():
    bmp_paths = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith('.bmp'):
                bmp_paths.append(os.path.join(root, file))

    selected_paths = random.sample(bmp_paths, min(NUM_IMAGES, len(bmp_paths)))

    # LDPC setup
    n = 512
    d_v = 2
    d_c = 4
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    output_grid = []
    for img_path in selected_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        print(f"✅ Processing {img_path}...")

        rec = process_image(img, H, G)
        if rec is not None:
            combined = np.hstack((img, rec))
            output_grid.append(combined)
        else:
            print(f"⚠️ Reconstructed image is None for {img_path}")

    if output_grid:
        final_image = np.vstack(output_grid)
        os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
        imsave(OUTPUT_IMAGE, final_image)
        print(f"✅ Saved transmission result to {OUTPUT_IMAGE}")
    else:
        print("❌ No images were processed successfully.")

if __name__ == '__main__':
    main()
