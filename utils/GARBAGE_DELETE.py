import numpy as np
from pyldpc import make_ldpc, ldpc_images
from pyldpc.utils_img import gray2bin, rgb2bin
from matplotlib import pyplot as plt
from PIL import Image
from skimage.io import imsave
from time import time
import os


n = 200
d_v = 3
d_c = 4
seed = 42

# First we create a small LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)

snr = 8


img = np.asarray(Image.open("../outputs/PipelineTesting/Industrial_4.bmp"))
# Convert it to a binary matrix
img_bin = rgb2bin(img)
print("img shape: (%s, %s, %s)" % img.shape)
print("img Binary shape: (%s, %s, %s)" % img_bin.shape)


img_coded, _ = ldpc_images.encode_img(G, img_bin, 0, seed=seed)

print("Coded Tiger shape", img_coded.shape)

t = time()
tiger_decoded = ldpc_images.decode_img(G, H, tiger_coded, snr, tiger_bin.shape)
t = time() - t
print("Tiger | Decoding time: ", t)

error_decoded_tiger = abs(tiger - tiger_decoded).mean()
error_noisy_tiger = abs(tiger_noisy - tiger).mean()


# Prepare title overlays
titles_tiger = [
    f"Original",
    f"Noisy | Err = {error_noisy_tiger:.3f} %",
    f"Decoded | Err = {error_decoded_tiger:.3f} %"
]
images = [tiger, tiger_noisy, tiger_decoded]

# Save as a grid to disk
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, img, title in zip(axes, images, titles_tiger):
    ax.imshow(img)
    ax.set_title(title, fontsize=16)
    ax.axis("off")
plt.tight_layout()

output_path = "../outputs/PipelineTesting/comparison_grid.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()
print(f"✅ Saved comparison grid to {output_path}")