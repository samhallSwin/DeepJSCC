from PIL import Image

def bin_to_bmp():
    bin_path = '../../wimax_ldpc_lib/build/lib/decoded.bin'
    bmp_path = '../../wimax_ldpc_lib/build/lib/test.bmp'

    # Manually set the original image size and mode
    width, height = 64, 64  # <-- Update if different
    mode = 'RGB'

    with open(bin_path, 'rb') as f:
        data = f.read()

    expected_size = width * height * len(mode)
    if len(data) != expected_size:
        raise ValueError(f"Binary file size mismatch. Expected {expected_size} bytes, got {len(data)}.")

    img = Image.frombytes(mode, (width, height), data)
    img.save(bmp_path)

    print(f"Converted {bin_path} to {bmp_path} with size ({width}, {height}) and mode {mode}.")

if __name__ == "__main__":
    bin_to_bmp()
