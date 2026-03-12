import os
import glob
from PIL import Image

def convert_to_bmp_uncompressed(src_root, dst_root):
    image_paths = glob.glob(os.path.join(src_root, '*', '*.jpg'))

    print(f"Found {len(image_paths)} JPEG images to convert...\n")

    total_input_size = 0
    total_output_size = 0

    for img_path in image_paths:
        # Destination path
        rel_path = os.path.relpath(img_path, src_root)
        dst_path = os.path.join(dst_root, os.path.splitext(rel_path)[0] + '.bmp')

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            # Open and convert to 24-bit RGB
            img = Image.open(img_path).convert('RGB')

            # Save as BMP (fully uncompressed)
            img.save(dst_path, format='BMP')

            input_size = os.path.getsize(img_path)
            output_size = os.path.getsize(dst_path)
            total_input_size += input_size
            total_output_size += output_size

            print(f"✅ {rel_path} | JPEG: {input_size/1024:.1f} KB → BMP: {output_size/1024:.1f} KB")
        except Exception as e:
            print(f"❌ Failed to convert {img_path}: {e}")

    print(f"\n✔️ Finished converting {len(image_paths)} images.")
    print(f"📊 Total JPEG size: {total_input_size / (1024**2):.2f} MB")
    print(f"📊 Total BMP size : {total_output_size / (1024**2):.2f} MB")
    print(f"📁 Output written to: {dst_root}")

if __name__ == '__main__':
    src_dir = '../dataset/EuroSAT_RGB/'
    dst_dir = '../dataset/EuroSAT_Uncompressed_BMP/'
    convert_to_bmp_uncompressed(src_dir, dst_dir)
