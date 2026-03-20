import tensorflow as tf


def compute_patch_grid(length, patch_size, stride):
    """Return top-left offsets that cover a dimension with overlap and edge coverage."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    if length <= patch_size:
        return [0]

    positions = list(range(0, max(length - patch_size, 0) + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def build_blend_window(patch_size, channels, power=2.0, epsilon=1e-3):
    """Construct a separable 2D window for smooth overlap-aware patch blending."""
    coords = tf.linspace(-1.0, 1.0, patch_size)
    ramp = 1.0 - tf.abs(coords)
    ramp = tf.pow(tf.maximum(ramp, epsilon), power)
    window_2d = tf.tensordot(ramp, ramp, axes=0)
    window_2d = window_2d / tf.reduce_max(window_2d)
    window = tf.reshape(window_2d, [patch_size, patch_size, 1])
    return tf.tile(window, [1, 1, channels])


def extract_overlapping_patches(images, patch_size, stride):
    """Extract overlapping patches for a statically-sized image tensor with dynamic batch support."""
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    height = images.shape[1]
    width = images.shape[2]
    channels = images.shape[3]
    if height is None or width is None or channels is None:
        raise ValueError("Patch extraction requires statically-known image height, width, and channels.")

    y_positions = compute_patch_grid(int(height), patch_size, stride)
    x_positions = compute_patch_grid(int(width), patch_size, stride)
    patch_rows = len(y_positions)
    patch_cols = len(x_positions)

    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    expected_rows = patches.shape[1]
    expected_cols = patches.shape[2]
    if expected_rows is not None and expected_rows != patch_rows:
        raise ValueError(f"Unexpected patch row count: expected {patch_rows}, got {expected_rows}")
    if expected_cols is not None and expected_cols != patch_cols:
        raise ValueError(f"Unexpected patch col count: expected {patch_cols}, got {expected_cols}")

    patches = tf.reshape(patches, [-1, patch_rows, patch_cols, patch_size, patch_size, channels])
    flat_patches = tf.reshape(patches, [-1, patch_size, patch_size, channels])

    metadata = {
        "y_positions": y_positions,
        "x_positions": x_positions,
        "image_height": int(height),
        "image_width": int(width),
        "patch_rows": patch_rows,
        "patch_cols": patch_cols,
    }
    return flat_patches, metadata


def fold_overlapping_patches(patches, metadata, output_shape, stride, window=None):
    """Blend decoded patches back onto an image canvas using weighted averaging."""
    del stride

    patches = tf.convert_to_tensor(patches, dtype=tf.float32)
    batch_size = tf.shape(patches)[0] // (metadata["patch_rows"] * metadata["patch_cols"])
    output_height = metadata["image_height"]
    output_width = metadata["image_width"]
    channels = patches.shape[-1]
    if channels is None:
        channels = output_shape[-1]

    if window is None:
        window = tf.ones([patches.shape[1], patches.shape[2], channels], dtype=tf.float32)
    else:
        window = tf.convert_to_tensor(window, dtype=tf.float32)

    patches = tf.reshape(
        patches,
        [batch_size, metadata["patch_rows"], metadata["patch_cols"], patches.shape[1], patches.shape[2], channels],
    )
    weighted_patches = patches * window[tf.newaxis, tf.newaxis, tf.newaxis, ...]

    accumulation = tf.zeros([batch_size, output_height, output_width, channels], dtype=tf.float32)
    normalizer = tf.zeros([batch_size, output_height, output_width, channels], dtype=tf.float32)

    for row_index, y in enumerate(metadata["y_positions"]):
        for col_index, x in enumerate(metadata["x_positions"]):
            patch = weighted_patches[:, row_index, col_index, :, :, :]
            paddings = [
                [0, 0],
                [y, output_height - y - patch.shape[1]],
                [x, output_width - x - patch.shape[2]],
                [0, 0],
            ]
            accumulation += tf.pad(patch, paddings)
            normalizer += tf.pad(window[tf.newaxis, ...], paddings)

    return accumulation / tf.maximum(normalizer, 1e-6)
