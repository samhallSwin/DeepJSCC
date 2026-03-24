import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_NODATA_SCAN_CACHE = {}


def _list_image_files(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    image_paths = []
    for extension in SUPPORTED_EXTENSIONS:
        image_paths.extend(directory.rglob(f"*{extension}"))
        image_paths.extend(directory.rglob(f"*{extension.upper()}"))
    image_paths = sorted({path.resolve() for path in image_paths})
    if not image_paths:
        raise FileNotFoundError(f"No image files found under {directory}")
    return [str(path) for path in image_paths]


def _infer_class_names(image_paths, root_dir):
    root_dir = Path(root_dir).resolve()
    label_names = []

    for image_path in image_paths:
        parent_name = Path(image_path).resolve().parent.relative_to(root_dir).parts
        label_name = parent_name[0] if parent_name else "images"
        label_names.append(label_name)

    unique_labels = sorted(set(label_names))
    if unique_labels == ["images"]:
        return [], [0] * len(image_paths)

    name_to_index = {name: idx for idx, name in enumerate(unique_labels)}
    labels = [name_to_index[name] for name in label_names]
    return unique_labels, labels


def _compute_nodata_fraction(image_array, black_threshold, white_threshold):
    black_mask = np.all(image_array <= black_threshold, axis=-1)
    white_mask = np.all(image_array >= white_threshold, axis=-1)
    black_fraction = float(np.mean(black_mask))
    white_fraction = float(np.mean(white_mask))
    return black_fraction, white_fraction, max(black_fraction, white_fraction)


def _write_nodata_report(rows, params, split_name):
    if not getattr(params, "save_nodata_report", True):
        return None

    report_dir = Path("logs")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{params.experiment_name}_nodata_{split_name}.tsv"

    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "black_fraction",
                "white_fraction",
                "nodata_fraction",
                "discarded",
            ],
            delimiter="	",
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(report_path)


def _nodata_cache_path(image_paths, params, split_name):
    cache_dir = Path("logs") / "nodata_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_payload = {
        "split_name": split_name,
        "image_root": str(Path(image_paths[0]).resolve().parent) if image_paths else "",
        "image_count": len(image_paths),
        "first_image": image_paths[0] if image_paths else "",
        "last_image": image_paths[-1] if image_paths else "",
        "black_threshold": int(getattr(params, "nodata_black_threshold", 4)),
        "white_threshold": int(getattr(params, "nodata_white_threshold", 251)),
        "discard_threshold": float(getattr(params, "nodata_discard_threshold", 0.2)),
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{split_name}_{cache_key}.tsv"


def _load_nodata_cache(cache_path):
    if not cache_path.exists():
        return None

    with cache_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="	")
        rows = []
        for row in reader:
            rows.append(
                {
                    "image_path": row["image_path"],
                    "black_fraction": float(row["black_fraction"]),
                    "white_fraction": float(row["white_fraction"]),
                    "nodata_fraction": float(row["nodata_fraction"]),
                    "discarded": row["discarded"].lower() == "true",
                }
            )
    return rows


def _write_nodata_cache(rows, cache_path):
    with cache_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "black_fraction",
                "white_fraction",
                "nodata_fraction",
                "discarded",
            ],
            delimiter="	",
        )
        writer.writeheader()
        writer.writerows(rows)


def _scan_nodata_rows(image_paths, params, split_name):
    cache_key = (
        tuple(image_paths),
        int(getattr(params, "nodata_black_threshold", 4)),
        int(getattr(params, "nodata_white_threshold", 251)),
        float(getattr(params, "nodata_discard_threshold", 0.2)),
    )
    if cache_key in _NODATA_SCAN_CACHE:
        return _NODATA_SCAN_CACHE[cache_key]

    cache_path = _nodata_cache_path(image_paths, params, split_name)
    cached_rows = _load_nodata_cache(cache_path)
    if cached_rows is not None:
        print(f"[{split_name}] no-data scan: loaded cached results from {cache_path}")
        report_path = _write_nodata_report(cached_rows, params, split_name)
        _NODATA_SCAN_CACHE[cache_key] = (cached_rows, report_path)
        return cached_rows, report_path

    black_threshold = int(getattr(params, "nodata_black_threshold", 4))
    white_threshold = int(getattr(params, "nodata_white_threshold", 251))
    discard_threshold = float(getattr(params, "nodata_discard_threshold", 0.2))

    print(f"[{split_name}] no-data scan: computing fresh results for {len(image_paths)} images")
    rows = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        black_fraction, white_fraction, nodata_fraction = _compute_nodata_fraction(
            image_array,
            black_threshold=black_threshold,
            white_threshold=white_threshold,
        )
        rows.append(
            {
                "image_path": image_path,
                "black_fraction": black_fraction,
                "white_fraction": white_fraction,
                "nodata_fraction": nodata_fraction,
                "discarded": nodata_fraction > discard_threshold,
            }
        )

    _write_nodata_cache(rows, cache_path)
    print(f"[{split_name}] no-data scan: saved cached results to {cache_path}")
    report_path = _write_nodata_report(rows, params, split_name)
    _NODATA_SCAN_CACHE[cache_key] = (rows, report_path)
    return rows, report_path


def _apply_nodata_filter(image_paths, labels, params, split_name):
    if not getattr(params, "enable_nodata_check", True):
        return image_paths, labels

    rows, report_path = _scan_nodata_rows(image_paths, params, split_name)
    discard_enabled = bool(getattr(params, "exclude_high_nodata_images", True))
    report_threshold = float(getattr(params, "nodata_report_threshold", 0.01))

    kept_paths = []
    kept_labels = []
    nodata_fractions = []
    affected_images = 0
    discarded_images = 0

    for image_path, label, row in zip(image_paths, labels, rows):
        nodata_fraction = float(row["nodata_fraction"])
        nodata_fractions.append(nodata_fraction)
        if nodata_fraction >= report_threshold:
            affected_images += 1
        if discard_enabled and row["discarded"]:
            discarded_images += 1
            continue
        kept_paths.append(image_path)
        kept_labels.append(label)

    mean_fraction = float(np.mean(nodata_fractions)) if nodata_fractions else 0.0
    max_fraction = float(np.max(nodata_fractions)) if nodata_fractions else 0.0
    print(
        f"[{split_name}] no-data scan: {affected_images}/{len(image_paths)} images >= "
        f"{report_threshold:.3f} fraction, mean={mean_fraction:.4f}, max={max_fraction:.4f}, "
        f"discarded={discarded_images}"
    )
    if report_path:
        print(f"[{split_name}] no-data report saved to {report_path}")

    if not kept_paths:
        raise RuntimeError(
            f"No images remained in {split_name} after no-data filtering. "
            f"Try increasing nodata_discard_threshold or disabling exclude_high_nodata_images."
        )

    return kept_paths, kept_labels


def _decode_and_prepare_image(path, params, training=False):
    image_bytes = tf.io.read_file(path)
    lowercase_path = tf.strings.lower(path)

    def _decode_png():
        png_u8 = tf.io.decode_png(image_bytes, channels=3, dtype=tf.uint8)
        png_u16 = tf.io.decode_png(image_bytes, channels=3, dtype=tf.uint16)
        png_u8_as_u16 = tf.cast(png_u8, tf.uint16)
        png_u8_promoted = png_u8_as_u16 * tf.constant(257, dtype=tf.uint16)
        is_promoted_u8 = tf.reduce_all(tf.equal(png_u16, png_u8_promoted))
        return tf.cond(is_promoted_u8, lambda: png_u8_as_u16, lambda: png_u16)

    def _decode_jpeg():
        return tf.cast(tf.io.decode_jpeg(image_bytes, channels=3), tf.uint16)

    def _decode_bmp():
        return tf.cast(tf.io.decode_bmp(image_bytes, channels=3), tf.uint16)

    def _decode_fallback():
        return tf.cast(tf.io.decode_image(image_bytes, channels=3, expand_animations=False), tf.uint16)

    image = tf.case(
        [
            (tf.strings.regex_full_match(lowercase_path, r'.*\.png'), _decode_png),
            (tf.strings.regex_full_match(lowercase_path, r'.*\.(jpg|jpeg)'), _decode_jpeg),
            (tf.strings.regex_full_match(lowercase_path, r'.*\.bmp'), _decode_bmp),
        ],
        default=_decode_fallback,
        exclusive=True,
    )

    source_dtype = image.dtype
    image = tf.cast(image, tf.float32)
    if source_dtype == tf.uint8:
        image = image / 255.0
    elif source_dtype == tf.uint16:
        scale = tf.cond(
            tf.reduce_max(image) > 255.0,
            lambda: tf.constant(65535.0, dtype=tf.float32),
            lambda: tf.constant(255.0, dtype=tf.float32),
        )
        image = image / scale
    else:
        image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.clip_by_value(image, 0.0, 1.0)

    target_height = int(params.image_height)
    target_width = int(params.image_width)
    crop_mode = getattr(params, "train_crop_mode", "resize") if training else getattr(params, "test_crop_mode", "resize")
    allow_variable = bool(getattr(params, "allow_variable_image_size", False) and not training)

    if crop_mode == "random":
        image = tf.image.resize_with_pad(
            image,
            tf.maximum(target_height, tf.shape(image)[0]),
            tf.maximum(target_width, tf.shape(image)[1]),
        )
        image = tf.image.random_crop(image, size=[target_height, target_width, 3])
    elif crop_mode == "center":
        image = tf.image.resize_with_crop_or_pad(image, target_height, target_width)
    elif crop_mode == "none" and allow_variable:
        pass
    else:
        image = tf.image.resize(image, [target_height, target_width], method="bilinear")

    if not allow_variable or crop_mode != "none":
        image.set_shape([target_height, target_width, 3])

    return image


def dataset_generator(dir, params, mode=None, shuffle=True):
    del mode

    image_paths = _list_image_files(dir)
    class_names, labels = _infer_class_names(image_paths, dir)
    split_name = Path(dir).parts[-2] if Path(dir).name == "images" and len(Path(dir).parts) >= 2 else Path(dir).name
    image_paths, labels = _apply_nodata_filter(image_paths, labels, params, split_name=split_name)

    auto = tf.data.AUTOTUNE
    is_training = shuffle

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(labels, dtype=tf.int32))
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        buffer_size = min(len(image_paths), int(getattr(params, "shuffle_buffer_size", 2048)))
        dataset = dataset.shuffle(buffer_size=max(buffer_size, 1), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: (_decode_and_prepare_image(path, params, training=is_training), label),
        num_parallel_calls=auto,
    )

    batch_size = int(params.batch_size)
    allow_variable = bool(getattr(params, "allow_variable_image_size", False) and not is_training)
    crop_mode = getattr(params, "test_crop_mode", "resize") if not is_training else getattr(params, "train_crop_mode", "resize")

    if allow_variable and crop_mode == "none":
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, None, 3], []),
            padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)),
            drop_remainder=False,
        )
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    dataset = dataset.prefetch(auto)
    dataset.class_names = class_names
    return dataset
