import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

import tensorflow as tf
from PIL import Image

from models.loss_functions import (
    LossWeightScheduler,
    combined_loss,
    combined_loss_verbose,
    gradient_loss,
    init_perceptual_loss,
    perceptual_loss_with_extractor,
    sobel_edge_loss,
    ssim_loss,
)
from models.model import deepJSCC
from utils.datasets import dataset_generator


def initialize_run(config, args, save_settings=True):
    if save_settings:
        settings_file = os.path.join("logs", f"{config.experiment_name}_settings.txt")
        save_config_to_file(config, settings_file)

    check_gpu()

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    config.setImageParamsFromDataset()
    enc, dec = config.setArc()

    print(enc)
    print(dec)
    print(f"Image size {config.image_width} X {config.image_height} X {config.image_channels}")

    train_ds, test_ds = prepare_dataset(config)

    model_shape_file = os.path.join("logs", f"{config.experiment_name}_modelShape.txt")
    if os.path.exists(model_shape_file):
        os.remove(model_shape_file)
        print(f"Deleted existing debug log file: {model_shape_file}")

    print(f"Running {config.experiment_name}")

    return strategy, train_ds, test_ds, model_shape_file


def build_model(config, model_shape_file):
    enc, dec = config.setArc()
    return deepJSCC(
        input_size=config.image_width,
        has_gdn=config.has_gdn,
        num_symbols=config.num_symbols,
        snrdB=config.train_snrdB,
        set_channel_filters=config.set_channel_filters,
        channel_filters=config.channel_filters,
        encoder_config=enc,
        decoder_config=dec,
        channel=config.channel_type,
        debug_file=model_shape_file,
        use_snr_side_info=config.use_snr_side_info,
        film_hidden_units=config.film_hidden_units,
        rician_k_factor=config.rician_k_factor,
    )


def compile_model(model, config):
    print(f"Compiling model with {config.loss_func} loss function")

    if config.loss_func == "mse":
        loss = "mse"
    elif config.loss_func == "perceptual_loss":
        init_perceptual_loss()
        loss = perceptual_loss_with_extractor
    elif config.loss_func == "sobel_edge_loss":
        loss = sobel_edge_loss
    elif config.loss_func == "ssim_loss":
        loss = ssim_loss
    elif config.loss_func == "gradient_loss":
        loss = gradient_loss
    elif config.loss_func in {"combined", "combined_loss_verbose"}:
        loss_functions = []
        weights = []

        if "mse" in config.combined_loss_weights:
            loss_functions.append(
                tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.NONE
                )
            )
            weights.append(config.combined_loss_weights["mse"])

        if "perceptual_loss" in config.combined_loss_weights:
            init_perceptual_loss()
            loss_functions.append(perceptual_loss_with_extractor)
            weights.append(config.combined_loss_weights["perceptual_loss"])

        if "sobel_edge_loss" in config.combined_loss_weights:
            loss_functions.append(sobel_edge_loss)
            weights.append(config.combined_loss_weights["sobel_edge_loss"])

        if "ssim_loss" in config.combined_loss_weights:
            loss_functions.append(ssim_loss)
            weights.append(config.combined_loss_weights["ssim_loss"])

        if "gradient_loss" in config.combined_loss_weights:
            loss_functions.append(gradient_loss)
            weights.append(config.combined_loss_weights["gradient_loss"])

        if config.loss_func == "combined":
            loss = combined_loss(loss_functions, weights)
        else:
            loss = combined_loss_verbose(loss_functions, weights)
    elif config.loss_func == "combined_schedule":
        dynamic_weights = [tf.Variable(1.0), tf.Variable(0.0)]
        loss_functions = [
            tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            sobel_edge_loss,
        ]
        loss = combined_loss(loss_functions, dynamic_weights)
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_func}")

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
        metrics=[psnr, ssim_metric],
    )


def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)


def ssim_metric(y_true, y_pred):
    ssim_values = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(ssim_values)


def check_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Num GPUs Available: {len(physical_devices)}")
    else:
        print("No GPU available. Running on CPU.")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set successfully for all GPUs.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")


def prepare_dataset(config):
    auto = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator(config.test_dir, config)
    train_ds = dataset_generator(config.train_dir, config).cache()

    class_names = test_ds.class_names
    print(class_names)

    print(f"Length of dataset (batches) = {len(train_ds)}")
    for images, labels in train_ds.take(1):
        print(f"Shape of dataset{images.shape}")
        print(labels.shape)

    if config.image_height == 28:
        train_ds = train_ds.map(pad_images)
        test_ds = test_ds.map(pad_images)
        config.image_width = 32
        config.image_height = 32
        for images, labels in train_ds.take(1):
            print(images.shape)
            print(labels.shape)

    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    augment_layer = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
        ]
    )

    def normalize_and_augment(image, training):
        return augment_layer(image, training=training)

    def add_random_snr(images, targets):
        snr_min, snr_max = config.train_snr_range
        batch_size = tf.shape(images)[0]
        snr = tf.random.uniform(
            (batch_size,), minval=snr_min, maxval=snr_max, dtype=tf.float32
        )
        return (images, snr), targets

    def add_fixed_snr(images, targets):
        batch_size = tf.shape(images)[0]
        snr = tf.fill((batch_size,), tf.cast(config.train_snrdB, tf.float32))
        return (images, snr), targets

    train_ds = (
        train_ds.shuffle(50000, reshuffle_each_iteration=True)
        .map(
            lambda x, y: (normalize_and_augment(x, training=True), y),
            num_parallel_calls=auto,
        )
        .map(lambda x, _: (x, x))
        .prefetch(auto)
    )

    test_ds = (
        test_ds.map(lambda x, y: (normalize(x), y))
        .map(lambda x, _: (x, x))
        .cache()
        .prefetch(auto)
    )

    if config.use_snr_side_info:
        if config.random_snr_training:
            train_ds = train_ds.map(add_random_snr, num_parallel_calls=auto)
        else:
            train_ds = train_ds.map(add_fixed_snr, num_parallel_calls=auto)
        test_ds = test_ds.map(add_fixed_snr, num_parallel_calls=auto)

    return train_ds, test_ds


def pad_images(images, labels):
    images = tf.pad(images, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
    return images, labels


def save_config_to_file(config, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        f.write("Experiment Configuration Settings\n")
        f.write("-" * 40 + "\n")

        for attr in dir(config):
            if not attr.startswith("__") and not callable(getattr(config, attr)):
                value = getattr(config, attr)
                f.write(f"{attr} = {value}\n")

    print(f"Configuration settings saved to {filename}")


def display_image(dataset):
    for images, labels in dataset.take(1):
        for i in range(9):
            example_image = images[i] * 255.0
            example_image = example_image.numpy().astype("uint8")
            img = Image.fromarray(example_image)
            filename = f"outputs/example_images/example_image_{i}.png"
            img.save(filename)
