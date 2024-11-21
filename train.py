import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.utils import plot_model
from models.model import deepJSCC
from utils.datasets import dataset_generator

from config import config


def main():
    args = parse_args()
    check_gpu()

    print(config.architecture)
    print(config.arcChoice)

    # Configure dataset paths and dimensions based on dataset type
    if config.dataset == 'cifar10':
        print('Using CIFAR10 dataset')
        config.train_dir = './dataset/CIFAR10/train/'
        config.test_dir = './dataset/CIFAR10/test/'
        config.image_width = 32
        config.image_height = 32
        config.image_channels = 3
    elif config.dataset == 'eurosatrgb':
        print('Using Eurosat RGB dataset')
        config.train_dir = './dataset/EuroSAT_RGB_split/train/'
        config.test_dir = './dataset/EuroSAT_RGB_split/test/'
        config.image_width = 64
        config.image_height = 64
        config.image_channels = 3
    else:
        raise ValueError(f"{config.dataset} not accepted (check spelling)")

    # Print configuration variables
    print(f"Running experiment: {config.experiment_name}")
    for attr, value in vars(config).items():
        if not attr.startswith("__"):
            print(f"{attr} = {value}")

    # Prepare dataset
    train_ds, test_ds = prepare_dataset()

    # Initialize the model
    model = deepJSCC()

    # Display a few example images from the training dataset
    # display_image(train_ds)

    # PSNR as a custom metric
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)

    # Compile the model
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
        metrics=[psnr]
    )

    # Build the model and display a summary
    model.build(input_shape=(None, 64, 64, config.image_channels))
    model.summary()

    # Save model visualization
    plot_model(model, to_file='example_images/model_plot.png', show_shapes=True, show_layer_names=True)

    # Load checkpoint if provided
    if args.ckpt:
        model.load_weights(args.ckpt)

    # Define callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./ckpt/{config.experiment_name}_{{epoch}}",
        save_best_only=True,
        monitor="val_loss",
        save_weights_only=True,
        options=tf.train.CheckpointOptions(
            experimental_io_device=None, experimental_enable_async_checkpoint=True
        )
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{config.experiment_name}')

    # Train the model
    history = model.fit(
        train_ds,
        initial_epoch=config.initial_epoch,
        epochs=config.epochs,
        callbacks=[tensorboard_callback, checkpoint_callback],
        validation_data=test_ds,
    )

    # Save the model weights
    model.save_weights(f'models/saved_models/{config.experiment_name}_{config.epochs}.h5')


def check_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare_dataset():
    """
    Prepares the dataset using the paths and configurations defined in config.py.
    """
    AUTO = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator(config.test_dir, config)
    train_ds = dataset_generator(config.train_dir, config).cache()

    # Display class names
    print(test_ds.class_names)

    # Data augmentation and normalization
    augment_layer = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomFlip("horizontal"),
    ])

    normalize = tf.keras.layers.Rescaling(1. / 255)

    def normalize_and_augment(image, training):
        return augment_layer(image, training=training)

    # Shuffle, augment, and map datasets
    train_ds = (
        train_ds.shuffle(50000, reshuffle_each_iteration=True)
        .map(lambda x, y: (normalize_and_augment(x, training=True), x), num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    test_ds = (
        test_ds.map(lambda x, y: (normalize(x), x))
        .cache()
        .prefetch(AUTO)
    )

    return train_ds, test_ds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--ckpt', type=str, help='Checkpoint file')
    return parser.parse_args()


def display_image(dataset):
    """
    Save example images from the dataset to disk.
    """
    for images, _ in dataset.take(1):  # Take one batch from the dataset
        for i in range(9):
            example_image = images[i] * 255.0
            example_image = example_image.numpy().astype(np.uint8)  # Convert to numpy array
            img = Image.fromarray(example_image)  # Create an Image object
            filename = f'example_images/example_image_{i}.png'
            img.save(filename)  # Save the image to disk


if __name__ == "__main__":
    main()