import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser

from models.model import deepJSCC

from utils.datasets import dataset_generator

def main():
    args = parse_args()
    check_gpu()

    params = read_config(args.config_file, args.config_name)

    for key, value in vars(params).items():
        print(f"{key} = {value}, Type: {type(value)}")

    train_ds, test_ds = prepare_dataset(params.batch_size)

    EXPERIMENT_NAME = params.experiment_name
    print(f'Running {EXPERIMENT_NAME}')

    
    model = deepJSCC(
        has_gdn=params.has_gdn,
        num_symbols=params.data_size,
        snrdB=params.train_snrdb,
        channel=params.channel_type
    )
    #display_image(train_ds)

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)
  
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=1e-4
        ),
        metrics=[
            psnr
        ]
    )

    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    
    if args.ckpt is not None:
        model.load_weights(args.ckpt)

    save_ckpt = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=f"./ckpt/{EXPERIMENT_NAME}_" + "{epoch}",
          save_best_only=True,
          monitor="val_loss",
          save_weights_only=True,
          options=tf.train.CheckpointOptions(
              experimental_io_device=None, experimental_enable_async_checkpoint=True
          )
      )
  ]

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{EXPERIMENT_NAME}')
    history = model.fit(
        train_ds,
        initial_epoch=params.initial_epoch,
        epochs=params.epochs,
        callbacks=[tensorboard, save_ckpt],
        validation_data=test_ds,
    )

    model.save_weights(f"{EXPERIMENT_NAME}_" + f"{params.epochs}")

def check_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare_dataset(BATCH_SIZE):
    AUTO = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator('./dataset/CIFAR10/test/', BATCH_SIZE = BATCH_SIZE)
    train_ds = dataset_generator('./dataset/CIFAR10/train/', BATCH_SIZE = BATCH_SIZE).cache()

    class_names = test_ds.class_names
    print(class_names) 

    for images, labels in train_ds.take(1):
        print(images.shape)  # (batch_size, img_height, img_width, 3)
        print(labels.shape)  # (batch_size,)

    normalize = tf.keras.layers.Rescaling(1./255)
    augment_layer = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
    ])

    def normalize_and_augment(image, training):
        image = augment_layer(image, training=training)
        return image
    
    #shuffle, augment and discard labels to replace them with the image data for autoenc
    train_ds = (
    train_ds.shuffle(50000, reshuffle_each_iteration=True)
            .map(lambda x, y: (normalize_and_augment(x, training=True), y), num_parallel_calls=AUTO)
            .map(lambda x, _: (x, x))
            .prefetch(AUTO)
    )

    test_ds = (
    test_ds.map(lambda x, y: (normalize(x), y))
            .map(lambda x, _: (x, x))
            .cache()
            .prefetch(AUTO)
    )

    return train_ds, test_ds

def read_config(config_file, config_name):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if config_name in config:
        params = type('Params', (object,), {})()
        for key, value in config[config_name].items():
            # Attempt to convert the value to an integer
            try:
                int_value = int(value)
                setattr(params, key, int_value)
            except ValueError:
                setattr(params, key, value)
        return params
    else:
        raise ValueError(f"Configuration '{config_name}' not found in {config_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",'--config_name', type=str, default='default',help='use config within config file'),
    parser.add_argument("-c",'--config_file', type=str, default='config/config.txt',help='config file location'),
    parser.add_argument("-p",'--ckpt', type=str,help='checkpoint file')

    return parser.parse_args()

def display_image(dataset):
    #denormalise and save example image to disk
    
    i=0
    for images, labels in dataset.take(1):  # Take one batch from the dataset
        for i in range(9):
            example_image = images[i] * 255.0
            example_image = example_image.numpy().astype(np.uint8)  # Convert to numpy array
            img = Image.fromarray(example_image)  # Create an Image object
            filename = 'example_images/example_image_' + str(i) + '.png' 
            img.save(filename)  # Save the image to disk
            
    


if __name__ == "__main__":
    main()