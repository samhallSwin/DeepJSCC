import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
from config import config
from utils import image_proc

from tensorflow.keras.utils import plot_model

from models.model import deepJSCC

from utils.datasets import dataset_generator

def main():
    args = parse_args() #TODO read other config file 
    check_gpu()

    #Set dataset relevant params in config.[param]
    config.setImageParamsFromDataset()

    print(f"Image size {config.image_width} X {config.image_height} X {config.image_channels}")

    train_ds, test_ds = prepare_dataset(config)

    print(f'Running {config.experiment_name}')

    
    model = deepJSCC(
        input_size = config.image_width,
        has_gdn=config.has_gdn,
        num_symbols=config.data_size,
        snrdB=config.train_snrdB,
        channel=config.channel_type
    )
    
    if config.workflow == "train": 
        train_model(model, config, train_ds, test_ds, args)
    elif config.workflow == "loadAndTest": 
        load_and_analyse(model, config, train_ds, test_ds, args)
    else:
        print('No valid workflow detected (Check spelling)')
    
    #model.save('models/' + f"{config.experiment_name}_" + f"{config.epochs}" + '.h5')

def train_model(model, config, train_ds, test_ds, args):
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=1e-4
        ),
        metrics=[
            psnr
        ]
    )

    model.build(input_shape=(None, config.image_width, config.image_height, config.image_channels))
    model.summary()

    plot_model(model, to_file='example_images/model_plot.png', show_shapes=True, show_layer_names=True)
    
    if args.ckpt is not None:
        model.load_weights(args.ckpt)

    save_ckpt = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=f"./ckpt/{config.experiment_name}_" + "{epoch}",
          save_best_only=True,
          monitor="val_loss",
          save_weights_only=True,
          options=tf.train.CheckpointOptions(
              experimental_io_device=None, experimental_enable_async_checkpoint=True
          )
      )
  ]

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{config.experiment_name}')
    history = model.fit(
        train_ds,
        initial_epoch=config.initial_epoch,
        epochs=config.epochs,
        callbacks=[tensorboard, save_ckpt],
        validation_data=test_ds,
    )

    model.save_weights('models/saved_models/' + f"{config.experiment_name}_" + f"{config.epochs}" + '.h5')

def load_and_analyse(model, config, train_ds, test_ds, args):
    input_shape = (config.image_width,config.image_height,config.image_channels)
    dummy_input = np.zeros((1,*input_shape))
    model(dummy_input)

    model.load_weights(f'models/saved_models/{config.modelFile}')

    model.summary()

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=1e-4
        ),
        metrics=[
            psnr
        ]
    )

    output_dir = "example_images"
    num_images = 8
    os.makedirs(output_dir, exist_ok=True)

    original_images, processed_images, original_sizes, processed_sizes = image_proc.process_and_save_images(train_ds, model, output_dir, num_images=num_images)
    output_path = "example_images/visualization.png"
    image_proc.visualize_and_save_images(original_images, processed_images, original_sizes, processed_sizes, output_path, num_images=num_images)
    print(f'8 random images captured to /{output_dir}')




def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)

def check_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare_dataset(config):
    AUTO = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator(config.test_dir, config)
    train_ds = dataset_generator(config.train_dir, config).cache()

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