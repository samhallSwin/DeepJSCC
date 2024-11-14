import tensorflow as tf
from models.model import deepJSCC
import configparser
import numpy as np
import matplotlib.pyplot as plt
from utils.datasets import dataset_generator

def main():

    single_image_pass = True

    params = read_config('config/config.txt', 'default')

    if params.dataset == 'cifar10':
        print('Using CIFAR10 dataset')
        setattr(params,'train_dir', './dataset/CIFAR10/train/')
        setattr(params,'test_dir', './dataset/CIFAR10/test/')
        setattr(params,'image_width', 32)
        setattr(params,'image_height', 32)
        setattr(params,'image_channels', 3)
    elif params.dataset == 'eurosatrgb':
        print('Using Eurosat RGB dataset')
        setattr(params,'train_dir', './dataset/EuroSAT_RGB_split/train/')
        setattr(params,'test_dir', './dataset/EuroSAT_RGB_split/test/')
        setattr(params,'image_width', 64)
        setattr(params,'image_height', 64)
        setattr(params,'image_channels', 3)
    else:
        print(params.dataset + ' not accepted (check spelling)')

    for key, value in vars(params).items():
        print(f"{key} = {value}, Type: {type(value)}")

    model = deepJSCC(
        input_size = params.image_width,
        has_gdn=params.has_gdn,
        num_symbols=params.data_size,
        snrdB=10,
        channel=params.channel_type
    )

    input_shape = (params.image_width,params.image_height,params.image_channels)
    dummy_input = np.zeros((1,*input_shape))
    model(dummy_input)

    model.load_weights('models/eurosat_RGB_test_10.h5')


    model.summary()


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

    

    if single_image_pass:
        image_path = 'dataset/EuroSAT_RGB_split/test/Industrial/Industrial_117.jpg'
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        #image = tf.image.resize(image, [28, 28])  # Resize to match your autoencoder input size
        image = tf.cast(image, tf.float32)
        image = image / 255.0  # Normalize the image
    
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        reconstructed_image = model(image)


        print(reconstructed_image.shape)
        
        save_image(image, 'example_images/input_image.jpg')
        

        save_image(reconstructed_image, 'example_images/output_image.jpg')
        
    else:
        train_ds, test_ds = prepare_dataset(params)
        loss = model.evaluate(test_ds)
        print(f"Loss: {loss}")   


    

def prepare_dataset(params):
    AUTO = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator(params.test_dir, params)
    train_ds = dataset_generator(params.train_dir, params).cache()

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
    

def save_image(image, filename):
    image = tf.squeeze(image)  # Remove batch dimension
    image = tf.clip_by_value(image, 0.0, 1.0)  # Ensure pixel values are in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.uint8)
    encoded_image = tf.io.encode_jpeg(image)
    tf.io.write_file(filename, encoded_image)


def display_image(image):
    #denormalise and save example image to disk
    
            example_image = image * 255.0
            example_image = example_image.numpy().astype(np.uint8)  # Convert to numpy array
            img = Image.fromarray(example_image)  # Create an Image object
            filename = 'example_images/in_image.png' 
            img.save(filename)  # Save the image to disk
            


if __name__ == "__main__":
    main()