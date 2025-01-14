import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from PIL import Image
import numpy as np
from config import configOverride
from config import config
#from analysis import attmaps
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.utils import plot_model

from models.model import deepJSCC
from models.loss_functions import sobel_edge_loss, combined_loss, perceptual_loss_with_extractor, init_perceptual_loss, gradient_loss, combined_loss_verbose, LossWeightScheduler, ssim_loss
import tests

from utils.datasets import dataset_generator

def main():
    args = configOverride.parse_args() #override config with args

    configOverride.override_config_with_args(config, args)

    settings_file = os.path.join('logs', f"{config.experiment_name}_settings.txt")
    save_config_to_file(config, settings_file)

    check_gpu()

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    #Set dataset relevant params in config.[param]
    config.setImageParamsFromDataset()
    enc, dec = config.setArc()

    print(enc)
    print(dec)

    print(f"Image size {config.image_width} X {config.image_height} X {config.image_channels}")

    train_ds, test_ds = prepare_dataset(config)

    #Stores the model shape and attributes
    model_shape_file = f"logs/{config.experiment_name}_modelShape.txt"
    if os.path.exists(model_shape_file):
        os.remove(model_shape_file)
        print(f"Deleted existing debug log file: {model_shape_file}")

    print(f'Running {config.experiment_name}')


    with strategy.scope():
        model = deepJSCC(
            input_size = config.image_width,
            has_gdn=config.has_gdn,
            num_symbols=config.num_symbols,
            snrdB=config.train_snrdB,
            set_channel_filters = config.set_channel_filters,
            channel_filters=config.channel_filters,
            encoder_config=enc,
            decoder_config=dec,
            channel=config.channel_type,
            debug_file=model_shape_file
        )
        
        if config.workflow == "train": 
            train_model(model, config, train_ds, test_ds, args)
        elif config.workflow == "loadAndTest": 
            load_and_analyse(model, config, train_ds, test_ds, args)
        else:
            print('No valid workflow detected (Check spelling)')
    

def train_model(model, config, train_ds, test_ds, args):

    compile_model(model)

    model.build(input_shape=(None, config.image_width, config.image_height, config.image_channels))
    model.summary()

    #plot_model(model, to_file='outputs/example_images/model_plot.png', show_shapes=True, show_layer_names=True)
    
    if args.checkpoint_filepath is not None:
        model.load_weights(args.checkpoint_filepath)

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
    
    #if we're changing loss function weighting between epochs we need an extra callback  
    if config.loss_func == 'combined_schedule':
        dynamic_weights = [tf.Variable(1.0), tf.Variable(0.0)] 
        loss_weight_scheduler = LossWeightScheduler(dynamic_weights, config.loss_schedule)

        history = model.fit(
            train_ds,
            initial_epoch=config.initial_epoch,
            epochs=config.epochs,
            callbacks=[tensorboard, save_ckpt, loss_weight_scheduler],
            validation_data=test_ds,
        )
    else:
         history = model.fit(
            train_ds,
            initial_epoch=config.initial_epoch,
            epochs=config.epochs,
            callbacks=[tensorboard, save_ckpt],
            validation_data=test_ds,
        )       

    save_dir = 'models/saved_models/'
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, f"{config.experiment_name}_{config.epochs}.h5")
    model.save_weights(save_dir)
    print(f'Model saved to {save_dir}')

def load_and_analyse(model, config, train_ds, test_ds, args):
    input_shape = (config.image_width,config.image_height,config.image_channels)
    dummy_input = np.zeros((1,*input_shape))
    model(dummy_input)

    model.load_weights(f'models/saved_models/{config.modelFile}')

    model.summary()

    compile_model(model)


    if "validate_model" in config.TESTS_TO_RUN:
        tests.validate_model(model, test_ds)
    
    if "save_latent" in config.TESTS_TO_RUN:
        tests.save_latent(model, test_ds, config)

    if "compare_to_BPG_LDPC" in config.TESTS_TO_RUN:
        tests.compare_to_BPG_LDPC(model, test_ds, train_ds, config)

    if "compare_to_JPEG2000" in config.TESTS_TO_RUN:
        tests.compare_to_JPEG2000(model, test_ds, train_ds, config)

    if "time_analysis" in config.TESTS_TO_RUN:
        tests.time_analysis(model)

    if "process_All_SNR" in config.TESTS_TO_RUN:
        tests.process_All_SNR(model, test_ds, train_ds, config)

    if "process_images_through_channel" in config.TESTS_TO_RUN:
        tests.process_images_through_channel(model, test_ds, config, num_images=8, snr_range=(-10, 20))
    
    if "process_random_image_at_snrs" in config.TESTS_TO_RUN:
        tests.process_random_image_at_snrs(model, test_ds, num_images=5, snr_range=(-20, 20), step=5, save_dir="outputs/channel_state_est/")

    if "hacky_tests" in config.TESTS_TO_RUN:
        tests.hacky_tests(model, test_ds)


def compile_model(model):
    #Primary purpose is to run model.compile at the bottom of the function, the rest is here to set up the loss function we want

    print(f'Compiling model with {config.loss_func} loss function')

    if config.loss_func == 'mse':  # Mean Squared Error
        loss = 'mse'
    elif config.loss_func == 'perceptual_loss':  # Perceptual loss using VGG16

        init_perceptual_loss()
        loss = perceptual_loss_with_extractor

    elif config.loss_func == 'sobel_edge_loss':  
        loss = sobel_edge_loss
    elif config.loss_func == 'ssim_loss':  
        loss = ssim_loss
    elif config.loss_func == 'gradient_loss':  
        loss = gradient_loss
    elif (config.loss_func == 'combined') or (config.loss_func == 'combined_loss_verbose'):  # Combine losses with weights
        loss_functions = []
        weights = []
        
        if 'mse' in config.combined_loss_weights:
            loss_functions.append(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE))
            weights.append(config.combined_loss_weights['mse'])
        
        if 'perceptual_loss' in config.combined_loss_weights:
            init_perceptual_loss()
            loss = perceptual_loss_with_extractor
            
            loss_functions.append(perceptual_loss_with_extractor)
            weights.append(config.combined_loss_weights['perceptual_loss'])
        
        if 'sobel_edge_loss' in config.combined_loss_weights:
            loss_functions.append(sobel_edge_loss)
            weights.append(config.combined_loss_weights['sobel_edge_loss'])
        
        if 'ssim_loss' in config.combined_loss_weights:
            loss_functions.append(ssim_loss)
            weights.append(config.combined_loss_weights['ssim_loss'])
        
        if 'gradient_loss' in config.combined_loss_weights:
            loss_functions.append(gradient_loss)
            weights.append(config.combined_loss_weights['gradient_loss'])
        
        if (config.loss_func == 'combined'): 
            loss = combined_loss(loss_functions, weights)
        else:
            loss = combined_loss_verbose(loss_functions, weights)

    elif config.loss_func == 'combined_schedule': 
        dynamic_weights = [tf.Variable(1.0), tf.Variable(0.0)] 
        loss_functions = [
            tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            sobel_edge_loss
        ]
        loss = combined_loss(loss_functions, dynamic_weights)

    else:
        raise ValueError(f"Unsupported loss function: {config.loss_func}")
    
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
        metrics=[
            psnr,
            ssim_metric
        ]
    )

def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)

def ssim_metric(y_true, y_pred):
    """
    Compute the mean SSIM across a batch.
    """
    ssim_values = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(ssim_values)

def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Num GPUs Available: {len(physical_devices)}")
    else:
        print("No GPU available. Running on CPU.")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set successfully for all GPUs.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")


##Replace 'mse' in model.compile with this
#def perceptual_loss(y_true, y_pred):
#    """
#    Calculate perceptual loss by comparing features extracted from a VGG16 model.
#    """
#    # Resize input to match VGG input size
#    y_true_resized = tf.image.resize(y_true, (224, 224))
#    y_pred_resized = tf.image.resize(y_pred, (224, 224))
#
#    # Extract features and compute mean squared error
#    true_features = feature_extractor(y_true_resized)
#    pred_features = feature_extractor(y_pred_resized)
#    return tf.reduce_mean(tf.square(true_features - pred_features))

def prepare_dataset(config):
    AUTO = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator(config.test_dir, config)
    train_ds = dataset_generator(config.train_dir, config).cache()

    class_names = test_ds.class_names
    print(class_names) 

    print(f'Length of dataset (batches) = {len(train_ds)}')
    for images, labels in train_ds.take(1):
        print(f'Shape of dataset{images.shape}')  # (batch_size, img_height, img_width, 3)
        print(labels.shape)  # (batch_size,)

    #    train_to_pad = train_ds
    #test_to_pad = test_ds

    if config.image_height == 28:
        train_ds = train_ds.map(pad_images)
        test_ds = test_ds.map(pad_images)
        config.image_width = 32
        config.image_height = 32
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

# Add padding transformation
def pad_images(images, labels):
    # Pad the images to make them 32x32
    images = tf.pad(images, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
    return images, labels



def save_config_to_file(config, filename):
    """
    Save user-defined config settings to a file, excluding built-ins and special attributes.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
    
    with open(filename, 'w') as f:
        f.write("Experiment Configuration Settings\n")
        f.write("-" * 40 + "\n")
        
        # Filter out built-in attributes and functions
        for attr in dir(config):
            if not attr.startswith("__") and not callable(getattr(config, attr)):
                value = getattr(config, attr)
                f.write(f"{attr} = {value}\n")
    
    print(f"Configuration settings saved to {filename}")

def display_image(dataset):
    #denormalise and save example image to disk
    
    i=0
    for images, labels in dataset.take(1):  # Take one batch from the dataset
        for i in range(9):
            example_image = images[i] * 255.0
            example_image = example_image.numpy().astype(np.uint8)  # Convert to numpy array
            img = Image.fromarray(example_image)  # Create an Image object
            filename = 'outputs/example_images/example_image_' + str(i) + '.png' 
            img.save(filename)  # Save the image to disk
            

if __name__ == "__main__":
    main()