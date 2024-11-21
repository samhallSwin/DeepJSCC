
experiment_name = "New_CIFAR" #effects file locations. Will overwrite previous with same name for the most part
workflow = "train" #train, loadAndTest
checkpoint_filepath = "checkpoint.ckpt"

modelFile = 'New_CIFAR.h5' #used for loading model for testing. Must be in /models/saved_models/

#training params
batch_size = 32
epochs = 15
train_snrdB = 15
data_size = 512
initial_epoch = 0

#Architecture params
has_gdn = True
channel_type = "AWGN" #Rayleigh, AWGN, Rician, None
loss_func = 'mse' #mse, perceptual_loss(NOT WORKING)


#Must match entry in setImageParamsFromDataset()
dataset = "CIFAR10" #eurosatrgb, CIFAR10, OV_MNIST



#A list of mdoel configs that can be selected by arc_choice. Must match image dimensions

arc_choice = 'original'# neive_64, original

#Original config from paper
#input image 32x32x3
encoder_config_original = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_original  = [
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},

]

#First attempt for eurosat data
#input image 64x64x3
encoder_config_neive_64 = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_neive_64 =  [
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},

]

def setArc():
    if arc_choice == 'original':
        enc = encoder_config_original
        dec = decoder_config_original
    elif arc_choice == 'neive_64':
        enc = encoder_config_neive_64
        dec = decoder_config_neive_64
    else:
        print('architecture not found (check spelling)')

    return enc, dec

def setImageParamsFromDataset():
    #Set image params based on dataset
    global train_dir, test_dir, image_width, image_height, image_channels
    if dataset == "CIFAR10":
        train_dir = './dataset/CIFAR10/train/'
        test_dir = './dataset/CIFAR10/test/'
        image_width = 32
        image_height = 32
        image_channels =3
        print('CIFAR10 dataset selected')
    elif dataset == "eurosatrgb":
        train_dir = './dataset/EuroSAT_RGB_split/train/'
        test_dir = './dataset/EuroSAT_RGB_split/test/'
        image_width = 64
        image_height = 64
        image_channels = 3
        print('eurosat_RGB dataset selected')
    elif dataset == "OV_MNIST":
        train_dir = './dataset/OV-MNIST/training/'
        test_dir = './dataset/OV-MNIST/testing/'
        image_width = 28
        image_height = 28
        image_channels = 3
        print('eurosat_RGB dataset selected')
    else:
        print('No supported dataset - check spelling')
