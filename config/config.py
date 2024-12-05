
experiment_name = "Eurosat_128_filt_1024_data_size" #effects file locations. Will overwrite previous with same name for the most part
workflow = "loadAndTest" #train, loadAndTest
#checkpoint_filepath = "checkpoint.ckpt"

modelFile = 'Eurosat_rgb_26_11_10.h5' #used for loading model for testing. Must be in /models/saved_models/

#training params
batch_size = 32
epochs = 2
train_snrdB = -30
num_symbols = 512
initial_epoch = 0

#Architecture params
has_gdn = True
channel_type = "AWGN" #Rayleigh, AWGN, Rician, None
loss_func = 'mse' #mse, perceptual_loss


#Trad compression test features for comparison
#bw_ratio = [1/12, 1/6, 1/4, 1/3, 1/2]
#snrs = [0, 10]
#mcs = [(k, n, m) for k, n in [(3072, 6144), (3072, 4608), (1536, 4608)] for m in (2, 4, 16, 64)]
LDPCon = True
bw_ratio = 1 / 6
mcs = (3072, 6144, 2)

#add or remove tests (in tests.py)
TESTS_TO_RUN = [
    #"validate_model",
    #"time_analysis",
    #"compare_to_BPG_LDPC",
    #"compare_to_JPEG2000",
    'save_latent',
]

#Must match entry in setImageParamsFromDataset()
dataset = "eurosatrgb" #eurosatrgb, CIFAR10, OV_MNIST



#A list of mdoel configs that can be selected by arc_choice. Must match image dimensions

arc_choice = 'neive_64'# neive_64, original, reduced_filters_64

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

#First attempt for eurosat data
#input image 64x64x3
encoder_reduced_filters_64 = [
    {"filters": 128, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 128, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C"},
    #{"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_reduced_filters_64 =  [
    {"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 128, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
    #{"filters": 128, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},

]


def setArc():
    if arc_choice == 'original':
        enc = encoder_config_original
        dec = decoder_config_original
    elif arc_choice == 'neive_64':
        enc = encoder_config_neive_64
        dec = decoder_config_neive_64
    elif arc_choice == 'reduced_filters_64':
        enc = encoder_reduced_filters_64
        dec = decoder_reduced_filters_64
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
