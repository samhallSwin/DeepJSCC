
experiment_name = "Decrease_channel_filt" #effects file locations. Will overwrite previous with same name for the most part
workflow = "train" #train, loadAndTest
checkpoint_filepath = ""

modelFile = 'decreasing_filters_test_20.h5' #used for loading model for testing. Must be in /models/saved_models/

#training params
batch_size = 32
epochs = 20
train_snrdB = 10
num_symbols = 512
initial_epoch = 0

set_channel_filters = True #Should the number of params be set by channel_filters (True) or calculated? 
channel_filters = 32

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

#For processing for channel state est testing
snr_range=(-10, 15)

#add or remove tests (in tests.py) 
# WARNING: Tests should work independantly but have not been properly tested running sequentially
TESTS_TO_RUN = [
    "validate_model",
    #"time_analysis",
    "compare_to_BPG_LDPC",
    #"compare_to_JPEG2000",
    #'process_All_SNR',
    'process_random_image_at_snrs',
    #'process_images_through_channel',
    #'save_latent',
]

#Must match entry in setImageParamsFromDataset()
dataset = "eurosatrgb" #eurosatrgb, CIFAR10, OV_MNIST

#A list of mdoel configs that can be selected by arc_choice. Must match image dimensions

arc_choice = 'Filter_decrease'# neive_64, original, reduced_filters_64, Filter_decrease, original_64 

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

#Sliming filters progressively
#input image 64x64x3
encoder_config_Filter_decrease = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 128, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 64, "kernel_size": 5, "stride": 1, "block_type": "C"},
    {"filters": 32, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_Filter_decrease =  [
    {"filters": 32, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 64, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 128, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},

]

#Original but for 64x64
#input image 64x64x3
encoder_config_original_64 = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_original_64  = [
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},

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
    elif arc_choice == 'Filter_decrease':
        enc = encoder_config_Filter_decrease
        dec = decoder_config_Filter_decrease
    elif arc_choice == 'original_64':
        enc = encoder_config_original_64
        dec = decoder_config_original_64
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
