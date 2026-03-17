
experiment_name = "film_test_20_Proper" #effects file locations. Will overwrite previous with same name for the most part
workflow = "loadAndTest" #train, loadAndTest
checkpoint_filepath = ""

modelFile = 'film_test_20_Proper_20.h5' #used for loading model for testing. Must be in /models/saved_models/

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
rician_k_factor = 2.0
loss_func = 'combined_schedule' #mse, perceptual_loss, sobel_edge_loss, combined, gradient_loss, combined_loss_verbose, combined_schedule, ssim_loss
use_snr_side_info = True
film_hidden_units = 64
random_snr_training = True
train_snr_range = (-10.0, 10.0)

#If loss_func=combined, set relative amounts here. Comment out unsused elements (ie. don't set them to zero)
combined_loss_weights = {
    'mse': 1.0,
    #'sobel_edge_loss': 0.1,
    #'gradient_loss': 0.5,
}

#if loss_func=combined_schedule, this sets the relative loss weights in the form epoch: [loss1 weight, loss2 weight]
#Currently only works for 2 loss functions
loss_schedule = {
        0: [1.0, 0.0],
        10: [0.5, 0.5],  
        15: [0.2, 0.8],  
    }

#Trad compression test features for comparison
#bw_ratio = [1/12, 1/6, 1/4, 1/3, 1/2]
#snrs = [0, 10]
#mcs = [(k, n, m) for k, n in [(3072, 6144), (3072, 4608), (1536, 4608)] for m in (2, 4, 16, 64)]
LDPCon = True
bw_ratio = 1 / 6
mcs = (3072, 6144, 2)
adaptive_bpg_ldpc = True
adaptive_mcs_table = [
    (-100, (3072, 6144, 2)),
    (2, (3072, 4608, 2)),
    (6, (3072, 4608, 4)),
    (10, (1536, 4608, 4)),
]

#CLIP settings if running CLIP tests
enable_clip_metric = True
clip_device = "cpu" 
clip_model_name = "ViT-B/32"
num_semantic_eval_images = 8

#Downstream task metric settings
enable_downstream_metric = True
downstream_model_id = "cm93/resnet18-eurosat"
downstream_device = "cpu"

#For processing for channel state est testing
snr_range=(-20, 10)
snr_eval_step = 1
num_snr_eval_images = 32
snr_sweep_output_dir = "outputs/snr_sweep_rician_film3"
num_visual_eval_images = 8
visual_eval_output_dir = "outputs/visual_eval"
bpg_ldpc_eval_output_dir = "outputs/bpg_ldpc_eval_film3"

#add or remove tests (in tests.py) 
# WARNING: Tests should work independantly but have not been properly tested running sequentially
TESTS_TO_RUN = [
    #"compare_to_BPG_LDPC",
    "compare_to_BPG_LDPC_sweep",
]

#Must match entry in setImageParamsFromDataset()
dataset = "eurosatrgb" #eurosatrgb, CIFAR10, OV_MNIST

#A list of mdoel configs that can be selected by arc_choice. Must match image dimensions

arc_choice = 'neive_64'# neive_64, original, reduced_filters_64, Filter_decrease, original_64 

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
