# ###################################
# 
# [USED FOR BOTH TRAINING AND TESTING]
# 
# ###################################
experiment_name = "split_test_smoke"  # Used in artifact/log file names.
batch_size = 32
train_snrdB = 10
num_symbols = 512

# model and channel parameters
set_channel_filters = True  # If False, channel filter count is inferred from num_symbols.
channel_filters = 32
has_gdn = True
channel_type = "AWGN"  # Rayleigh, AWGN, Rician, None
rician_k_factor = 2.0
loss_func = 'combined_schedule'  # mse, perceptual_loss, sobel_edge_loss, combined, gradient_loss, combined_loss_verbose, combined_schedule, ssim_loss

#Set true to enable FILM
use_snr_side_info = False
film_hidden_units = 64 
random_snr_training = False
train_snr_range = (-10.0, 10.0)

# Shared dataset and architecture parameters
dataset = "eurosatrgb"  # eurosatrgb, CIFAR10, OV_MNIST
arc_choice = 'neive_64'  # neive_64, original, reduced_filters_64, Filter_decrease, original_64


# ###################################
# 
# [USED ONLY IN TRAIN.PY]
# 
# ###################################
checkpoint_filepath = ""
epochs = 1
initial_epoch = 0

# If loss_func=combined, set relative amounts here. Comment out unused elements instead of setting them to zero.
combined_loss_weights = {
    'mse': 1.0,
    #'sobel_edge_loss': 0.1,
    #'gradient_loss': 0.5,
}

# If loss_func=combined_schedule, this sets relative loss weights in the form epoch: [loss1 weight, loss2 weight].
# Currently only works for 2 loss functions.
loss_schedule = {
        0: [1.0, 0.0],
        10: [0.5, 0.5],  
        15: [0.2, 0.8],  
    }

# ###################################
# 
# [USED ONLY IN TRAIN.PY]
# 
# ###################################
modelFile = 'test_smoke.h5'  # Must be present in models/saved_models/.

# Testing-only baseline comparison settings
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

# CLIP settings
enable_clip_metric = True
clip_device = "cpu" 
clip_model_name = "ViT-B/32"
num_semantic_eval_images = 8

# downstream task settings
enable_downstream_metric = True
downstream_model_id = "cm93/resnet18-eurosat"
downstream_device = "cpu"

# evaluation output settings
snr_range = (-20, 10)
snr_eval_step = 1
num_snr_eval_images = 32
snr_sweep_output_dir = "outputs/snr_sweep_base2"
num_visual_eval_images = 8
visual_eval_output_dir = "outputs/visual_eval"
bpg_ldpc_eval_output_dir = "outputs/bpg_ldpc_eval_film3"

# test selection
TESTS_TO_RUN = [
    #"compare_to_BPG_LDPC",
    "compare_to_BPG_LDPC_sweep",
]

# ###################################
# 
# [MODEL ARCHITECTURES, SELECTED VIA arc_choice VARIABLE] 
# 
# ###################################

# Original config from paper
# input image 32x32x3
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

# First attempt for EuroSAT data
# input image 64x64x3
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

# Slimming filters progressively
# input image 64x64x3
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

# Original but for 64x64
# input image 64x64x3
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

# Reduced filters for 64x64 input
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
