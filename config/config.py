# ###################################
#
# [USED FOR BOTH TRAINING AND TESTING]
#
# ###################################
experiment_name = "Patch_tester"
batch_size = 1
train_snrdB = 10
num_symbols = 512
model_family = "patch_deepjscc"  # deepjscc, patch_deepjscc

# model and channel parameters
set_channel_filters = True
channel_filters = 32
has_gdn = True
channel_type = "AWGN"  # Rayleigh, AWGN, Rician, None
rician_k_factor = 2.0
loss_func = "combined_schedule"  # mse, perceptual_loss, sobel_edge_loss, combined, gradient_loss, combined_loss_verbose, combined_schedule, ssim_loss

# SNR side information
use_snr_side_info = False
film_hidden_units = 64
random_snr_training = False
train_snr_range = (-10.0, 10.0)

# Shared dataset and architecture parameters
dataset = "xview2"  # eurosatrgb, CIFAR10, OV_MNIST, xview2
#arc_choice only relevant for model_family == "deepjscc"
arc_choice = "neive_64"  # neive_64, original, reduced_filters_64, Filter_decrease, original_64
shuffle_buffer_size = 2048
cache_train_dataset = True
cache_test_dataset = True
train_crop_mode = "resize"  # resize, random, center, none
test_crop_mode = "resize"  # resize, random, center, none
allow_variable_image_size = False
override_image_width = None
override_image_height = None

#checks data for missing regions and excludes images with too much missing data from training and testing. Only relevant for datasets with nodata regions, like xview2.
enable_nodata_check = True
exclude_high_nodata_images = True
save_nodata_report = True
nodata_black_threshold = 4
nodata_white_threshold = 251
nodata_report_threshold = 0.01
nodata_discard_threshold = 0.20

# PatchDeepJSCC settings
# The patch model works on large images by combining a low-resolution global branch
# with a shared overlapping local-patch branch.
global_downsample_size = 128
patch_size = 128
patch_stride = 96
global_latent_channels = 64
local_latent_channels = 32
global_branch_filters = (64, 96, 128)
local_branch_filters = (48, 64, 96)
global_branch_output = "rgb"  # rgb, features
local_branch_output = "rgb"  # rgb, features
enable_global_branch = True
enable_local_branch = True
enable_refinement = True
refinement_filters = 64
refinement_depth = 3
overlap_power = 2.0
save_patch_intermediates = True
patch_normalization_type = "groupnorm"  # batchnorm, groupnorm, none (Use groupnorm for better performance on small batch sizes, like batch_size=1-4)
patch_group_norm_groups = 8

# ###################################
#
# [USED ONLY IN TRAIN.PY]
#
# ###################################
checkpoint_filepath = ""
epochs = 1
initial_epoch = 0

combined_loss_weights = {
    "mse": 1.0,
    #"sobel_edge_loss": 0.1,
    #"gradient_loss": 0.5,
}

loss_schedule = {
    0: [1.0, 0.0],
    10: [0.5, 0.5],
    15: [0.2, 0.8],
}

# ###################################
#
# [USED ONLY IN LOAD_AND_TEST.PY]
#
# ###################################
modelFile = "split_test_smoke_1.h5"

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
num_visual_eval_images = 8

TESTS_TO_RUN = [
    #"compare_to_BPG_LDPC",
    "save_reconstructions",
    "compare_to_BPG_LDPC_sweep",
]

# ###################################
#
# [MODEL ARCHITECTURES, SELECTED VIA arc_choice VARIABLE]
#
# ###################################
encoder_config_original = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_original = [
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},
]

encoder_config_neive_64 = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_neive_64 = [
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},
]

encoder_config_Filter_decrease = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 128, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 64, "kernel_size": 5, "stride": 1, "block_type": "C"},
    {"filters": 32, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_Filter_decrease = [
    {"filters": 32, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 64, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 128, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": None},
]

encoder_config_original_64 = [
    {"filters": 256, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 2, "block_type": "C"},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_config_original_64 = [
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 256, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 256, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
]

encoder_reduced_filters_64 = [
    {"filters": 128, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 128, "kernel_size": 9, "stride": 2, "block_type": "C"},
    {"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C"},
]

decoder_reduced_filters_64 = [
    {"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 16},
    {"filters": 128, "kernel_size": 5, "stride": 1, "block_type": "C", "upsample_size": 32},
    {"filters": 128, "kernel_size": 9, "stride": 1, "block_type": "C", "upsample_size": 64},
]


def setArc():
    if arc_choice == "original":
        enc = encoder_config_original
        dec = decoder_config_original
    elif arc_choice == "neive_64":
        enc = encoder_config_neive_64
        dec = decoder_config_neive_64
    elif arc_choice == "reduced_filters_64":
        enc = encoder_reduced_filters_64
        dec = decoder_reduced_filters_64
    elif arc_choice == "Filter_decrease":
        enc = encoder_config_Filter_decrease
        dec = decoder_config_Filter_decrease
    elif arc_choice == "original_64":
        enc = encoder_config_original_64
        dec = decoder_config_original_64
    else:
        raise ValueError(f"architecture not found: {arc_choice}")

    return enc, dec


def setImageParamsFromDataset():
    global train_dir, test_dir, image_width, image_height, image_channels
    global cache_train_dataset, cache_test_dataset, allow_variable_image_size
    global override_image_width, override_image_height

    if dataset == "CIFAR10":
        train_dir = "./dataset/CIFAR10/train/"
        test_dir = "./dataset/CIFAR10/test/"
        image_width = 32
        image_height = 32
        image_channels = 3
        cache_train_dataset = True
        cache_test_dataset = True
        allow_variable_image_size = False
        print("CIFAR10 dataset selected")
    elif dataset == "eurosatrgb":
        train_dir = "./dataset/EuroSAT_RGB_split/train/"
        test_dir = "./dataset/EuroSAT_RGB_split/test/"
        image_width = 64
        image_height = 64
        image_channels = 3
        cache_train_dataset = True
        cache_test_dataset = True
        allow_variable_image_size = False
        print("eurosat_RGB dataset selected")
    elif dataset == "OV_MNIST":
        train_dir = "./dataset/OV-MNIST/training/"
        test_dir = "./dataset/OV-MNIST/testing/"
        image_width = 28
        image_height = 28
        image_channels = 3
        cache_train_dataset = True
        cache_test_dataset = True
        allow_variable_image_size = False
        print("OV_MNIST dataset selected")
    elif dataset == "xview2":
        train_dir = "../datasets/xView2/train/images/"
        test_dir = "../datasets/xView2/test/images/"
        image_width = 512
        image_height = 512
        image_channels = 3
        cache_train_dataset = False
        cache_test_dataset = False
        print("xView2 dataset selected")
    else:
        raise ValueError(f"No supported dataset: {dataset}")

    if override_image_width is not None:
        image_width = int(override_image_width)
    if override_image_height is not None:
        image_height = int(override_image_height)
