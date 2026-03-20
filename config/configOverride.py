import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DeepJSCC Model Execution")

    parser.add_argument("-n", "--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("-p", "--checkpoint_filepath", type=str, help="Checkpoint file")
    parser.add_argument("-m", "--modelFile", type=str, help="File to load saved model")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("-s", "--train_snrdB", type=int, help="SNR value for training")
    parser.add_argument("--num_symbols", type=int, help="Number of symbols")
    parser.add_argument("--initial_epoch", type=int, help="Initial epoch for training")
    parser.add_argument("--has_gdn", action="store_true", help="Enable GDN")
    parser.add_argument("-c", "--channel_type", type=str, choices=["Rayleigh", "AWGN", "Rician", "None"], help="Channel type")
    parser.add_argument("--rician_k_factor", type=float, help="Rician K factor")
    parser.add_argument("-l", "--loss_func", type=str, choices=["mse", "perceptual_loss", "sobel_edge_loss", "combined", "gradient_loss", "combined_loss_verbose", "combined_schedule", "ssim_loss"], help="Loss function")
    parser.add_argument("--use_snr_side_info", action="store_true", help="Condition encoder and decoder on SNR side information")
    parser.add_argument("--film_hidden_units", type=int, help="Hidden size for FiLM conditioning MLPs")
    parser.add_argument("--random_snr_training", action="store_true", help="Sample a random SNR per training image from train_snr_range")
    parser.add_argument("--train_snr_range", type=str, help="Training SNR range as min,max for random SNR training")
    parser.add_argument("--set_channel_filters", action="store_true", help="Set number of parameters by channel_filters")
    parser.add_argument("--channel_filters", type=int, help="Number of channel filters")
    parser.add_argument("-d", "--dataset", type=str, choices=["CIFAR10", "eurosatrgb", "OV_MNIST", "xview2"], help="Dataset name")
    parser.add_argument("-a", "--arc_choice", type=str, help="Architecture choice")
    parser.add_argument("--model_family", type=str, choices=["deepjscc", "patch_deepjscc"], help="Model family")
    parser.add_argument("--train_crop_mode", type=str, choices=["resize", "random", "center", "none"], help="Training image crop mode")
    parser.add_argument("--test_crop_mode", type=str, choices=["resize", "random", "center", "none"], help="Evaluation image crop mode")
    parser.add_argument("--allow_variable_image_size", action="store_true", help="Allow variable-size evaluation images via padded batching")
    parser.add_argument("--image_width", type=int, help="Override dataset image width")
    parser.add_argument("--image_height", type=int, help="Override dataset image height")
    parser.add_argument("--disable_nodata_check", action="store_true", help="Disable dataset no-data scanning")
    parser.add_argument("--keep_high_nodata_images", action="store_true", help="Keep images even if their no-data fraction exceeds the discard threshold")
    parser.add_argument("--nodata_black_threshold", type=int, help="Black-pixel threshold for no-data detection")
    parser.add_argument("--nodata_white_threshold", type=int, help="White-pixel threshold for no-data detection")
    parser.add_argument("--nodata_report_threshold", type=float, help="Fraction threshold used for reporting affected images")
    parser.add_argument("--nodata_discard_threshold", type=float, help="Fraction threshold above which images are excluded")
    parser.add_argument("--global_downsample_size", type=int, help="Global branch input size")
    parser.add_argument("--patch_size", type=int, help="Local patch size")
    parser.add_argument("--patch_stride", type=int, help="Local patch stride")
    parser.add_argument("--global_latent_channels", type=int, help="Global branch latent channels")
    parser.add_argument("--local_latent_channels", type=int, help="Local branch latent channels")
    parser.add_argument("--global_branch_output", type=str, choices=["rgb", "features"], help="Global branch output type")
    parser.add_argument("--local_branch_output", type=str, choices=["rgb", "features"], help="Local branch output type")
    parser.add_argument("--patch_normalization_type", type=str, choices=["batchnorm", "groupnorm", "none"], help="Normalization used in PatchDeepJSCC conv blocks")
    parser.add_argument("--patch_group_norm_groups", type=int, help="Number of groups for PatchDeepJSCC group normalization")
    parser.add_argument("--disable_global_branch", action="store_true", help="Disable the global branch for ablations")
    parser.add_argument("--disable_local_branch", action="store_true", help="Disable the local branch for ablations")
    parser.add_argument("--disable_refinement", action="store_true", help="Disable the refinement/fusion head")
    parser.add_argument("--snr_range", type=str, help="SNR range for tests (comma-separated, e.g., -10,15)")
    parser.add_argument("--snr_eval_step", type=int, help="Step size for SNR sweep tests")
    parser.add_argument("--num_snr_eval_images", type=int, help="Number of test images to use in the SNR sweep")
    parser.add_argument("--LDPCon", action="store_true", help="Enable LDPC comparison")
    parser.add_argument("--bw_ratio", type=float, help="Bandwidth ratio for traditional compression test")
    parser.add_argument("--mcs", type=str, help="Modulation and coding scheme (comma-separated, e.g., 3072,6144,2)")

    return parser.parse_args()


def override_config_with_args(config, args):
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.modelFile:
        config.modelFile = args.modelFile
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.train_snrdB:
        config.train_snrdB = args.train_snrdB
    if args.num_symbols:
        config.num_symbols = args.num_symbols
    if args.initial_epoch:
        config.initial_epoch = args.initial_epoch
    if args.checkpoint_filepath:
        config.checkpoint_filepath = args.checkpoint_filepath
    if args.has_gdn:
        config.has_gdn = True
    if args.channel_type:
        config.channel_type = args.channel_type
    if args.rician_k_factor is not None:
        config.rician_k_factor = args.rician_k_factor
    if args.loss_func:
        config.loss_func = args.loss_func
    if args.use_snr_side_info:
        config.use_snr_side_info = True
    if args.film_hidden_units:
        config.film_hidden_units = args.film_hidden_units
    if args.random_snr_training:
        config.random_snr_training = True
    if args.train_snr_range:
        snr_min, snr_max = map(float, args.train_snr_range.split(","))
        config.train_snr_range = (snr_min, snr_max)
    if args.set_channel_filters:
        config.set_channel_filters = True
    if args.channel_filters:
        config.channel_filters = args.channel_filters
    if args.dataset:
        config.dataset = args.dataset
    if args.arc_choice:
        config.arc_choice = args.arc_choice
    if args.model_family:
        config.model_family = args.model_family
    if args.train_crop_mode:
        config.train_crop_mode = args.train_crop_mode
    if args.test_crop_mode:
        config.test_crop_mode = args.test_crop_mode
    if args.allow_variable_image_size:
        config.allow_variable_image_size = True
    if args.image_width:
        config.override_image_width = args.image_width
    if args.image_height:
        config.override_image_height = args.image_height
    if args.disable_nodata_check:
        config.enable_nodata_check = False
    if args.keep_high_nodata_images:
        config.exclude_high_nodata_images = False
    if args.nodata_black_threshold is not None:
        config.nodata_black_threshold = args.nodata_black_threshold
    if args.nodata_white_threshold is not None:
        config.nodata_white_threshold = args.nodata_white_threshold
    if args.nodata_report_threshold is not None:
        config.nodata_report_threshold = args.nodata_report_threshold
    if args.nodata_discard_threshold is not None:
        config.nodata_discard_threshold = args.nodata_discard_threshold
    if args.global_downsample_size:
        config.global_downsample_size = args.global_downsample_size
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.patch_stride:
        config.patch_stride = args.patch_stride
    if args.global_latent_channels:
        config.global_latent_channels = args.global_latent_channels
    if args.local_latent_channels:
        config.local_latent_channels = args.local_latent_channels
    if args.global_branch_output:
        config.global_branch_output = args.global_branch_output
    if args.local_branch_output:
        config.local_branch_output = args.local_branch_output
    if args.patch_normalization_type:
        config.patch_normalization_type = args.patch_normalization_type
    if args.patch_group_norm_groups:
        config.patch_group_norm_groups = args.patch_group_norm_groups
    if args.disable_global_branch:
        config.enable_global_branch = False
    if args.disable_local_branch:
        config.enable_local_branch = False
    if args.disable_refinement:
        config.enable_refinement = False
    if args.snr_range:
        snr_min, snr_max = map(int, args.snr_range.split(","))
        config.snr_range = (snr_min, snr_max)
    if args.snr_eval_step:
        config.snr_eval_step = args.snr_eval_step
    if args.num_snr_eval_images:
        config.num_snr_eval_images = args.num_snr_eval_images
    if args.LDPCon:
        config.LDPCon = True
    if args.bw_ratio:
        config.bw_ratio = args.bw_ratio
    if args.mcs:
        mcs_values = list(map(int, args.mcs.split(",")))
        if len(mcs_values) == 3:
            config.mcs = tuple(mcs_values)
