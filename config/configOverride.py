import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DeepJSCC Model Execution")

    # Add arguments to override config.py
    parser.add_argument("-n", '--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument("-p", '--checkpoint_filepath', type=str, help='Checkpoint file')
    parser.add_argument("-w", '--workflow', type=str, choices=['train', 'loadAndTest'], help='Workflow type')
    parser.add_argument("-m", '--modelFile', type=str, help='File to load saved model')
    parser.add_argument("-b", '--batch_size', type=int, help='Batch size for training')
    parser.add_argument("-e", '--epochs', type=int, help='Number of epochs')
    parser.add_argument("-s", '--train_snrdB', type=int, help='SNR value for training')
    parser.add_argument("--num_symbols", type=int, help='Number of symbols')
    parser.add_argument("--initial_epoch", type=int, help='Initial epoch for training')
    parser.add_argument("--has_gdn", action='store_true', help='Enable GDN')
    parser.add_argument("-c", '--channel_type', type=str, choices=['Rayleigh', 'AWGN', 'Rician', 'None'], help='Channel type')
    parser.add_argument("-l", '--loss_func', type=str, choices=['mse', 'perceptual_loss', 'sobel_edge_loss', 'combined', 'gradient_loss', 'combined_loss_verbose', 'combined_schedule', 'ssim_loss'], help='Loss function')
    parser.add_argument("--set_channel_filters", action='store_true', help='Set number of parameters by channel_filters')
    parser.add_argument("--channel_filters", type=int, help='Number of channel filters')
    parser.add_argument("-d", '--dataset', type=str, choices=['CIFAR10', 'eurosatrgb', 'OV_MNIST'], help='Dataset name')
    parser.add_argument("-a", '--arc_choice', type=str, help='Architecture choice')
    parser.add_argument("--snr_range", type=str, help='SNR range for tests (comma-separated, e.g., "-10,15")')
    parser.add_argument("--LDPCon", action='store_true', help='Enable LDPC comparison')
    parser.add_argument("--bw_ratio", type=float, help='Bandwidth ratio for traditional compression test')
    parser.add_argument("--mcs", type=str, help='Modulation and coding scheme (comma-separated, e.g., "3072,6144,2")')

    return parser.parse_args()

def override_config_with_args(config, args):
    """
    Update config variables with command-line arguments.
    """
    if args.experiment_name: config.experiment_name = args.experiment_name
    if args.workflow: config.workflow = args.workflow
    if args.modelFile: config.modelFile = args.modelFile
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.epochs = args.epochs
    if args.train_snrdB: config.train_snrdB = args.train_snrdB
    if args.num_symbols: config.num_symbols = args.num_symbols
    if args.initial_epoch: config.initial_epoch = args.initial_epoch
    if args.checkpoint_filepath: config.checkpoint_filepath = args.checkpoint_filepath
    if args.has_gdn: config.has_gdn = True
    if args.channel_type: config.channel_type = args.channel_type
    if args.loss_func: config.loss_func = args.loss_func
    if args.set_channel_filters: config.set_channel_filters = True
    if args.channel_filters: config.channel_filters = args.channel_filters
    if args.dataset: config.dataset = args.dataset
    if args.arc_choice: config.arc_choice = args.arc_choice
    if args.snr_range:
        snr_min, snr_max = map(int, args.snr_range.split(","))
        config.snr_range = (snr_min, snr_max)
    if args.LDPCon: config.LDPCon = True
    if args.bw_ratio: config.bw_ratio = args.bw_ratio
    if args.mcs:
        mcs_values = list(map(int, args.mcs.split(",")))
        if len(mcs_values) == 3:
            config.mcs = tuple(mcs_values)
