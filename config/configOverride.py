import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DeepJSCC Model Execution")
    
    # Add arguments to override config.py
    parser.add_argument("-n", '--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument("-p",'--checkpoint_filepath', type=str,help='checkpoint file')
    parser.add_argument("-w", '--workflow', type=str, choices=['train', 'loadAndTest'], help='Workflow type')
    parser.add_argument("-m", '--modelFile', type=str, help='File to load saved model')
    parser.add_argument("-b", '--batch_size', type=int, help='Batch size for training')
    parser.add_argument("-e", '--epochs', type=int, help='Number of epochs')
    parser.add_argument("-s", '--train_snrdB', type=int, help='SNR value for training')
    parser.add_argument("--num_symbols", type=int, help='Number of symbols')
    parser.add_argument("--initial_epoch", type=int, help='Initial epoch for training')
    parser.add_argument("--has_gdn", action='store_true', help='Enable GDN')
    parser.add_argument("-c", '--channel_type', type=str, choices=['Rayleigh', 'AWGN', 'Rician', 'None'], help='Channel type')
    parser.add_argument("-l", '--loss_func', type=str, choices=['mse', 'perceptual_loss'], help='Loss function')
    parser.add_argument("-d", '--dataset', type=str, choices=['CIFAR10', 'eurosatrgb', 'OV_MNIST'], help='Dataset name')
    parser.add_argument("-a", '--arc_choice', type=str, help='Architecture choice')
    parser.add_argument("--snr_range", type=str, help='SNR range for tests (comma-separated, e.g., "-10,15")')
    
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
    if args.dataset: config.dataset = args.dataset
    if args.arc_choice: config.arc_choice = args.arc_choice
    if args.snr_range:
        snr_min, snr_max = map(int, args.snr_range.split(","))
        config.snr_range = (snr_min, snr_max)