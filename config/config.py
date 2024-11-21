experiment_name = "eurosat_RGB_test"
checkpoint_filepath = "checkpoint.ckpt"

mode = "test"
dataset = "eurosatrgb"  # Options are eurosatrgb, cifar10

has_gdn = True
snrdB = 15
num_symbols = 512
input_size = 64

# Dataset configurations
datasets_config = {
    "cifar10": {
        "name": "CIFAR10",
        "train_dir": "./dataset/CIFAR10/train/",
        "test_dir": "./dataset/CIFAR10/test/",
        "image_width": 32,
        "image_height": 32,
        "image_channels": 3,
    },
    "eurosatrgb": {
        "name": "EuroSAT RGB",
        "train_dir": "./dataset/EuroSAT_RGB_split/train/",
        "test_dir": "./dataset/EuroSAT_RGB_split/test/",
        "image_width": 64,
        "image_height": 64,
        "image_channels": 3,
    },
}

# Selected architecture based on arcChoice
arcChoice = 'nieve_4_layers'
architectures = {
    'nieve_4_layers': {
        'has_gdn': True,
        'snrdB': 10,
        'channel': 'AWGN',
        'encoder': {
            'layers': [
                {'type': 'conv', 'filters': 256, 'kernel_size': 9, 'strides': 2},
                {'type': 'gdn'},
                {'type': 'conv', 'filters': 256, 'kernel_size': 9, 'strides': 2},
                {'type': 'gdn'},
                {'type': 'conv', 'filters': 256, 'kernel_size': 5, 'strides': 2},
                {'type': 'gdn'},
                {'type': 'conv', 'filters': 256, 'kernel_size': 5},
            ]
        },
        'decoder': {
            'layers': [
                {'type': 'conv', 'filters': 256, 'kernel_size': 5},
                {'type': 'resize', 'height': 16, 'width': 16},
                {'type': 'conv', 'filters': 256, 'kernel_size': 5},
                {'type': 'resize', 'height': 32, 'width': 32},
                {'type': 'conv', 'filters': 256, 'kernel_size': 9},
                {'type': 'resize', 'height': 64, 'width': 64},
                {'type': 'conv', 'filters': 256, 'kernel_size': 9},
                {'type': 'conv', 'filters': 3, 'kernel_size': 1, 'activation': 'sigmoid'},
            ]
        },
    },
    'Original': {
        'has_gdn': False,
        'snrdB': 20,
        'channel': 'Rician',
        'encoder': {
            'layers': [
                {'type': 'conv', 'filters': 256, 'kernel_size': 9, 'strides': 2},
                {'type': 'conv', 'filters': 256, 'kernel_size': 5, 'strides': 2},
                {'type': 'conv', 'filters': 256, 'kernel_size': 5},
            ]
        },
        'decoder': {
            'layers': [
                {'type': 'conv', 'filters': 256, 'kernel_size': 5},
                {'type': 'resize', 'height': 16, 'width': 16},
                {'type': 'conv', 'filters': 256, 'kernel_size': 5},
                {'type': 'resize', 'height': 32, 'width': 32},
                {'type': 'conv', 'filters': 256, 'kernel_size': 9},
                {'type': 'conv', 'filters': 3, 'kernel_size': 1, 'activation': 'sigmoid'},
            ]
        },
    },
}

architecture = architectures[arcChoice]

# Training parameters
batch_size = 32
epochs = 1
data_size = 512
initial_epoch = 0

# Access dataset configuration
current_dataset_config = datasets_config[dataset]