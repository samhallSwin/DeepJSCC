
experiment_name = "eurosat_RGB_test"
checkpoint_filepath = "checkpoint.ckpt"

mode = "test"
dataset = "cifar10" #Options are eurosatrgb, cifar10

has_gdn = True
snrdB =15
num_symbols = 512
input_size = 32
# Configurations for different architectures. Select one by setting arcChoice to desired arc_name
#new arcs can be added as required by following the format


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
    # Add more architectures as needed
}


# Selected architecture based on arcChoice
architecture = architectures[arcChoice]


#Training params
batch_size = 64
epochs = 1
data_size = 512
initial_epoch = 0