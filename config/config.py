
experiment_name = "eurosat_rgb_test" #effects file locations. Will overwrite previous with same name for the most part
workflow = "loadAndTest" #train, loadAndTest
checkpoint_filepath = "checkpoint.ckpt"

modelFile = 'eurosat_RGB_test_10.h5' #used for loading model for testing. Must be in /models/saved_models/

#training params
batch_size = 32
epochs = 10
train_snrdB = 15
data_size = 512
initial_epoch = 0

#Architecture params
has_gdn = True
channel_type = "AWGN"


#Must match entry in setImageParamsFromDataset()
dataset = "eurosatrgb" #eurosatrgb, CIFAR10, OV_MNIST


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
    else:
        print('No supported dataset - check spelling')
