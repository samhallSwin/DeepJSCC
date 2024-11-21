from tensorflow.keras.preprocessing import image_dataset_from_directory


def dataset_generator(dir, config, mode=None, shuffle=True):
    if mode:
        dataset = image_dataset_from_directory(
            directory=dir,
            label_mode='int',
            labels='inferred',
            color_mode='rgb',
            batch_size=config.batch_size,
            image_size=(config.image_width, config.image_height),
            shuffle=shuffle,
            interpolation='bilinear',
            validation_split=0.1,
            subset=mode,
            seed=0
        )
    else:
        dataset = image_dataset_from_directory(
            directory=dir,
            label_mode='int',
            labels='inferred',
            color_mode='rgb',
            batch_size=config.batch_size,
            image_size=(config.image_width, config.image_height),
            shuffle=shuffle,
            interpolation='bilinear'
        )

    return dataset