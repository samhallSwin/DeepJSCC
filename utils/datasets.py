from tensorflow.keras.preprocessing import image_dataset_from_directory


def dataset_generator(dir, params, mode=None, shuffle=True):
    if mode:
        dataset = image_dataset_from_directory(
            directory=dir,
            label_mode='int',
            labels='inferred',
            color_mode='rgb',
            batch_size=params.batch_size,
            image_size=(params.image_width, params.image_height),
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
            batch_size=params.batch_size,
            image_size=(params.image_width, params.image_height),
            shuffle=shuffle,
            interpolation='bilinear'
        )

    return dataset