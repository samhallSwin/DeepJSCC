import os

import numpy as np

from config import config
from config import configOverride
from common_runtime import LossWeightScheduler
from common_runtime import build_model
from common_runtime import compile_model
from common_runtime import initialize_run
from common_runtime import tf


def main():
    args = configOverride.parse_args()
    configOverride.override_config_with_args(config, args)

    strategy, train_ds, test_ds, model_shape_file = initialize_run(config, args)

    with strategy.scope():
        model = build_model(config, model_shape_file)
        train_model(model, config, train_ds, test_ds, args)


def _warmup_model(model, config):
    input_shape = (1, config.image_width, config.image_height, config.image_channels)
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    if getattr(model, "use_snr_side_info", False):
        dummy_snr = np.full((1,), config.train_snrdB, dtype=np.float32)
        model((dummy_input, dummy_snr), training=False)
    else:
        model(dummy_input, training=False)


def train_model(model, config, train_ds, test_ds, args):
    compile_model(model, config)
    _warmup_model(model, config)
    model.summary()

    if args.checkpoint_filepath is not None:
        model.load_weights(args.checkpoint_filepath)

    save_ckpt = model_checkpoint(config)
    tensorboard = tensorboard_callback(config)

    if config.loss_func == "combined_schedule":
        dynamic_weights = [tf.Variable(1.0), tf.Variable(0.0)]
        loss_weight_scheduler = LossWeightScheduler(dynamic_weights, config.loss_schedule)
        model.fit(
            train_ds,
            initial_epoch=config.initial_epoch,
            epochs=config.epochs,
            callbacks=[tensorboard, save_ckpt, loss_weight_scheduler],
            validation_data=test_ds,
        )
    else:
        model.fit(
            train_ds,
            initial_epoch=config.initial_epoch,
            epochs=config.epochs,
            callbacks=[tensorboard, save_ckpt],
            validation_data=test_ds,
        )

    save_dir = "models/saved_models/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config.experiment_name}_{config.epochs}.h5")
    model.save_weights(save_path)
    print(f"Model saved to {save_path}")


def model_checkpoint(config):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./ckpt/{config.experiment_name}_" + "{epoch}",
        save_best_only=True,
        monitor="val_loss",
        save_weights_only=True,
        options=tf.train.CheckpointOptions(
            experimental_io_device=None, experimental_enable_async_checkpoint=True
        ),
    )


def tensorboard_callback(config):
    return tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{config.experiment_name}_{config.epochs}"
    )


if __name__ == "__main__":
    main()
