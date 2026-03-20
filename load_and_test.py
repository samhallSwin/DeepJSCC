import numpy as np

import tests
from config import config
from config import configOverride
from common_runtime import build_model
from common_runtime import compile_model
from common_runtime import initialize_run


def main():
    args = configOverride.parse_args()
    configOverride.override_config_with_args(config, args)

    strategy, train_ds, test_ds, model_shape_file = initialize_run(config, args)

    with strategy.scope():
        model = build_model(config, model_shape_file)
        load_and_test_model(model, config, train_ds, test_ds)


def _warmup_model(model, config):
    input_shape = (1, config.image_width, config.image_height, config.image_channels)
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    if getattr(model, "use_snr_side_info", False):
        dummy_snr = np.full((1,), config.train_snrdB, dtype=np.float32)
        model((dummy_input, dummy_snr), training=False)
    else:
        model(dummy_input, training=False)


def load_and_test_model(model, config, train_ds, test_ds):
    _warmup_model(model, config)
    model.load_weights(f"models/saved_models/{config.modelFile}")
    model.summary()
    compile_model(model, config)

    if "save_reconstructions" in config.TESTS_TO_RUN:
        tests.save_reconstructions(model, test_ds, config)

    if "compare_to_BPG_LDPC" in config.TESTS_TO_RUN:
        tests.compare_to_BPG_LDPC(model, test_ds, train_ds, config)

    if "compare_to_BPG_LDPC_sweep" in config.TESTS_TO_RUN:
        tests.compare_to_BPG_LDPC_sweep(model, test_ds, config)


if __name__ == "__main__":
    main()
