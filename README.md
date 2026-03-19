# DeepJSCC

TensorFlow implementation of a Deep Joint Source-Channel Coding pipeline for image transmission over noisy channels. The project trains an autoencoder-style model that maps images to complex channel symbols, transmits them through a simulated wireless channel, and reconstructs the image at the receiver.

The current codebase is centered around:

- training and checkpointing with [train.py](/home/sam/git/DeepJSCC/train.py)
- configurable convolutional encoder/decoder architectures in [models/model.py](/home/sam/git/DeepJSCC/models/model.py)
- AWGN, Rayleigh, and Rician channel layers in [models/channellayer.py](/home/sam/git/DeepJSCC/models/channellayer.py)
- evaluation utilities in [tests.py](/home/sam/git/DeepJSCC/tests.py)
- dataset loading from folder-structured image datasets in [utils/datasets.py](/home/sam/git/DeepJSCC/utils/datasets.py)

## Features

- DeepJSCC image transmission model built with TensorFlow/Keras
- Channel simulation for `AWGN`, `Rayleigh`, and `Rician`
- Optional SNR side information with FiLM conditioning
- Multiple reconstruction losses:
  - `mse`
  - `perceptual_loss`
  - `sobel_edge_loss`
  - `gradient_loss`
  - `ssim_loss`
  - combined and scheduled variants
- Evaluation against BPG+LDPC and raw BPG baselines
- Optional semantic and downstream metrics via CLIP and timm

## Repository Layout

```text
DeepJSCC/
├── train.py                  Main entry point for training and evaluation
├── tests.py                  Evaluation helpers and BPG/LDPC comparisons
├── config/
│   ├── config.py             Default experiment configuration
│   └── configOverride.py     CLI overrides for config values
├── models/
│   ├── model.py              DeepJSCC model definition
│   ├── channellayer.py       Channel simulation layers
│   └── loss_functions.py     Custom training losses
├── utils/
│   ├── datasets.py           Dataset loading
│   ├── analysis_tools.py     BPG/LDPC analysis helpers
│   └── clip_metrics.py       Optional CLIP-based metrics
├── split_data.py             One-off dataset split helper
├── channelStateEst.py        Separate SNR estimation experiment
├── testjob.sh                Example SLURM job script
└── requirements.txt          Python dependencies
```

## Requirements

- Python 3.10 recommended
- TensorFlow 2.14
- CUDA-capable GPU recommended for training

Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some evaluation features require extra heavyweight dependencies already listed in `requirements.txt`, notably:

- `sionna` for LDPC/channel baselines
- `clip` and `torch`-ecosystem packages for semantic metrics
- `timm` for downstream classification metrics

## Dataset Layout

Datasets are expected in class-folder format compatible with `image_dataset_from_directory`, for example:

```text
dataset/
└── EuroSAT_RGB_split/
    ├── train/
    │   ├── AnnualCrop/
    │   ├── Forest/
    │   └── ...
    └── test/
        ├── AnnualCrop/
        ├── Forest/
        └── ...
```

Supported dataset names in [config/config.py](/home/sam/git/DeepJSCC/config/config.py) are:

- `eurosatrgb`
- `CIFAR10`
- `OV_MNIST`

If you need to create a train/test split for EuroSAT, [split_data.py](/home/sam/git/DeepJSCC/split_data.py) contains a simple one-off splitter.

## Configuration

Default settings live in [config/config.py](/home/sam/git/DeepJSCC/config/config.py). The most important fields are:

- `workflow`: `train` or `loadAndTest`
- `dataset`
- `experiment_name`
- `modelFile`
- `epochs`
- `train_snrdB`
- `num_symbols`
- `channel_type`
- `loss_func`
- `TESTS_TO_RUN`

CLI overrides are defined in [config/configOverride.py](/home/sam/git/DeepJSCC/config/configOverride.py).

Example override flags:

```bash
python train.py \
  --workflow train \
  --dataset eurosatrgb \
  --experiment_name eurosat_awgn_10db \
  --epochs 20 \
  --train_snrdB 10 \
  --num_symbols 512 \
  --channel_type AWGN
```

## Training

Run training with:

```bash
python train.py --workflow train
```

Useful example:

```bash
python train.py \
  --workflow train \
  --dataset eurosatrgb \
  --experiment_name film_awgn_10db \
  --epochs 20 \
  --train_snrdB 10 \
  --num_symbols 512 \
  --channel_type AWGN \
  --loss_func combined_schedule
```

Training artifacts are written to:

- `ckpt/` for epoch checkpoints
- `logs/` for saved config and TensorBoard logs
- `models/saved_models/` for final saved weights

## Loading a Trained Model and Running Evaluation

Set `workflow = "loadAndTest"` in config or pass it on the command line:

```bash
python train.py \
  --workflow loadAndTest \
  --modelFile your_model_name.h5
```

Evaluation behavior is controlled by `TESTS_TO_RUN` in [config/config.py](/home/sam/git/DeepJSCC/config/config.py).

At the moment, the implemented evaluation entry points in [tests.py](/home/sam/git/DeepJSCC/tests.py) are:

- `save_reconstructions`
- `compare_to_BPG_LDPC`
- `compare_to_BPG_LDPC_sweep`

The default config currently uses:

```python
TESTS_TO_RUN = [
    "compare_to_BPG_LDPC_sweep",
]
```

## BPG + LDPC Sweep

The most complete analysis path in the current repo is `compare_to_BPG_LDPC_sweep`. It:

- evaluates the model over a configured SNR range
- compares against BPG+LDPC and raw BPG baselines
- writes reconstructed images
- saves per-image and summary TSV metrics
- generates sweep plots for PSNR, SSIM, BER, CRC success, CLIP similarity, and downstream accuracy when enabled

Relevant config fields:

- `snr_range`
- `snr_eval_step`
- `num_snr_eval_images`
- `snr_sweep_output_dir`
- `bw_ratio`
- `mcs`
- `adaptive_bpg_ldpc`
- `adaptive_mcs_table`
- `enable_clip_metric`
- `enable_downstream_metric`

Example:

```bash
python train.py \
  --workflow loadAndTest \
  --modelFile film_awgn_10db_20.h5 \
  --snr_range=-10,15 \
  --snr_eval_step 1 \
  --num_snr_eval_images 32 \
  --snr_sweep_output_dir outputs/snr_sweep_run1
```

## SNR Side Information

The model supports conditioning the encoder and decoder on SNR side information via FiLM layers.

Enable it with:

```bash
python train.py \
  --workflow train \
  --use_snr_side_info \
  --film_hidden_units 64
```

To sample a random training SNR per image:

```bash
python train.py \
  --workflow train \
  --use_snr_side_info \
  --random_snr_training \
  --train_snr_range=-10,10
```

## Existing Helper Scripts

- [split_data.py](/home/sam/git/DeepJSCC/split_data.py): one-off dataset split utility
- [channelStateEst.py](/home/sam/git/DeepJSCC/channelStateEst.py): separate experiment for predicting SNR from channel-corrupted images
- [get_model_Ozstar.sh](/home/sam/git/DeepJSCC/get_model_Ozstar.sh): helper to sync trained models/logs from OzSTAR (Likely not relevant to most people who aren't me)
- [models/get_model_Ozstar.sh](/home/sam/git/DeepJSCC/models/get_model_Ozstar.sh): older variant of the sync helper
- [testjob.sh](/home/sam/git/DeepJSCC/testjob.sh): example SLURM submission script

## Notes and Caveats

- The repo has been cleaned up, so some old utilities referenced in historic notes are no longer present.
- Some strings in config comments still refer to older experiments and may not match the currently implemented paths.
- `train.py` references more test names than are currently defined in [tests.py](/home/sam/git/DeepJSCC/tests.py). If you add extra names to `TESTS_TO_RUN`, make sure the corresponding function actually exists first.
- The current pipeline expects datasets on disk and does not download them automatically.

## License

This project is distributed under the terms in [LICENSE](/home/sam/git/DeepJSCC/LICENSE).
