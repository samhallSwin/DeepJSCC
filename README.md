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
├── train.py                  Training entry point
├── load_and_test.py          Evaluation entry point for saved models
├── common_runtime.py         Shared runtime, dataset, and compile helpers
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

For OzSTAR GPU runs, this repo is currently aligned to:

- TensorFlow `2.14.1`
- `tensorflow_compression` `2.14.1`
- CUDA `11.8`
- cuDNN `8.7`

This is intentional: `tensorflow_compression`'s full Python package is only available for TensorFlow 2.14.x in the current project setup, so the newer CUDA 12.6 / cuDNN 9.5 module combination in older job scripts is not a safe match.

Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want the optional semantic/downstream evaluation features as well:

```bash
pip install -r requirements-eval.txt
```

Optional extras include:

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
python train.py
```

Useful example:

```bash
python train.py \
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

Use the dedicated evaluation entrypoint:

```bash
python load_and_test.py --modelFile your_model_name.h5
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
- `bw_ratio`
- `mcs`
- `adaptive_bpg_ldpc`
- `adaptive_mcs_table`
- `enable_clip_metric`
- `enable_downstream_metric`

Evaluation outputs are written automatically under:

```text
outputs/[experiment_name]/[test_name]/
```

For example:

- `outputs/my_run/save_reconstructions/`
- `outputs/my_run/compare_to_BPG_LDPC/`
- `outputs/my_run/compare_to_BPG_LDPC_sweep/`

Example:

```bash
python load_and_test.py \
  --modelFile film_awgn_10db_20.h5 \
  --snr_range=-10,15 \
  --snr_eval_step 1 \
  --num_snr_eval_images 32
```

## SNR Side Information

The model supports conditioning the encoder and decoder on SNR side information via FiLM layers.

Enable it with:

```bash
python train.py \
  --use_snr_side_info \
  --film_hidden_units 64
```

To sample a random training SNR per image:

```bash
python train.py \
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

## PatchDeepJSCC

A new `PatchDeepJSCC` model family is available for larger images. It combines:
- a low-resolution global branch for scene structure
- a shared overlapping local patch branch for detail
- overlap-aware blending plus an optional refinement head for final fusion

Typical training example on xView2 with 512x512 crops:

```bash
python train.py \
  --model_family patch_deepjscc \
  --dataset xview2 \
  --image_width 512 \
  --image_height 512 \
  --batch_size 2 \
  --patch_size 128 \
  --patch_stride 96 \
  --global_downsample_size 128 \
  --train_snrdB 10
```

Ablation examples:
- disable the global branch with `--disable_global_branch`
- disable the local branch with `--disable_local_branch`
- disable refinement with `--disable_refinement`
- switch branch outputs between `rgb` and `features` with `--global_branch_output` and `--local_branch_output`

Evaluation uses the same entry point and can save intermediate global/local reconstructions when the patch model is active:

```bash
python load_and_test.py \
  --model_family patch_deepjscc \
  --dataset xview2 \
  --image_width 1024 \
  --image_height 1024 \
  --batch_size 1 \
  --modelFile your_patch_model.h5
```
