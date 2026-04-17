# CS-4375 Project: NumPy Denoising Autoencoder (DAE)

## Team Members
- Casey Nguyen (CXN220034)
- Nguyen Phuc Do (NPD220001)

## Project Topic
Image denoising using custom NumPy autoencoders on CIFAR-10.

## Technique Summary
- Add controlled noise to clean images.
- Train encoder-decoder network to reconstruct clean images from noisy inputs.
- Train using **MSE** or optional hybrid **MSE + L1** reconstruction loss.
- Use **K-Fold Cross Validation** (default K=5) for more reliable performance estimates.
- Train a custom **NumPy Softmax Classifier** on denoised images to predict object class.

## Algorithms Implemented
- Custom NumPy Conv2D (forward + backward + updates)
- Custom NumPy convolutional DAE with spatial bottleneck
- Custom NumPy fully connected DAE (configurable): input_dim -> hidden -> bottleneck -> hidden -> input_dim
- Optional hybrid reconstruction loss: `loss = MSE + l1_weight * MAE`
- Custom NumPy Softmax Regression classifier (multiclass prediction)
- Classification on encoder latent features (not decoded pixels) for stronger class separation
- Classifier training augmentation using both noisy and clean encoder features

## Dataset
- Name: CIFAR-10
- Official source: https://www.cs.toronto.edu/~kriz/cifar.html
- Python tar file: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- Loading method used in this project: `torchvision.datasets.CIFAR10`
- Data shape: 32 x 32 x 3 per image (3072 features)
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Pixel range: [0, 255], normalized to [0, 1]

### Kaggle Variant Notes (CIFAR-10 - Object Recognition in Images)
- Kaggle dataset description: 60,000 labeled 32x32 color images across 10 classes.
- Preserved official split: 50,000 training images and 10,000 official test images.
- Kaggle files are typically `train.7z`, `test.7z`, and `trainLabels.csv`.
- Kaggle competition test folder contains 300,000 images total:
	- 10,000 are the official test images (with small modifications)
	- 290,000 are junk images used to discourage cheating
- If you use Kaggle files, generate predictions for all 300,000 competition test images.
- This project code uses torchvision to automatically download/load CIFAR-10 from the official source.

Class definitions used by Kaggle and CIFAR-10 are mutually exclusive:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Important class note:
- `automobile` includes regular road cars (sedans, SUVs, similar vehicles)
- `truck` includes only large trucks (not pickup trucks)

## Repository Structure

```
CS-4375-Project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── .gitkeep
│   └── cifar-10-batches-py/    # auto-downloaded cache
├── experiments/
│   ├── checkpoints/
│   ├── configs/
│   └── logs/
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/
│   └── run_full_dataset.sh
├── src/
│   ├── classification/
│   ├── config.py
│   ├── main.py
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

## Setup

From project root:

Recommended Python version for easiest dependency resolution:
- Python 3.9 to 3.12

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download CIFAR-10

Dataset download is automatic on first run:
- Preferred path: `torchvision.datasets.CIFAR10`
- Fallback path: official CIFAR-10 Python archive from the same source

Optional pre-download command using the project .venv:

### macOS/Linux

```bash
.venv/bin/python -c "from torchvision import datasets; datasets.CIFAR10(root='./data', train=True, download=True); datasets.CIFAR10(root='./data', train=False, download=True)"
```

### Windows (PowerShell)

```powershell
& .\.venv\Scripts\python.exe -c "from torchvision import datasets; datasets.CIFAR10(root='./data', train=True, download=True); datasets.CIFAR10(root='./data', train=False, download=True)"
```

Equivalent code used by this project:

```python
from torchvision import datasets

train_data = datasets.CIFAR10(root="./data", train=True, download=True)
test_data = datasets.CIFAR10(root="./data", train=False, download=True)
```

The training script validates the official dataset constraints at startup:
- 60,000 total images
- 50,000 train + 10,000 test
- 10 classes
- 6,000 images per class overall (5,000 train + 1,000 test)

## Run

Use exactly one script for the entire project workflow:

### macOS/Linux

```bash
bash scripts/run_full_dataset.sh
```

### Windows (PowerShell + Git Bash)

```powershell
& "C:\Program Files\Git\bin\bash.exe" scripts/run_full_dataset.sh
```

Default behavior:
- Uses `.venv` interpreter
- Verifies `torchvision` in `.venv` before training/evaluation
- Uses optimized defaults for course-scale experiments:
	- mode: `single` (one new experiment row)
	- model: `conv` (NumPy convolutional DAE, 3->32->96 bottleneck->32->3)
	- train/test subset: `5000/1000`
	- image size: `32x32` (no downsampling)
	- noise: gaussian with adjustable std (default `FULL_NOISE_STD=0.05` for higher PSNR runs)
	- denoising-focused defaults: `FULL_EPOCHS=80`, `FULL_BATCH_SIZE=32`, `FULL_LEARNING_RATE=0.001`, `FULL_WEIGHT_DECAY=0.0`, `FULL_LATENT_CHANNELS=96`, `FULL_CONV_SKIP_CONNECTION_WEIGHT=0.8`, `FULL_LR_DECAY=0.995`, `FULL_EARLY_STOPPING_PATIENCE=12`, `FULL_VAL_RATIO=0.1`
	- dynamic train subset per epoch: `FULL_EPOCH_TRAIN_SUBSET_MIN=2500`, `FULL_EPOCH_TRAIN_SUBSET_MAX=3000` (improves sample diversity per epoch)
	- per-batch gaussian noise scheduling is optional and disabled by default (`FULL_SAMPLE_NOISE_PER_BATCH=0`, `FULL_BATCH_NOISE_STD_OPTIONS=0.04,0.05,0.06`)
	- hybrid denoising loss (optional): default `FULL_L1_WEIGHT=0.0` (pure MSE for PSNR-focused runs)
	- automatic post-training skip calibration on validation set refines the effective skip weight to minimize validation MSE
	- runtime speed-up: per-epoch train metrics are sampled with `FULL_TRAIN_METRICS_MAX_SAMPLES=1024` (set `0` for full-train metrics)
	- SSIM tracking (optional): `FULL_TRACK_SSIM=1`
- Keeps denoising as the primary objective while reporting classification accuracy by default (`FULL_RUN_CLASSIFICATION=1`), using classifier MLP defaults `64,32` with dropout `0.5` (`FULL_CLASSIFIER_EPOCHS=40`, `FULL_CLASSIFIER_BATCH_SIZE=32`, `FULL_CLASSIFIER_LR=0.005`, `FULL_CLASSIFIER_WEIGHT_DECAY=0.001`)
- Saves checkpoint(s), denoising figure(s), loss curve(s), summary table(s), and prediction CSV(s)
- Default checkpoint: `experiments/checkpoints/cdae_denoise_focus.npz`
- Default denoising figure: `reports/figures/denoise_focus_preview.png`
- Default loss curve: `reports/figures/denoise_focus_loss_curve.png`
- Default denoising preview count: `10` images
- Figure files are regenerated on each run, and the loss-curve image is refreshed during training epochs to show progress.
- Removes stale run artifacts by default before each run (`FULL_CLEAN_OUTPUTS=1`).
- Disable output generation with CLI flags when needed: `--no-save-figure` and/or `--no-save-loss-curve`
- Default predictions CSV: `reports/tables/full_predictions.csv`
- Default summary table: `reports/tables/denoise_focus_summary_latest.csv`
- Logs new experiment rows to `experiments/logs/experiment_runs.csv`

Evaluation-only K-fold mode:
- Set `FULL_MODE=kfold_eval` to evaluate a pre-trained checkpoint only.
- Requires `FULL_INIT_CHECKPOINT` to point to an existing `.npz` checkpoint.
- This mode performs forward pass inference per fold and reports validation MSE/RMSE/PSNR.
- No weight updates occur in this mode.
- Classification is disabled automatically in this mode.
- Uses shuffled, non-overlapping folds (`FULL_K_FOLDS=5` default, or `3` for faster runs).
- Writes a report-ready table to `reports/tables/kfold_eval_summary_latest.csv`.

Useful overrides:

### macOS/Linux

```bash
FULL_MODE=kfold FULL_K_FOLDS=3 bash scripts/run_full_dataset.sh
FULL_MODE=kfold_eval FULL_INIT_CHECKPOINT=experiments/checkpoints/cdae_denoise_focus.npz bash scripts/run_full_dataset.sh
FULL_EPOCHS=5 bash scripts/run_full_dataset.sh
FULL_MODEL_TYPE=conv bash scripts/run_full_dataset.sh
FULL_MAX_SAMPLES=2000 FULL_MAX_TEST_SAMPLES=500 bash scripts/run_full_dataset.sh
FULL_NOISE_STD=0.05 bash scripts/run_full_dataset.sh
FULL_SAMPLE_NOISE_PER_BATCH=1 FULL_BATCH_NOISE_STD_OPTIONS=0.04,0.05,0.06 bash scripts/run_full_dataset.sh
FULL_EPOCH_TRAIN_SUBSET_MIN=1800 FULL_EPOCH_TRAIN_SUBSET_MAX=1800 bash scripts/run_full_dataset.sh
FULL_L1_WEIGHT=0.0 bash scripts/run_full_dataset.sh
FULL_TRAIN_METRICS_MAX_SAMPLES=2048 bash scripts/run_full_dataset.sh
FULL_TRACK_SSIM=1 bash scripts/run_full_dataset.sh
FULL_CONV_SKIP_CONNECTION_WEIGHT=0.85 bash scripts/run_full_dataset.sh
FULL_CLASSIFIER_HIDDEN_DIMS=64,32 FULL_CLASSIFIER_DROPOUT=0.5 FULL_CLASSIFIER_EPOCHS=40 FULL_CLASSIFIER_BATCH_SIZE=32 FULL_CLASSIFIER_WEIGHT_DECAY=0.0005 bash scripts/run_full_dataset.sh
FULL_RESIZE_TO=32 bash scripts/run_full_dataset.sh
FULL_RUN_CLASSIFICATION=1 bash scripts/run_full_dataset.sh
FULL_MODE=kfold_eval FULL_K_FOLDS=3 FULL_KFOLD_EVAL_OUTPUT=reports/tables/kfold_eval_summary_latest.csv bash scripts/run_full_dataset.sh
FULL_NUM_PREDICTION_SAMPLES=10000 bash scripts/run_full_dataset.sh
FULL_INIT_CHECKPOINT=experiments/checkpoints/cdae_denoise_focus.npz bash scripts/run_full_dataset.sh
```

### Windows (PowerShell + Git Bash)

```powershell
$env:FULL_MODE='kfold'
$env:FULL_K_FOLDS='3'
& "C:\Program Files\Git\bin\bash.exe" scripts/run_full_dataset.sh
Remove-Item Env:FULL_MODE,Env:FULL_K_FOLDS

$env:FULL_MODE='kfold_eval'
$env:FULL_INIT_CHECKPOINT='experiments/checkpoints/cdae_denoise_focus.npz'
& "C:\Program Files\Git\bin\bash.exe" scripts/run_full_dataset.sh
Remove-Item Env:FULL_MODE,Env:FULL_INIT_CHECKPOINT
```

## Experiment Logging Requirement
The project includes a log template at:
- `experiments/logs/experiment_log_template.csv`

Primary run log generated by code:
- `experiments/logs/experiment_runs.csv`

Logged fields include:
- experiment number
- denoiser hyperparameters
- noise configuration
- train/validation/test denoising metrics (MSE/RMSE/PSNR, optional SSIM)
- train/validation/test classification accuracy
- checkpoint paths and run notes

## Denoising Results Snapshot

Recent measured runs from `experiments/logs/experiment_runs.csv`:

- Latest best full-split run (id=29, 5k/1k subset, single mode, noise_std=0.05):
	- train_psnr = 26.2391 dB
	- val_psnr = 26.2289 dB
	- test_psnr = 26.2388 dB
	- classification_train_accuracy = 0.8316
	- classification_val_accuracy = 0.3780
	- classification_test_accuracy = 0.3950
	- calibrated_skip_weight = 0.95

- Baseline FC run (id=1, 10k/2.5k subset):
	- val_psnr = 13.6659 dB
	- test_psnr = 13.7179 dB
- Upgraded Conv run (id=5, quick benchmark on 1k/500 subset):
	- val_psnr = 20.7215 dB
	- test_psnr = 20.6975 dB
	- classification_test_accuracy = 0.2720
