# CS-4375 Project: NumPy Denoising Autoencoder (DAE)

## Team Members
- Casey Nguyen (CXN220034)
- Nguyen Phuc Do (NPD220001)

## Project Topic
Image denoising using custom NumPy autoencoders on CIFAR-10.

## Technique Summary
- Add controlled noise to clean images.
- Train encoder-decoder network to reconstruct clean images from noisy inputs.
- Train using **MSE loss**.
- Use **K-Fold Cross Validation** (default K=5) for more reliable performance estimates.
- Train a custom **NumPy Softmax Classifier** on denoised images to predict object class.

## Algorithms Implemented
- Custom NumPy Conv2D (forward + backward + updates)
- Custom NumPy convolutional DAE (encoder/decoder + backprop)
- Custom NumPy fully connected DAE (configurable): input_dim -> hidden -> bottleneck -> hidden -> input_dim
- Custom NumPy Softmax Regression classifier (multiclass prediction)

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
	- model: `fc` (fully connected DAE, 1024 -> 256 bottleneck)
	- train/test subset: `10000/2500`
	- image size: `32x32` (no downsampling)
	- noise: gaussian with adjustable std
- Keeps denoising as the primary objective while reporting classification accuracy by default (`FULL_RUN_CLASSIFICATION=1`)
- Saves checkpoint(s), denoising figure(s), loss curve(s), summary table(s), and prediction CSV(s)
- Default checkpoint: `experiments/checkpoints/cdae_denoise_focus.npz`
- Default denoising figure: `reports/figures/denoise_focus_preview.png`
- Default loss curve: `reports/figures/denoise_focus_loss_curve.png`
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
FULL_MAX_SAMPLES=4000 FULL_MAX_TEST_SAMPLES=1000 bash scripts/run_full_dataset.sh
FULL_NOISE_STD=0.2 bash scripts/run_full_dataset.sh
FULL_RESIZE_TO=32 bash scripts/run_full_dataset.sh
FULL_RUN_CLASSIFICATION=0 bash scripts/run_full_dataset.sh
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
- train/validation/test denoising metrics (MSE/RMSE/PSNR)
- train/validation/test classification accuracy
- checkpoint paths and run notes
