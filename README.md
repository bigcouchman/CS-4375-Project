# CS-4375 Project: Convolutional Denoising Autoencoder (CDAE)

## Team Members
- Casey Nguyen (CXN220034)
- Nguyen Phuc Do (NPD220001)

## Project Topic
Image denoising using a Convolutional Denoising Autoencoder (CDAE) on CIFAR-10.

## Technique Summary
- Add controlled noise to clean images.
- Train encoder-decoder network to reconstruct clean images from noisy inputs.
- Train using **MSE loss**.
- Use **K-Fold Cross Validation** (default K=5) for more reliable performance estimates.
- Train a custom **NumPy Softmax Classifier** on denoised images to predict object class.

## Algorithms Implemented
- Custom NumPy Conv2D (forward + backward + updates)
- Custom NumPy CDAE (encoder/decoder + backprop)
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
├── pytest.ini
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

## Setup (Python3 + venv + requirements.txt)

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Download CIFAR-10

Dataset download is automatic on first run:
- Preferred path: `torchvision.datasets.CIFAR10`
- Fallback path: official CIFAR-10 Python archive from the same source

Optional (if torchvision is installed) pre-download command:

```bash
python3 -c "from torchvision import datasets; datasets.CIFAR10(root='./data', train=True, download=True); datasets.CIFAR10(root='./data', train=False, download=True)"
```

If torchvision is not installed in your environment, just run the project normally and the fallback loader will download the official Python-version archive automatically.

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

If the dataset is already cached locally, prevent re-download with:

```bash
python3 -m src.main --no-download-dataset
```

## Run

Use exactly one script for the entire project workflow:

```bash
bash scripts/run_full_dataset.sh
```

This single script runs **full-dataset K-fold** on CIFAR-10 and produces all required outputs.

Default behavior:
- Uses full dataset size: 50,000 train + 10,000 test
- Uses K-fold cross-validation (`FULL_K_FOLDS=5`)
- Initializes each fold from `experiments/checkpoints/cdae_finetuned.npz` when available
- Trains with decay/regularization/early stopping defaults tuned for this project
- Runs classification metrics per fold (train/val/test accuracy)
- Saves per-fold checkpoints: `experiments/checkpoints/cdae_full_kfold_fold*.npz`
- Saves best checkpoint: `experiments/checkpoints/cdae_full_kfold_best.npz`
- Saves per-fold denoising figures: `reports/figures/full_kfold_preview_fold*.png`
- Saves per-fold prediction CSVs: `reports/tables/full_kfold_predictions_fold*.csv`
- Logs every fold to `experiments/logs/experiment_runs.csv`
- Writes aggregate mean/std metrics to `reports/tables/full_kfold_summary_latest.csv`
- Appends a consolidated K-fold row to `reports/tables/final_results.csv`

Useful overrides:

```bash
FULL_K_FOLDS=3 bash scripts/run_full_dataset.sh
FULL_EPOCHS=5 bash scripts/run_full_dataset.sh
FULL_RUN_CLASSIFICATION=0 bash scripts/run_full_dataset.sh
FULL_NUM_PREDICTION_SAMPLES=10000 bash scripts/run_full_dataset.sh
FULL_INIT_CHECKPOINT=experiments/checkpoints/cdae_finetuned.npz bash scripts/run_full_dataset.sh
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
