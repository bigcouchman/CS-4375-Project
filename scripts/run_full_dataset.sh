#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing virtual environment at .venv/bin/activate"
  echo "Create it and install dependencies with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python3 -m pip install --upgrade pip"
  echo "  python3 -m pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate

FULL_K_FOLDS="${FULL_K_FOLDS:-5}"
FULL_EPOCHS="${FULL_EPOCHS:-8}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-64}"
FULL_LEARNING_RATE="${FULL_LEARNING_RATE:-0.0003}"
FULL_LR_DECAY="${FULL_LR_DECAY:-0.95}"
FULL_LR_DECAY_EVERY="${FULL_LR_DECAY_EVERY:-1}"
FULL_WEIGHT_DECAY="${FULL_WEIGHT_DECAY:-0.00005}"
FULL_EARLY_STOPPING_PATIENCE="${FULL_EARLY_STOPPING_PATIENCE:-3}"
FULL_MIN_DELTA="${FULL_MIN_DELTA:-0.0}"
FULL_MAX_SAMPLES="${FULL_MAX_SAMPLES:-50000}"
FULL_MAX_TEST_SAMPLES="${FULL_MAX_TEST_SAMPLES:-10000}"
FULL_LATENT_CHANNELS="${FULL_LATENT_CHANNELS:-auto}"
FULL_NOISE_TYPE="${FULL_NOISE_TYPE:-gaussian}"
FULL_NOISE_STD="${FULL_NOISE_STD:-0.08}"
FULL_INIT_CHECKPOINT="${FULL_INIT_CHECKPOINT:-experiments/checkpoints/cdae_finetuned.npz}"
FULL_CHECKPOINT_PREFIX="${FULL_CHECKPOINT_PREFIX:-experiments/checkpoints/cdae_full_kfold}"
FULL_RUN_CLASSIFICATION="${FULL_RUN_CLASSIFICATION:-1}"
FULL_CLASSIFIER_EPOCHS="${FULL_CLASSIFIER_EPOCHS:-20}"
FULL_CLASSIFIER_BATCH_SIZE="${FULL_CLASSIFIER_BATCH_SIZE:-128}"
FULL_CLASSIFIER_LR="${FULL_CLASSIFIER_LR:-0.05}"
FULL_CLASSIFIER_WEIGHT_DECAY="${FULL_CLASSIFIER_WEIGHT_DECAY:-0.00005}"
FULL_FIGURE_NAME="${FULL_FIGURE_NAME:-full_kfold_preview.png}"
FULL_NUM_FIGURE_IMAGES="${FULL_NUM_FIGURE_IMAGES:-8}"
FULL_PREDICTIONS_OUTPUT="${FULL_PREDICTIONS_OUTPUT:-reports/tables/full_kfold_predictions.csv}"
FULL_NUM_PREDICTION_SAMPLES="${FULL_NUM_PREDICTION_SAMPLES:-$FULL_MAX_TEST_SAMPLES}"
FULL_SUMMARY_OUTPUT="${FULL_SUMMARY_OUTPUT:-reports/tables/full_kfold_summary_latest.csv}"
FULL_APPEND_FINAL_RESULTS="${FULL_APPEND_FINAL_RESULTS:-1}"

START_EXPERIMENT_ID="$(( $(python3 - <<'PY'
import csv
from pathlib import Path

log_path = Path('experiments/logs/experiment_runs.csv')
if not log_path.exists():
    print(1)
else:
    with log_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    max_id = 0
    for row in rows:
        try:
            max_id = max(max_id, int(row.get('experiment_id', 0)))
        except (TypeError, ValueError):
            pass
    print(max_id + 1)
PY
) ))"

args=(
  --mode kfold
  --k-folds "$FULL_K_FOLDS"
  --epochs "$FULL_EPOCHS"
  --batch-size "$FULL_BATCH_SIZE"
  --learning-rate "$FULL_LEARNING_RATE"
  --lr-decay "$FULL_LR_DECAY"
  --lr-decay-every "$FULL_LR_DECAY_EVERY"
  --weight-decay "$FULL_WEIGHT_DECAY"
  --early-stopping-patience "$FULL_EARLY_STOPPING_PATIENCE"
  --min-delta "$FULL_MIN_DELTA"
  --noise-type "$FULL_NOISE_TYPE"
  --noise-std "$FULL_NOISE_STD"
  --max-samples "$FULL_MAX_SAMPLES"
  --max-test-samples "$FULL_MAX_TEST_SAMPLES"
  --save-checkpoint "${FULL_CHECKPOINT_PREFIX}.npz"
  --save-figure
  --figure-name "$FULL_FIGURE_NAME"
  --num-figure-images "$FULL_NUM_FIGURE_IMAGES"
)

if [[ -d "data/cifar-10-batches-py" ]]; then
  args+=(--no-download-dataset)
else
  args+=(--download-dataset)
fi

if [[ -f "$FULL_INIT_CHECKPOINT" ]]; then
  if [[ "$FULL_LATENT_CHANNELS" == "auto" ]]; then
    FULL_LATENT_CHANNELS="$(
      python3 - <<PY
import numpy as np
checkpoint = np.load("$FULL_INIT_CHECKPOINT", allow_pickle=False)
print(int(checkpoint["meta_latent_channels"].ravel()[0]))
PY
    )"
  fi
  args+=(--resume-checkpoint "$FULL_INIT_CHECKPOINT")
elif [[ "$FULL_LATENT_CHANNELS" == "auto" ]]; then
  FULL_LATENT_CHANNELS="32"
fi

args+=(--latent-channels "$FULL_LATENT_CHANNELS")

if [[ "$FULL_RUN_CLASSIFICATION" -eq 1 ]]; then
  args+=(
    --run-classification
    --classifier-epochs "$FULL_CLASSIFIER_EPOCHS"
    --classifier-batch-size "$FULL_CLASSIFIER_BATCH_SIZE"
    --classifier-learning-rate "$FULL_CLASSIFIER_LR"
    --classifier-weight-decay "$FULL_CLASSIFIER_WEIGHT_DECAY"
    --save-predictions
    --predictions-output "$FULL_PREDICTIONS_OUTPUT"
    --num-prediction-samples "$FULL_NUM_PREDICTION_SAMPLES"
  )
fi

echo "Running one-command full-dataset K-fold pipeline..."
python3 -m src.main "${args[@]}"

python3 - <<PY
import csv
from pathlib import Path

start_id = int("$START_EXPERIMENT_ID")
append_final_results = int("$FULL_APPEND_FINAL_RESULTS")
summary_output_path = Path("$FULL_SUMMARY_OUTPUT")

log_path = Path("experiments/logs/experiment_runs.csv")
if not log_path.exists():
    raise SystemExit("No run log found at experiments/logs/experiment_runs.csv")

with log_path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

new_rows = []
for row in rows:
    try:
        experiment_id = int(row.get("experiment_id", "0"))
    except (TypeError, ValueError):
        continue
    if experiment_id >= start_id and row.get("mode", "") == "kfold":
        new_rows.append(row)

if not new_rows:
    raise SystemExit("No new K-fold rows found for this run.")

print("\n===== Full K-Fold Results (Current Run) =====")
for row in new_rows:
    print(
        f"Fold {row.get('fold', '')}: "
        f"train_mse={row.get('train_mse', '')} "
        f"val_mse={row.get('val_mse', '')} "
        f"test_mse={row.get('test_mse', '')} "
        f"train_psnr={row.get('train_psnr', '')} "
        f"val_psnr={row.get('val_psnr', '')} "
        f"test_psnr={row.get('test_psnr', '')} "
        f"cls_test_acc={row.get('classification_test_accuracy', '')}"
    )

def aggregate(key: str):
    values = []
    for row in new_rows:
        raw = (row.get(key, "") or "").strip()
        if raw == "":
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    return mean, std

metrics = {
    "train_mse": aggregate("train_mse"),
    "val_mse": aggregate("val_mse"),
    "test_mse": aggregate("test_mse"),
    "train_rmse": aggregate("train_rmse"),
    "val_rmse": aggregate("val_rmse"),
    "test_rmse": aggregate("test_rmse"),
    "train_psnr": aggregate("train_psnr"),
    "val_psnr": aggregate("val_psnr"),
    "test_psnr": aggregate("test_psnr"),
    "classification_train_accuracy": aggregate("classification_train_accuracy"),
    "classification_val_accuracy": aggregate("classification_val_accuracy"),
    "classification_test_accuracy": aggregate("classification_test_accuracy"),
}

best_row = min(
    new_rows,
    key=lambda row: float((row.get("val_mse", "inf") or "inf")),
)
best_fold = best_row.get("fold", "")

checkpoint_root = Path("$FULL_CHECKPOINT_PREFIX")
best_checkpoint_path = checkpoint_root.parent / f"{checkpoint_root.name}_best.npz"
best_checkpoint = str(best_checkpoint_path) if best_checkpoint_path.exists() else ""

summary = {
    "run_type": "full_dataset_kfold",
    "fold_count": len(new_rows),
    "epochs": best_row.get("epochs_requested", ""),
    "batch_size": best_row.get("batch_size", ""),
    "learning_rate": best_row.get("learning_rate", ""),
    "noise_type": best_row.get("noise_type", ""),
    "noise_std": best_row.get("noise_std", ""),
    "latent_channels": best_row.get("latent_channels", ""),
    "max_samples": best_row.get("max_samples", ""),
    "max_test_samples": best_row.get("max_test_samples", ""),
    "best_fold": best_fold,
    "best_checkpoint_path": best_checkpoint,
    "train_mse_mean": metrics["train_mse"][0],
    "train_mse_std": metrics["train_mse"][1],
    "val_mse_mean": metrics["val_mse"][0],
    "val_mse_std": metrics["val_mse"][1],
    "test_mse_mean": metrics["test_mse"][0],
    "test_mse_std": metrics["test_mse"][1],
    "train_rmse_mean": metrics["train_rmse"][0],
    "train_rmse_std": metrics["train_rmse"][1],
    "val_rmse_mean": metrics["val_rmse"][0],
    "val_rmse_std": metrics["val_rmse"][1],
    "test_rmse_mean": metrics["test_rmse"][0],
    "test_rmse_std": metrics["test_rmse"][1],
    "train_psnr_mean": metrics["train_psnr"][0],
    "train_psnr_std": metrics["train_psnr"][1],
    "val_psnr_mean": metrics["val_psnr"][0],
    "val_psnr_std": metrics["val_psnr"][1],
    "test_psnr_mean": metrics["test_psnr"][0],
    "test_psnr_std": metrics["test_psnr"][1],
    "classification_train_accuracy_mean": metrics["classification_train_accuracy"][0],
    "classification_train_accuracy_std": metrics["classification_train_accuracy"][1],
    "classification_val_accuracy_mean": metrics["classification_val_accuracy"][0],
    "classification_val_accuracy_std": metrics["classification_val_accuracy"][1],
    "classification_test_accuracy_mean": metrics["classification_test_accuracy"][0],
    "classification_test_accuracy_std": metrics["classification_test_accuracy"][1],
}

summary_output_path.parent.mkdir(parents=True, exist_ok=True)
with summary_output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

print("\nK-Fold Means (+/- std):")
for metric_name in (
    "train_mse",
    "val_mse",
    "test_mse",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "train_psnr",
    "val_psnr",
    "test_psnr",
    "classification_train_accuracy",
    "classification_val_accuracy",
    "classification_test_accuracy",
):
    mean, std = metrics[metric_name]
    if mean == "":
        print(f"{metric_name}_mean: ")
        print(f"{metric_name}_std: ")
    else:
        print(f"{metric_name}_mean: {mean:.6f}")
        print(f"{metric_name}_std: {std:.6f}")

if append_final_results == 1:
    final_path = Path("reports/tables/final_results.csv")
    default_fieldnames = [
        "experiment_id", "run_type", "mode", "k_folds", "epochs", "batch_size",
        "learning_rate", "noise_type", "noise_std", "latent_channels", "max_samples",
        "max_test_samples", "train_mse", "val_mse", "test_mse", "train_psnr", "val_psnr",
        "test_psnr", "classification_train_accuracy", "classification_val_accuracy",
        "classification_test_accuracy", "notes"
    ]

    existing_rows = []
    if final_path.exists():
        with final_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = reader.fieldnames or default_fieldnames
    else:
        fieldnames = default_fieldnames

    for key in default_fieldnames:
        if key not in fieldnames:
            fieldnames.append(key)

    next_id = 1
    for row in existing_rows:
        try:
            next_id = max(next_id, int(row.get("experiment_id", "0")) + 1)
        except (TypeError, ValueError):
            pass

    final_row = {
        "experiment_id": str(next_id),
        "run_type": "full_dataset_kfold",
        "mode": "kfold",
        "k_folds": str(summary["fold_count"]),
        "epochs": summary["epochs"],
        "batch_size": summary["batch_size"],
        "learning_rate": summary["learning_rate"],
        "noise_type": summary["noise_type"],
        "noise_std": summary["noise_std"],
        "latent_channels": summary["latent_channels"],
        "max_samples": summary["max_samples"],
        "max_test_samples": summary["max_test_samples"],
        "train_mse": "" if metrics["train_mse"][0] == "" else f"{metrics['train_mse'][0]:.6f}",
        "val_mse": "" if metrics["val_mse"][0] == "" else f"{metrics['val_mse'][0]:.6f}",
        "test_mse": "" if metrics["test_mse"][0] == "" else f"{metrics['test_mse'][0]:.6f}",
        "train_psnr": "" if metrics["train_psnr"][0] == "" else f"{metrics['train_psnr'][0]:.6f}",
        "val_psnr": "" if metrics["val_psnr"][0] == "" else f"{metrics['val_psnr'][0]:.6f}",
        "test_psnr": "" if metrics["test_psnr"][0] == "" else f"{metrics['test_psnr'][0]:.6f}",
        "classification_train_accuracy": "" if metrics["classification_train_accuracy"][0] == "" else f"{metrics['classification_train_accuracy'][0]:.6f}",
        "classification_val_accuracy": "" if metrics["classification_val_accuracy"][0] == "" else f"{metrics['classification_val_accuracy'][0]:.6f}",
        "classification_test_accuracy": "" if metrics["classification_test_accuracy"][0] == "" else f"{metrics['classification_test_accuracy'][0]:.6f}",
        "notes": f"full dataset K-fold means; best_fold={best_fold}; best_checkpoint={best_checkpoint or 'n/a'}",
    }

    with final_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(final_row)

print("\nArtifacts:")
print("- experiments/logs/experiment_runs.csv")
print(f"- {summary_output_path}")
print("- reports/figures/full_kfold_preview_fold*.png")
if int("$FULL_RUN_CLASSIFICATION") == 1:
    print("- reports/tables/full_kfold_predictions_fold*.csv")
if best_checkpoint:
    print(f"- {best_checkpoint}")
print("- reports/tables/final_results.csv")
PY
