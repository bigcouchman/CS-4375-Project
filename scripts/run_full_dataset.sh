#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
else
    echo "Missing project virtual environment at .venv"
    echo "Create it and install dependencies with:"
    echo "  macOS/Linux:"
    echo "    python3 -m venv .venv"
    echo "    .venv/bin/python -m pip install --upgrade pip"
    echo "    .venv/bin/python -m pip install -r requirements.txt"
    echo "  Windows (PowerShell):"
    echo "    python -m venv .venv"
    echo "    .venv\\Scripts\\python.exe -m pip install --upgrade pip"
    echo "    .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    exit 1
fi

echo "[ENV] Using interpreter: $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import importlib.util

if importlib.util.find_spec("torchvision") is None:
        raise SystemExit(
                "[ERROR] torchvision is not installed in .venv. "
                "Install it with the project venv interpreter before running this script."
        )

print("[ENV] torchvision is installed in .venv")
PY

FULL_MODE="${FULL_MODE:-single}"
FULL_K_FOLDS="${FULL_K_FOLDS:-5}"
FULL_EPOCHS="${FULL_EPOCHS:-80}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_LEARNING_RATE="${FULL_LEARNING_RATE:-0.001}"
FULL_LR_DECAY="${FULL_LR_DECAY:-0.995}"
FULL_LR_DECAY_EVERY="${FULL_LR_DECAY_EVERY:-1}"
FULL_WEIGHT_DECAY="${FULL_WEIGHT_DECAY:-0.0}"
FULL_EARLY_STOPPING_PATIENCE="${FULL_EARLY_STOPPING_PATIENCE:-12}"
FULL_MIN_DELTA="${FULL_MIN_DELTA:-0.0}"
FULL_MAX_SAMPLES="${FULL_MAX_SAMPLES:-5000}"
FULL_MAX_TEST_SAMPLES="${FULL_MAX_TEST_SAMPLES:-1000}"
FULL_VAL_RATIO="${FULL_VAL_RATIO:-0.1}"
FULL_MODEL_TYPE="${FULL_MODEL_TYPE:-conv}"
FULL_LATENT_CHANNELS="${FULL_LATENT_CHANNELS:-96}"
FULL_CONV_SKIP_CONNECTION_WEIGHT="${FULL_CONV_SKIP_CONNECTION_WEIGHT:-0.8}"
FULL_FC_HIDDEN_DIM="${FULL_FC_HIDDEN_DIM:-1024}"
FULL_FC_BOTTLENECK_DIM="${FULL_FC_BOTTLENECK_DIM:-256}"
FULL_L1_WEIGHT="${FULL_L1_WEIGHT:-0.05}"
FULL_RESIZE_TO="${FULL_RESIZE_TO:-32}"
FULL_RANDOM_SUBSET="${FULL_RANDOM_SUBSET:-1}"
FULL_NOISE_TYPE="${FULL_NOISE_TYPE:-gaussian}"
FULL_NOISE_STD="${FULL_NOISE_STD:-0.05}"
FULL_SAMPLE_NOISE_PER_BATCH="${FULL_SAMPLE_NOISE_PER_BATCH:-0}"
FULL_BATCH_NOISE_STD_OPTIONS="${FULL_BATCH_NOISE_STD_OPTIONS:-0.04,0.05,0.06}"
FULL_TRAIN_METRICS_MAX_SAMPLES="${FULL_TRAIN_METRICS_MAX_SAMPLES:-1024}"
FULL_EPOCH_TRAIN_SUBSET_MIN="${FULL_EPOCH_TRAIN_SUBSET_MIN:-2500}"
FULL_EPOCH_TRAIN_SUBSET_MAX="${FULL_EPOCH_TRAIN_SUBSET_MAX:-3000}"
FULL_TRACK_SSIM="${FULL_TRACK_SSIM:-0}"
FULL_INIT_CHECKPOINT="${FULL_INIT_CHECKPOINT:-}"
FULL_CHECKPOINT_PREFIX="${FULL_CHECKPOINT_PREFIX:-experiments/checkpoints/cdae_denoise_focus}"
FULL_RUN_CLASSIFICATION="${FULL_RUN_CLASSIFICATION:-1}"
FULL_CLASSIFIER_EPOCHS="${FULL_CLASSIFIER_EPOCHS:-40}"
FULL_CLASSIFIER_BATCH_SIZE="${FULL_CLASSIFIER_BATCH_SIZE:-32}"
FULL_CLASSIFIER_LR="${FULL_CLASSIFIER_LR:-0.005}"
FULL_CLASSIFIER_WEIGHT_DECAY="${FULL_CLASSIFIER_WEIGHT_DECAY:-0.001}"
FULL_CLASSIFIER_HIDDEN_DIMS="${FULL_CLASSIFIER_HIDDEN_DIMS:-64,32}"
FULL_CLASSIFIER_DROPOUT="${FULL_CLASSIFIER_DROPOUT:-0.5}"
FULL_FIGURE_NAME="${FULL_FIGURE_NAME:-denoise_focus_preview.png}"
FULL_SAVE_LOSS_CURVE="${FULL_SAVE_LOSS_CURVE:-1}"
FULL_LOSS_CURVE_NAME="${FULL_LOSS_CURVE_NAME:-denoise_focus_loss_curve.png}"
FULL_LOSS_CURVE_UPDATE_EVERY="${FULL_LOSS_CURVE_UPDATE_EVERY:-1}"
FULL_KFOLD_EVAL_OUTPUT="${FULL_KFOLD_EVAL_OUTPUT:-reports/tables/kfold_eval_summary_latest.csv}"
FULL_NUM_FIGURE_IMAGES="${FULL_NUM_FIGURE_IMAGES:-10}"
FULL_PREDICTIONS_OUTPUT="${FULL_PREDICTIONS_OUTPUT:-reports/tables/full_predictions.csv}"
FULL_NUM_PREDICTION_SAMPLES="${FULL_NUM_PREDICTION_SAMPLES:-$FULL_MAX_TEST_SAMPLES}"
FULL_SUMMARY_OUTPUT="${FULL_SUMMARY_OUTPUT:-reports/tables/denoise_focus_summary_latest.csv}"
FULL_APPEND_FINAL_RESULTS="${FULL_APPEND_FINAL_RESULTS:-1}"
FULL_CLEAN_OUTPUTS="${FULL_CLEAN_OUTPUTS:-1}"

if [[ "$FULL_MODE" != "single" && "$FULL_MODE" != "kfold" && "$FULL_MODE" != "kfold_eval" ]]; then
    echo "FULL_MODE must be one of: single, kfold, kfold_eval."
    exit 1
fi

if [[ "$FULL_MODE" == "kfold_eval" ]]; then
    if [[ ! -f "$FULL_INIT_CHECKPOINT" ]]; then
        echo "FULL_MODE=kfold_eval requires FULL_INIT_CHECKPOINT to point to an existing .npz checkpoint."
        exit 1
    fi
    FULL_RUN_CLASSIFICATION="0"
    FULL_SAVE_LOSS_CURVE="0"
fi

remove_artifact_pattern() {
    local pattern="$1"
    local artifact

    while IFS= read -r artifact; do
        if [[ "$(basename "$artifact")" == ".gitkeep" ]]; then
            continue
        fi
        rm -f "$artifact"
    done < <(compgen -G "$pattern" || true)
}

if [[ "$FULL_CLEAN_OUTPUTS" -eq 1 ]]; then
    figure_stem="${FULL_FIGURE_NAME%.*}"
    figure_ext="${FULL_FIGURE_NAME##*.}"
    if [[ "$figure_stem" == "$FULL_FIGURE_NAME" ]]; then
        figure_ext="png"
    fi

    loss_stem="${FULL_LOSS_CURVE_NAME%.*}"
    loss_ext="${FULL_LOSS_CURVE_NAME##*.}"
    if [[ "$loss_stem" == "$FULL_LOSS_CURVE_NAME" ]]; then
        loss_ext="png"
    fi

    pred_stem="${FULL_PREDICTIONS_OUTPUT%.*}"
    pred_ext="${FULL_PREDICTIONS_OUTPUT##*.}"
    if [[ "$pred_stem" == "$FULL_PREDICTIONS_OUTPUT" ]]; then
        pred_ext="csv"
    fi

    remove_artifact_pattern "reports/figures/$FULL_FIGURE_NAME"
    remove_artifact_pattern "reports/figures/${figure_stem}_fold*.${figure_ext}"
    remove_artifact_pattern "reports/figures/$FULL_LOSS_CURVE_NAME"
    remove_artifact_pattern "reports/figures/${loss_stem}_fold*.${loss_ext}"

    remove_artifact_pattern "$FULL_SUMMARY_OUTPUT"
    remove_artifact_pattern "$FULL_PREDICTIONS_OUTPUT"
    remove_artifact_pattern "${pred_stem}_fold*.${pred_ext}"
    remove_artifact_pattern "$FULL_KFOLD_EVAL_OUTPUT"
fi

START_EXPERIMENT_ID="$(( $("$PYTHON_BIN" - <<'PY'
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
    --mode "$FULL_MODE"
    --kfold-eval-output "$FULL_KFOLD_EVAL_OUTPUT"
  --epochs "$FULL_EPOCHS"
  --batch-size "$FULL_BATCH_SIZE"
  --learning-rate "$FULL_LEARNING_RATE"
  --lr-decay "$FULL_LR_DECAY"
  --lr-decay-every "$FULL_LR_DECAY_EVERY"
  --weight-decay "$FULL_WEIGHT_DECAY"
  --early-stopping-patience "$FULL_EARLY_STOPPING_PATIENCE"
  --min-delta "$FULL_MIN_DELTA"
    --val-ratio "$FULL_VAL_RATIO"
  --noise-type "$FULL_NOISE_TYPE"
  --noise-std "$FULL_NOISE_STD"
    --l1-weight "$FULL_L1_WEIGHT"
    --batch-noise-std-options "$FULL_BATCH_NOISE_STD_OPTIONS"
    --train-metrics-max-samples "$FULL_TRAIN_METRICS_MAX_SAMPLES"
        --epoch-train-subset-min "$FULL_EPOCH_TRAIN_SUBSET_MIN"
        --epoch-train-subset-max "$FULL_EPOCH_TRAIN_SUBSET_MAX"
  --max-samples "$FULL_MAX_SAMPLES"
  --max-test-samples "$FULL_MAX_TEST_SAMPLES"
    --model-type "$FULL_MODEL_TYPE"
    --resize-to "$FULL_RESIZE_TO"
  --save-checkpoint "${FULL_CHECKPOINT_PREFIX}.npz"
  --save-figure
  --figure-name "$FULL_FIGURE_NAME"
    --loss-curve-update-every "$FULL_LOSS_CURVE_UPDATE_EVERY"
  --num-figure-images "$FULL_NUM_FIGURE_IMAGES"
)

if [[ "$FULL_MODE" == "kfold" || "$FULL_MODE" == "kfold_eval" ]]; then
    args+=(--k-folds "$FULL_K_FOLDS")
fi

if [[ "$FULL_MODEL_TYPE" == "fc" ]]; then
    args+=(
        --fc-hidden-dim "$FULL_FC_HIDDEN_DIM"
        --fc-bottleneck-dim "$FULL_FC_BOTTLENECK_DIM"
    )
else
    args+=(
        --latent-channels "$FULL_LATENT_CHANNELS"
        --conv-skip-connection-weight "$FULL_CONV_SKIP_CONNECTION_WEIGHT"
    )
fi

if [[ "$FULL_RANDOM_SUBSET" -eq 1 ]]; then
    args+=(--random-subset)
else
    args+=(--no-random-subset)
fi

if [[ "$FULL_SAMPLE_NOISE_PER_BATCH" -eq 1 ]]; then
    args+=(--sample-noise-per-batch)
else
    args+=(--no-sample-noise-per-batch)
fi

if [[ "$FULL_TRACK_SSIM" -eq 1 ]]; then
    args+=(--track-ssim)
else
    args+=(--no-track-ssim)
fi

if [[ "$FULL_SAVE_LOSS_CURVE" -eq 1 ]]; then
    args+=(--save-loss-curve --loss-curve-name "$FULL_LOSS_CURVE_NAME")
fi

if [[ -d "data/cifar-10-batches-py" ]]; then
  args+=(--no-download-dataset)
else
  args+=(--download-dataset)
fi

if [[ -f "$FULL_INIT_CHECKPOINT" ]]; then
  args+=(--resume-checkpoint "$FULL_INIT_CHECKPOINT")
fi

if [[ "$FULL_RUN_CLASSIFICATION" -eq 1 ]]; then
  args+=(
    --run-classification
    --classifier-epochs "$FULL_CLASSIFIER_EPOCHS"
    --classifier-batch-size "$FULL_CLASSIFIER_BATCH_SIZE"
    --classifier-learning-rate "$FULL_CLASSIFIER_LR"
    --classifier-weight-decay "$FULL_CLASSIFIER_WEIGHT_DECAY"
    --classifier-hidden-dims "$FULL_CLASSIFIER_HIDDEN_DIMS"
    --classifier-dropout "$FULL_CLASSIFIER_DROPOUT"
    --save-predictions
    --predictions-output "$FULL_PREDICTIONS_OUTPUT"
    --num-prediction-samples "$FULL_NUM_PREDICTION_SAMPLES"
  )
fi

echo "Running one-command pipeline with .venv Python..."
"$PYTHON_BIN" -m src.main "${args[@]}"

"$PYTHON_BIN" - <<PY
import csv
from pathlib import Path

start_id = int("$START_EXPERIMENT_ID")
append_final_results = int("$FULL_APPEND_FINAL_RESULTS")
summary_output_path = Path("$FULL_SUMMARY_OUTPUT")
target_mode = "$FULL_MODE"

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
    if experiment_id >= start_id and row.get("mode", "") == target_mode:
        new_rows.append(row)

if not new_rows:
    raise SystemExit("No new run rows found for this execution.")

print(f"\n===== Script Results (mode={target_mode}) =====")
for row in new_rows:
    print(
        f"mode={row.get('mode', '')} fold={row.get('fold', '')}: "
        f"train_mse={row.get('train_mse', '')} "
        f"val_mse={row.get('val_mse', '')} "
        f"test_mse={row.get('test_mse', '')} "
        f"train_psnr={row.get('train_psnr', '')} "
        f"val_psnr={row.get('val_psnr', '')} "
        f"test_psnr={row.get('test_psnr', '')} "
        f"val_ssim={row.get('val_ssim', '')} "
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
    "train_ssim": aggregate("train_ssim"),
    "val_ssim": aggregate("val_ssim"),
    "test_ssim": aggregate("test_ssim"),
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
if target_mode == "kfold":
    best_checkpoint_path = checkpoint_root.parent / f"{checkpoint_root.name}_best.npz"
elif target_mode == "kfold_eval":
    best_checkpoint_path = Path("$FULL_INIT_CHECKPOINT")
else:
    best_checkpoint_path = Path(f"{checkpoint_root}.npz")
best_checkpoint = str(best_checkpoint_path) if best_checkpoint_path.exists() else ""

summary = {
    "run_type": f"script_{target_mode}",
    "fold_count": len(new_rows),
    "epochs": best_row.get("epochs_requested", ""),
    "batch_size": best_row.get("batch_size", ""),
    "learning_rate": best_row.get("learning_rate", ""),
    "noise_type": best_row.get("noise_type", ""),
    "noise_std": best_row.get("noise_std", ""),
    "l1_weight": best_row.get("l1_weight", ""),
    "sample_noise_per_batch": best_row.get("sample_noise_per_batch", ""),
    "batch_noise_std_options": best_row.get("batch_noise_std_options", ""),
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
    "train_ssim_mean": metrics["train_ssim"][0],
    "train_ssim_std": metrics["train_ssim"][1],
    "val_ssim_mean": metrics["val_ssim"][0],
    "val_ssim_std": metrics["val_ssim"][1],
    "test_ssim_mean": metrics["test_ssim"][0],
    "test_ssim_std": metrics["test_ssim"][1],
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

print("\nRun Means (+/- std):")
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
    "train_ssim",
    "val_ssim",
    "test_ssim",
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
        "test_psnr", "train_ssim", "val_ssim", "test_ssim",
        "classification_train_accuracy", "classification_val_accuracy",
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
        "run_type": f"script_{target_mode}",
        "mode": target_mode,
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
        "train_ssim": "" if metrics["train_ssim"][0] == "" else f"{metrics['train_ssim'][0]:.6f}",
        "val_ssim": "" if metrics["val_ssim"][0] == "" else f"{metrics['val_ssim'][0]:.6f}",
        "test_ssim": "" if metrics["test_ssim"][0] == "" else f"{metrics['test_ssim'][0]:.6f}",
        "classification_train_accuracy": "" if metrics["classification_train_accuracy"][0] == "" else f"{metrics['classification_train_accuracy'][0]:.6f}",
        "classification_val_accuracy": "" if metrics["classification_val_accuracy"][0] == "" else f"{metrics['classification_val_accuracy'][0]:.6f}",
        "classification_test_accuracy": "" if metrics["classification_test_accuracy"][0] == "" else f"{metrics['classification_test_accuracy'][0]:.6f}",
        "notes": f"script run means; mode={target_mode}; best_fold={best_fold}; best_checkpoint={best_checkpoint or 'n/a'}",
    }

    with final_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(final_row)

print("\nArtifacts:")
print("- experiments/logs/experiment_runs.csv")
print(f"- {summary_output_path}")
if target_mode == "kfold_eval":
    print(f"- {Path('$FULL_KFOLD_EVAL_OUTPUT')}")
if target_mode == "kfold":
    print("- reports/figures/*_fold*.png")
else:
    print("- reports/figures/*.png")
if int("$FULL_RUN_CLASSIFICATION") == 1:
    if target_mode == "kfold":
        print("- reports/tables/*_fold*.csv")
    else:
        print("- reports/tables/*.csv")
if best_checkpoint:
    print(f"- {best_checkpoint}")
print("- reports/tables/final_results.csv")
PY
