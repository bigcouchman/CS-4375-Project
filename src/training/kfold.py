from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
from sklearn.model_selection import KFold

from src.classification import run_softmax_classification_on_denoised
from src.evaluation.visualize import save_denoising_grid

from .trainer import train_fold


def _fold_output_path(base_path: str | Path, fold_number: int, default_suffix: str) -> Path:
    output_path = Path(base_path)
    suffix = output_path.suffix or default_suffix
    return output_path.with_name(f"{output_path.stem}_fold{fold_number}{suffix}")


def _write_prediction_samples(
    output_path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None,
    max_rows: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = min(max_rows, y_true.shape[0], y_pred.shape[0])
    label_names = class_names or [str(index) for index in range(int(np.max(y_true)) + 1)]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["sample_index", "true_label", "predicted_label", "correct"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample_index in range(row_count):
            true_index = int(y_true[sample_index])
            pred_index = int(y_pred[sample_index])

            writer.writerow(
                {
                    "sample_index": sample_index,
                    "true_label": label_names[true_index],
                    "predicted_label": label_names[pred_index],
                    "correct": int(true_index == pred_index),
                }
            )


def run_kfold_experiment(
    clean_images: np.ndarray,
    noisy_images: np.ndarray,
    model_builder,
    k_folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = 0.0,
    lr_decay: float = 1.0,
    lr_decay_every: int = 0,
    early_stopping_patience: int = 0,
    min_delta: float = 0.0,
    clean_test: np.ndarray | None = None,
    noisy_test: np.ndarray | None = None,
    checkpoint_prefix: str | None = None,
    train_labels: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    run_classification: bool = False,
    classifier_epochs: int = 20,
    classifier_batch_size: int = 64,
    classifier_learning_rate: float = 0.1,
    classifier_weight_decay: float = 0.0,
    classifier_seed: int = 123,
    save_figures: bool = False,
    figure_output_path: str | Path | None = None,
    num_figure_images: int = 8,
    save_predictions: bool = False,
    predictions_output_path: str | Path | None = None,
    num_prediction_samples: int = 100,
    class_names: list[str] | None = None,
) -> list[dict[str, object]]:
    if clean_images.shape != noisy_images.shape:
        raise ValueError("Clean and noisy arrays must have matching shapes.")

    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, object]] = []
    best_val_mse = float("inf")
    best_fold = 0
    best_checkpoint_path = ""

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(clean_images), start=1):
        print(f"\n===== Fold {fold_number}/{k_folds} =====")

        clean_train = clean_images[train_idx]
        noisy_train = noisy_images[train_idx]
        clean_val = clean_images[val_idx]
        noisy_val = noisy_images[val_idx]

        model = model_builder()

        fold_result = train_fold(
            model=model,
            noisy_train=noisy_train,
            clean_train=clean_train,
            noisy_val=noisy_val,
            clean_val=clean_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + fold_number,
            weight_decay=weight_decay,
            lr_decay=lr_decay,
            lr_decay_every=lr_decay_every,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            clean_test=clean_test,
            noisy_test=noisy_test,
        )

        if run_classification and train_labels is not None:
            y_train = train_labels[train_idx]
            y_val = train_labels[val_idx]

            classification_result = run_softmax_classification_on_denoised(
                denoiser_model=model,
                noisy_train=noisy_train,
                y_train=y_train,
                noisy_val=noisy_val,
                y_val=y_val,
                epochs=classifier_epochs,
                batch_size=classifier_batch_size,
                learning_rate=classifier_learning_rate,
                weight_decay=classifier_weight_decay,
                seed=classifier_seed + fold_number,
                noisy_test=noisy_test,
                y_test=test_labels,
                denoise_batch_size=batch_size,
            )

            fold_result["classification_train_accuracy"] = float(
                classification_result["classification_train_accuracy"]
            )
            fold_result["classification_val_accuracy"] = float(
                classification_result["classification_val_accuracy"]
            )
            fold_result["classifier_final_loss"] = float(classification_result["classifier_final_loss"])

            if "classification_test_accuracy" in classification_result:
                fold_result["classification_test_accuracy"] = float(
                    classification_result["classification_test_accuracy"]
                )

            if (
                save_predictions
                and predictions_output_path is not None
                and test_labels is not None
                and "test_predictions" in classification_result
            ):
                predictions_path = _fold_output_path(
                    predictions_output_path,
                    fold_number=fold_number,
                    default_suffix=".csv",
                )
                _write_prediction_samples(
                    output_path=predictions_path,
                    y_true=np.asarray(test_labels, dtype=np.int64),
                    y_pred=np.asarray(classification_result["test_predictions"], dtype=np.int64),
                    class_names=class_names,
                    max_rows=num_prediction_samples,
                )
                fold_result["predictions_path"] = str(predictions_path)

        if save_figures and figure_output_path is not None:
            preview_count = min(num_figure_images, noisy_val.shape[0], clean_val.shape[0])
            if preview_count > 0:
                reconstructed_val = model.forward(noisy_val[:preview_count])
                figure_path = _fold_output_path(
                    figure_output_path,
                    fold_number=fold_number,
                    default_suffix=".png",
                )
                save_denoising_grid(
                    clean_images=clean_val[:preview_count],
                    noisy_images=noisy_val[:preview_count],
                    reconstructed_images=reconstructed_val,
                    output_path=figure_path,
                    num_images=preview_count,
                )
                fold_result["figure_path"] = str(figure_path)

        if checkpoint_prefix:
            checkpoint_path = Path(f"{checkpoint_prefix}_fold{fold_number}.npz")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_checkpoint"):
                model.save_checkpoint(checkpoint_path)
                fold_result["checkpoint_path"] = str(checkpoint_path)

        val_mse = float(fold_result.get("val_mse", np.inf))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_fold = fold_number
            best_checkpoint_path = str(fold_result.get("checkpoint_path", ""))

        fold_result["fold"] = fold_number
        fold_results.append(fold_result)

    for fold_result in fold_results:
        fold_result["best_fold"] = best_fold
        fold_result["is_best_fold"] = int(int(fold_result["fold"]) == best_fold)

    if checkpoint_prefix and best_checkpoint_path:
        target_best_path = Path(f"{checkpoint_prefix}_best.npz")
        target_best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_checkpoint_path, target_best_path)

        for fold_result in fold_results:
            fold_result["best_checkpoint_path"] = str(target_best_path)

    return fold_results
