from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
from sklearn.model_selection import KFold

from src.classification import run_softmax_classification_on_denoised
from src.data import add_gaussian_noise, add_salt_pepper_noise
from src.evaluation.visualize import save_denoising_grid, save_loss_curve

from .trainer import compute_metrics, train_fold


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


def _numpy_kfold_validation_indices(
    sample_count: int,
    k_folds: int,
    seed: int,
) -> list[np.ndarray]:
    if k_folds <= 1:
        raise ValueError("k_folds must be >= 2.")
    if sample_count < k_folds:
        raise ValueError("k_folds cannot be greater than number of samples.")

    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(sample_count)
    rng.shuffle(shuffled_indices)

    folds = np.array_split(shuffled_indices, k_folds)
    return [np.asarray(fold, dtype=np.int64) for fold in folds]


def _summarize_eval_folds(fold_results: list[dict[str, float]]) -> dict[str, float]:
    val_mse = np.asarray([row["val_mse"] for row in fold_results], dtype=np.float64)
    val_rmse = np.asarray([row["val_rmse"] for row in fold_results], dtype=np.float64)
    val_psnr = np.asarray([row["val_psnr"] for row in fold_results], dtype=np.float64)

    finite_psnr = val_psnr[np.isfinite(val_psnr)]
    if finite_psnr.size > 0:
        psnr_mean = float(np.mean(finite_psnr))
        psnr_std = float(np.std(finite_psnr))
    else:
        psnr_mean = float("inf")
        psnr_std = 0.0

    return {
        "val_mse_mean": float(np.mean(val_mse)),
        "val_mse_std": float(np.std(val_mse)),
        "val_rmse_mean": float(np.mean(val_rmse)),
        "val_rmse_std": float(np.std(val_rmse)),
        "val_psnr_mean": psnr_mean,
        "val_psnr_std": psnr_std,
    }


def _print_kfold_eval_report(
    fold_results: list[dict[str, float]],
    summary: dict[str, float],
) -> None:
    print("\nIEEE-Style K-Fold Evaluation (Inference Only)")
    print("Fold |     MSE      |     RMSE     |   PSNR (dB)")
    print("-----+--------------+--------------+-------------")
    for row in fold_results:
        print(
            f"{int(row['fold']):>4} | "
            f"{row['val_mse']:.8f} | "
            f"{row['val_rmse']:.8f} | "
            f"{row['val_psnr']:.6f}"
        )
    print("-----+--------------+--------------+-------------")
    print(
        "Mean+/-Std | "
        f"{summary['val_mse_mean']:.8f} +/- {summary['val_mse_std']:.8f} | "
        f"{summary['val_rmse_mean']:.8f} +/- {summary['val_rmse_std']:.8f} | "
        f"{summary['val_psnr_mean']:.6f} +/- {summary['val_psnr_std']:.6f}"
    )


def _write_kfold_eval_csv(
    output_csv_path: str | Path,
    fold_results: list[dict[str, float]],
    summary: dict[str, float],
) -> Path:
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["row_type", "fold", "val_samples", "val_mse", "val_rmse", "val_psnr"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in fold_results:
            writer.writerow(
                {
                    "row_type": "fold",
                    "fold": int(row["fold"]),
                    "val_samples": int(row["val_samples"]),
                    "val_mse": f"{row['val_mse']:.10f}",
                    "val_rmse": f"{row['val_rmse']:.10f}",
                    "val_psnr": f"{row['val_psnr']:.10f}",
                }
            )

        writer.writerow(
            {
                "row_type": "mean",
                "fold": "",
                "val_samples": "",
                "val_mse": f"{summary['val_mse_mean']:.10f}",
                "val_rmse": f"{summary['val_rmse_mean']:.10f}",
                "val_psnr": f"{summary['val_psnr_mean']:.10f}",
            }
        )
        writer.writerow(
            {
                "row_type": "std",
                "fold": "",
                "val_samples": "",
                "val_mse": f"{summary['val_mse_std']:.10f}",
                "val_rmse": f"{summary['val_rmse_std']:.10f}",
                "val_psnr": f"{summary['val_psnr_std']:.10f}",
            }
        )

    return output_csv_path


def run_kfold_evaluation_only(
    model,
    clean_images: np.ndarray,
    k_folds: int,
    batch_size: int,
    seed: int,
    noise_type: str = "gaussian",
    noise_std: float = 0.1,
    salt_pepper_amount: float = 0.01,
    output_csv_path: str | Path | None = None,
) -> dict[str, object]:
    if clean_images.ndim != 4:
        raise ValueError("clean_images must have shape (N, H, W, C).")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    val_folds = _numpy_kfold_validation_indices(
        sample_count=clean_images.shape[0],
        k_folds=k_folds,
        seed=seed,
    )

    fold_results: list[dict[str, float]] = []

    for fold_number, val_idx in enumerate(val_folds, start=1):
        clean_val = clean_images[val_idx]

        if noise_type == "gaussian":
            noisy_val = add_gaussian_noise(
                clean_val,
                std=noise_std,
                seed=seed + fold_number,
            )
        elif noise_type == "salt_pepper":
            noisy_val = add_salt_pepper_noise(
                clean_val,
                amount=salt_pepper_amount,
                seed=seed + fold_number,
            )
        else:
            raise ValueError(f"Unsupported noise_type for evaluation: {noise_type}")

        fold_metrics = compute_metrics(
            model=model,
            noisy_images=noisy_val,
            clean_images=clean_val,
            batch_size=batch_size,
        )

        fold_results.append(
            {
                "fold": int(fold_number),
                "val_samples": int(clean_val.shape[0]),
                "val_mse": float(fold_metrics["mse"]),
                "val_rmse": float(fold_metrics["rmse"]),
                "val_psnr": float(fold_metrics["psnr"]),
            }
        )

    summary = _summarize_eval_folds(fold_results)
    _print_kfold_eval_report(fold_results, summary)

    written_output_path = ""
    if output_csv_path is not None:
        written_output_path = str(
            _write_kfold_eval_csv(
                output_csv_path=output_csv_path,
                fold_results=fold_results,
                summary=summary,
            )
        )
        print(f"[INFO] Saved K-fold evaluation table to: {written_output_path}")

    return {
        "fold_results": fold_results,
        "summary": summary,
        "output_csv_path": written_output_path,
    }


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
    save_loss_curves: bool = False,
    loss_curve_output_path: str | Path | None = None,
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

        if save_loss_curves and loss_curve_output_path is not None:
            loss_curve_path = _fold_output_path(
                loss_curve_output_path,
                fold_number=fold_number,
                default_suffix=".png",
            )
            save_loss_curve(
                history=[entry for entry in fold_result.get("history", []) if isinstance(entry, dict)],
                output_path=loss_curve_path,
            )
            fold_result["loss_curve_path"] = str(loss_curve_path)

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
