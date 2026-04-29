from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from src.data import add_gaussian_noise, add_salt_pepper_noise, sample_gaussian_noise_std
from src.evaluation.visualize import save_loss_curve
from src.evaluation.metrics import ssim


def iterate_minibatches(
    noisy_images: np.ndarray,
    clean_images: np.ndarray,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if noisy_images.shape != clean_images.shape:
        raise ValueError("Noisy and clean image arrays has to have identical shapes.")

    rng = np.random.default_rng(seed)
    indices = np.arange(noisy_images.shape[0])
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield noisy_images[batch_idx], clean_images[batch_idx]


def compute_metrics(
    model: Any,
    noisy_images: np.ndarray,
    clean_images: np.ndarray,
    batch_size: int,
    include_ssim: bool = False,
) -> dict[str, float]:
    if noisy_images.shape != clean_images.shape:
        raise ValueError("Noisy and clean image arrays has to have identical shapes.")
    if batch_size <= 0:
        raise ValueError("batch_size has to be greater than 0.")

    total_squared_error = 0.0
    total_elements = 0
    total_ssim = 0.0
    total_samples = 0

    for start in range(0, noisy_images.shape[0], batch_size):
        end = min(start + batch_size, noisy_images.shape[0])
        noisy_batch = noisy_images[start:end]
        clean_batch = clean_images[start:end]

        reconstructed_batch = model.forward(noisy_batch)
        diff = clean_batch - reconstructed_batch

        total_squared_error += float(np.sum(diff * diff, dtype=np.float64))
        total_elements += int(diff.size)
        total_samples += int(clean_batch.shape[0])

        if include_ssim:
            batch_ssim = ssim(clean_batch, reconstructed_batch)
            total_ssim += float(batch_ssim) * int(clean_batch.shape[0])

    mse_value = float(total_squared_error / max(total_elements, 1))
    rmse_value = float(np.sqrt(mse_value))
    psnr_value = (
        float("inf")
        if mse_value == 0.0
        else float(20.0 * np.log10(1.0) - 10.0 * np.log10(mse_value))
    )

    metrics = {
        "mse": mse_value,
        "rmse": rmse_value,
        "psnr": psnr_value,
    }

    if include_ssim:
        metrics["ssim"] = float(total_ssim / max(total_samples, 1))

    return metrics


def sample_metric_subset(
    noisy_images: np.ndarray,
    clean_images: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if noisy_images.shape != clean_images.shape:
        raise ValueError("Noisy and clean image arrays has to have identical shapes.")

    if max_samples <= 0 or max_samples >= noisy_images.shape[0]:
        return noisy_images, clean_images

    rng = np.random.default_rng(seed)
    indices = rng.choice(noisy_images.shape[0], size=max_samples, replace=False)
    return noisy_images[indices], clean_images[indices]


def sample_epoch_subset(
    noisy_images: np.ndarray,
    clean_images: np.ndarray,
    min_samples: int,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if noisy_images.shape != clean_images.shape:
        raise ValueError("Noisy and clean image arrays has to have identical shapes.")

    if min_samples < 0 or max_samples < 0:
        raise ValueError("min_samples and max_samples has to be greather than or equal to 0.")

    if max_samples == 0:
        return noisy_images, clean_images

    available_samples = noisy_images.shape[0]
    upper_bound = min(max_samples, available_samples)
    lower_bound = min_samples if min_samples > 0 else upper_bound
    lower_bound = min(lower_bound, upper_bound)

    if upper_bound >= available_samples and lower_bound >= available_samples:
        return noisy_images, clean_images

    rng = np.random.default_rng(seed)
    if lower_bound == upper_bound:
        sample_count = upper_bound
    else:
        sample_count = int(rng.integers(lower_bound, upper_bound + 1))

    if sample_count >= available_samples:
        return noisy_images, clean_images

    indices = rng.choice(available_samples, size=sample_count, replace=False)
    return noisy_images[indices], clean_images[indices]


def calibrate_skip_weight(
    model: Any,
    noisy_images: np.ndarray,
    clean_images: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """Modify residual skip weight on validation data to minimize MSE. """

    if not hasattr(model, "skip_connection_weight"):
        return {"applied": 0.0}

    if not hasattr(model, "output_act"):
        # Restrict calibration to conv model.
        return {"applied": 0.0}

    current_w = float(getattr(model, "skip_connection_weight"))
    if current_w <= 0.0 or current_w >= 0.95:
        return {"applied": 0.0, "current_skip_weight": current_w}

    blend_scale = 1.0 - current_w
    if blend_scale <= 1e-8:
        return {"applied": 0.0, "current_skip_weight": current_w}

    numerator = 0.0
    denominator = 0.0
    target_sq_sum = 0.0
    total_elements = 0

    for start in range(0, noisy_images.shape[0], batch_size):
        end = min(start + batch_size, noisy_images.shape[0])
        noisy_batch = noisy_images[start:end]
        clean_batch = clean_images[start:end]

        reconstructed_batch = model.forward(noisy_batch)
        decoded_batch = (reconstructed_batch - (current_w * noisy_batch)) / blend_scale

        residual_batch = noisy_batch - decoded_batch
        target_batch = clean_batch - decoded_batch

        numerator += float(np.sum(target_batch * residual_batch, dtype=np.float64))
        denominator += float(np.sum(residual_batch * residual_batch, dtype=np.float64))
        target_sq_sum += float(np.sum(target_batch * target_batch, dtype=np.float64))
        total_elements += int(target_batch.size)

    if denominator <= 1e-12 or total_elements == 0:
        return {"applied": 0.0, "current_skip_weight": current_w}

    optimal_w = float(np.clip(numerator / denominator, 0.0, 0.95))

    current_sse = target_sq_sum - (2.0 * current_w * numerator) + ((current_w**2) * denominator)
    optimal_sse = target_sq_sum - (2.0 * optimal_w * numerator) + ((optimal_w**2) * denominator)

    current_mse = float(current_sse / total_elements)
    optimal_mse = float(optimal_sse / total_elements)
    improved = optimal_mse + 1e-12 < current_mse

    if improved:
        model.skip_connection_weight = optimal_w

    return {
        "applied": 1.0 if improved else 0.0,
        "current_skip_weight": current_w,
        "optimal_skip_weight": optimal_w,
        "current_val_mse": current_mse,
        "optimal_val_mse": optimal_mse,
    }


def train_fold(
    model,
    noisy_train: np.ndarray,
    clean_train: np.ndarray,
    noisy_val: np.ndarray,
    clean_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = 0.0,
    lr_decay: float = 1.0,
    lr_decay_every: int = 0,
    early_stopping_patience: int = 0,
    min_delta: float = 0.0,
    noisy_test: np.ndarray | None = None,
    clean_test: np.ndarray | None = None,
    noise_type: str = "gaussian",
    salt_pepper_amount: float = 0.01,
    batch_noise_std_options: tuple[float, ...] | None = None,
    sample_noise_per_batch: bool = False,
    track_ssim: bool = False,
    train_metrics_max_samples: int = 1024,
    epoch_train_subset_min: int = 0,
    epoch_train_subset_max: int = 0,
    loss_curve_output_path: str | Path | None = None,
    loss_curve_update_every: int = 1,
) -> dict[str, object]:
    if loss_curve_update_every <= 0:
        raise ValueError("loss_curve_update_every has to be greater than 0.")
    if train_metrics_max_samples < 0:
        raise ValueError("train_metrics_max_samples has to be greater than or equal to 0.")
    if epoch_train_subset_min < 0 or epoch_train_subset_max < 0:
        raise ValueError("epoch_train_subset_min and epoch_train_subset_max has to be greater than or equal to 0.")
    if epoch_train_subset_max > 0 and epoch_train_subset_min > epoch_train_subset_max:
        raise ValueError("epoch_train_subset_min cannot be greater than epoch_train_subset_max.")

    history: list[dict[str, float]] = []
    current_learning_rate = learning_rate
    noise_rng = np.random.default_rng(seed + 10_000)

    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, np.ndarray] | None = None

    if hasattr(model, "state_dict"):
        best_state = model.state_dict()

    for epoch in range(1, epochs + 1):
        batch_losses: list[float] = []
        sampled_noise_stds: list[float] = []

        epoch_noisy_train, epoch_clean_train = sample_epoch_subset(
            noisy_images=noisy_train,
            clean_images=clean_train,
            min_samples=epoch_train_subset_min,
            max_samples=epoch_train_subset_max,
            seed=seed + epoch + 25_000,
        )
        epoch_train_samples = int(epoch_noisy_train.shape[0])

        for noisy_batch, clean_batch in iterate_minibatches(
            epoch_noisy_train,
            epoch_clean_train,
            batch_size=batch_size,
            seed=seed + epoch,
        ):
            current_noisy_batch = noisy_batch
            if sample_noise_per_batch and noise_type == "gaussian" and batch_noise_std_options:
                sampled_std = sample_gaussian_noise_std(list(batch_noise_std_options), noise_rng)
                sampled_noise_stds.append(sampled_std)
                current_noisy_batch = add_gaussian_noise(
                    clean_batch,
                    std=sampled_std,
                    seed=int(noise_rng.integers(0, np.iinfo(np.int32).max)),
                )
            elif sample_noise_per_batch and noise_type == "salt_pepper":
                current_noisy_batch = add_salt_pepper_noise(
                    clean_batch,
                    amount=salt_pepper_amount,
                    seed=int(noise_rng.integers(0, np.iinfo(np.int32).max)),
                )

            if hasattr(model, "loss_and_grad"):
                reconstructed = model.forward(current_noisy_batch)
                batch_loss, loss_grad = model.loss_and_grad(reconstructed, clean_batch)
            else:
                batch_loss, reconstructed = model.train_step(current_noisy_batch, clean_batch)
                loss_grad = model.mse_grad(reconstructed, clean_batch)

            batch_losses.append(batch_loss)

            model.backward_and_update(
                loss_grad,
                current_learning_rate,
                weight_decay=weight_decay,
            )

        noisy_train_eval, clean_train_eval = sample_metric_subset(
            noisy_images=noisy_train,
            clean_images=clean_train,
            max_samples=train_metrics_max_samples,
            seed=seed + epoch + 50_000,
        )

        train_metrics = compute_metrics(
            model,
            noisy_train_eval,
            clean_train_eval,
            batch_size=batch_size,
            include_ssim=track_ssim,
        )
        val_metrics = compute_metrics(
            model,
            noisy_val,
            clean_val,
            batch_size=batch_size,
            include_ssim=track_ssim,
        )

        train_mse_value = train_metrics["mse"]
        val_mse_value = val_metrics["mse"]

        history_entry = {
            "epoch": float(epoch),
            "batch_loss_mean": float(np.mean(batch_losses)),
            "learning_rate": float(current_learning_rate),
            "epoch_train_samples": float(epoch_train_samples),
            "train_mse": float(train_metrics["mse"]),
            "val_mse": float(val_metrics["mse"]),
            "train_rmse": float(train_metrics["rmse"]),
            "val_rmse": float(val_metrics["rmse"]),
            "train_psnr": float(train_metrics["psnr"]),
            "val_psnr": float(val_metrics["psnr"]),
        }

        if track_ssim:
            history_entry["train_ssim"] = float(train_metrics.get("ssim", float("nan")))
            history_entry["val_ssim"] = float(val_metrics.get("ssim", float("nan")))

        if sampled_noise_stds:
            history_entry["batch_noise_std_mean"] = float(np.mean(sampled_noise_stds))

        history.append(history_entry)

        if loss_curve_output_path is not None and (epoch % loss_curve_update_every == 0):
            save_loss_curve(
                history=history,
                output_path=Path(loss_curve_output_path),
            )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"epoch_train_samples={epoch_train_samples} | "
            f"batch_loss_mean={np.mean(batch_losses):.6f} | "
            f"lr={current_learning_rate:.6f} | "
            f"train_mse={train_mse_value:.6f} | val_mse={val_mse_value:.6f}"
        )

        improved = (best_val_mse - val_mse_value) > min_delta
        if improved:
            best_val_mse = val_mse_value
            best_epoch = epoch
            epochs_without_improvement = 0
            if hasattr(model, "state_dict"):
                best_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if lr_decay_every > 0 and lr_decay < 1.0 and (epoch % lr_decay_every == 0):
            current_learning_rate *= lr_decay
            print(f"[INFO] Learning rate decayed to {current_learning_rate:.6f}")

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                "[INFO] Early stopping triggered: "
                f"no val_mse improvement > {min_delta} for {early_stopping_patience} epochs."
            )
            break

    if loss_curve_output_path is not None and history:
        save_loss_curve(
            history=history,
            output_path=Path(loss_curve_output_path),
        )

    if history and best_state is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best checkpoint from epoch {best_epoch} (val_mse={best_val_mse:.6f}).")

    skip_calibration = calibrate_skip_weight(
        model=model,
        noisy_images=noisy_val,
        clean_images=clean_val,
        batch_size=batch_size,
    )
    if int(skip_calibration.get("applied", 0.0)) == 1:
        print(
            "[INFO] Calibrated skip weight on validation set: "
            f"{skip_calibration.get('current_skip_weight', float('nan')):.4f} -> "
            f"{skip_calibration.get('optimal_skip_weight', float('nan')):.4f} "
            f"(val_mse {skip_calibration.get('current_val_mse', float('nan')):.6f} -> "
            f"{skip_calibration.get('optimal_val_mse', float('nan')):.6f})"
        )

    final_train_metrics = compute_metrics(
        model,
        noisy_train,
        clean_train,
        batch_size=batch_size,
        include_ssim=track_ssim,
    )
    final_val_metrics = compute_metrics(
        model,
        noisy_val,
        clean_val,
        batch_size=batch_size,
        include_ssim=track_ssim,
    )

    if not history:
        best_val_mse = final_val_metrics["mse"]

    result = {
        "history": history,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "final_learning_rate": current_learning_rate,
        "skip_weight_calibration_applied": int(skip_calibration.get("applied", 0.0)),
        "calibrated_skip_weight": float(
            skip_calibration.get(
                "optimal_skip_weight",
                getattr(model, "skip_connection_weight", float("nan")),
            )
        ),
        "train_mse": final_train_metrics["mse"],
        "val_mse": final_val_metrics["mse"],
        "train_rmse": final_train_metrics["rmse"],
        "val_rmse": final_val_metrics["rmse"],
        "train_psnr": final_train_metrics["psnr"],
        "val_psnr": final_val_metrics["psnr"],
    }

    if track_ssim:
        result["train_ssim"] = final_train_metrics.get("ssim", float("nan"))
        result["val_ssim"] = final_val_metrics.get("ssim", float("nan"))

    if noisy_test is not None and clean_test is not None:
        final_test_metrics = compute_metrics(
            model,
            noisy_test,
            clean_test,
            batch_size=batch_size,
            include_ssim=track_ssim,
        )
        result["test_mse"] = final_test_metrics["mse"]
        result["test_rmse"] = final_test_metrics["rmse"]
        result["test_psnr"] = final_test_metrics["psnr"]
        if track_ssim:
            result["test_ssim"] = final_test_metrics.get("ssim", float("nan"))

    return result
