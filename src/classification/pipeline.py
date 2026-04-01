from __future__ import annotations

import numpy as np

from .softmax_classifier import SoftmaxClassifier


def _flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(np.float32)


def _denoise_and_flatten_in_batches(
    denoiser_model,
    noisy_images: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    if noisy_images.ndim != 4:
        raise ValueError("noisy_images must have shape (N, H, W, C).")

    sample_count, height, width, channels = noisy_images.shape
    feature_dim = int(height * width * channels)
    features = np.empty((sample_count, feature_dim), dtype=np.float32)

    for start in range(0, sample_count, batch_size):
        end = min(start + batch_size, sample_count)
        denoised_batch = denoiser_model.forward(noisy_images[start:end])
        features[start:end] = _flatten_images(denoised_batch)

    return features


def run_softmax_classification_on_denoised(
    denoiser_model,
    noisy_train: np.ndarray,
    y_train: np.ndarray,
    noisy_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    noisy_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    denoise_batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, object]:
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    x_train = _denoise_and_flatten_in_batches(
        denoiser_model,
        noisy_train,
        batch_size=denoise_batch_size,
    )
    x_val = _denoise_and_flatten_in_batches(
        denoiser_model,
        noisy_val,
        batch_size=denoise_batch_size,
    )

    class_count = int(max(np.max(y_train), np.max(y_val)) + 1)
    if y_test is not None:
        y_test = np.asarray(y_test, dtype=np.int64)
        class_count = int(max(class_count - 1, int(np.max(y_test))) + 1)
    classifier = SoftmaxClassifier(
        input_dim=x_train.shape[1],
        num_classes=class_count,
        seed=seed,
    )

    history = classifier.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        verbose=verbose,
    )

    result: dict[str, object] = {
        "classifier_epochs_completed": len(history),
        "classifier_final_loss": history[-1]["loss"] if history else float("nan"),
        "classification_train_accuracy": classifier.score(x_train, y_train),
        "classification_val_accuracy": classifier.score(x_val, y_val),
    }

    if noisy_test is not None and y_test is not None:
        x_test = _denoise_and_flatten_in_batches(
            denoiser_model,
            noisy_test,
            batch_size=denoise_batch_size,
        )
        result["classification_test_accuracy"] = classifier.score(x_test, y_test)
        result["test_predictions"] = classifier.predict(x_test)

    return result
