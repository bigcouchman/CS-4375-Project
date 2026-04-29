from __future__ import annotations

import numpy as np

from .softmax_classifier import SoftmaxClassifier


def _flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(np.float32)


def _apply_standardization(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    scaled_features = (features - mean) / std
    return np.clip(scaled_features, -6.0, 6.0).astype(np.float32)


def _encode_and_flatten_in_batches(
    denoiser_model,
    images: np.ndarray,
    batch_size: int,
    denoise_first: bool = False,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size has to be greater than 0.")

    if images.ndim != 4:
        raise ValueError("images has to be in the shape (N, H, W, C).")

    image_count = int(images.shape[0])
    if image_count == 0:
        raise ValueError("images need at least one sample.")

    def _encode_batch(batch: np.ndarray) -> np.ndarray:
        batch_data = np.asarray(batch, dtype=np.float32)
        if denoise_first:
            batch_data = np.asarray(denoiser_model.forward(batch_data), dtype=np.float32)

        if hasattr(denoiser_model, "encode_features"):
            return np.asarray(denoiser_model.encode_features(batch_data), dtype=np.float32)

        if denoise_first:
            return batch_data

        return np.asarray(denoiser_model.forward(batch_data), dtype=np.float32)

    first_batch_end = min(batch_size, image_count)
    first_batch = _encode_batch(images[:first_batch_end])
    first_flattened = _flatten_images(first_batch)

    encoded_features = np.empty((image_count, first_flattened.shape[1]), dtype=np.float32)
    encoded_features[:first_batch_end] = first_flattened

    for start_index in range(first_batch_end, image_count, batch_size):
        end_index = min(start_index + batch_size, image_count)
        batch_features = _encode_batch(images[start_index:end_index])
        encoded_features[start_index:end_index] = _flatten_images(batch_features)

    return encoded_features


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
    clean_train: np.ndarray | None = None,
    noisy_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    denoise_batch_size: int = 64,
    classifier_hidden_dims: tuple[int, ...] = (32,),
    classifier_dropout: float = 0.6,
    classifier_feature_noise_std: float = 0.01,
    classifier_early_stopping_patience: int = 6,
    classifier_early_stopping_min_delta: float = 0.001,
    verbose: bool = False,
) -> dict[str, object]:
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    # Use the denoiser's encoded features.
    x_train_noisy = _encode_and_flatten_in_batches(
        denoiser_model,
        noisy_train,
        batch_size=denoise_batch_size,
        denoise_first=False,
    )
    x_val = _encode_and_flatten_in_batches(
        denoiser_model,
        noisy_val,
        batch_size=denoise_batch_size,
        denoise_first=False,
    )

    x_train_fit = x_train_noisy
    y_train_fit = y_train
    if clean_train is not None:
        x_train_clean = _encode_and_flatten_in_batches(
            denoiser_model,
            np.asarray(clean_train, dtype=np.float32),
            batch_size=denoise_batch_size,
            denoise_first=False,
        )
        x_train_fit = np.concatenate((x_train_noisy, x_train_clean), axis=0)
        y_train_fit = np.concatenate((y_train, y_train), axis=0)

    train_mean = np.mean(x_train_fit, axis=0, keepdims=True)
    train_std = np.std(x_train_fit, axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)

    x_train_fit = _apply_standardization(x_train_fit, train_mean, train_std)
    x_train_noisy = _apply_standardization(x_train_noisy, train_mean, train_std)
    x_val = _apply_standardization(x_val, train_mean, train_std)

    if classifier_feature_noise_std < 0.0:
        raise ValueError("classifier_feature_noise_std must be >= 0.0.")

    class_count = int(max(np.max(y_train), np.max(y_val)) + 1)
    if y_test is not None:
        y_test = np.asarray(y_test, dtype=np.int64)
        class_count = int(max(class_count - 1, int(np.max(y_test))) + 1)

    classifier = SoftmaxClassifier(
        input_dim=x_train_fit.shape[1],
        num_classes=class_count,
        hidden_dims=classifier_hidden_dims,
        dropout_rate=classifier_dropout,
        input_noise_std=classifier_feature_noise_std,
        seed=seed,
    )

    history = classifier.fit(
        x_train_fit,
        y_train_fit,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        val_features=x_val,
        val_labels=y_val,
        early_stopping_patience=classifier_early_stopping_patience,
        early_stopping_min_delta=classifier_early_stopping_min_delta,
        verbose=verbose,
    )

    result: dict[str, object] = {
        "classifier_epochs_completed": len(history),
        "classifier_final_loss": history[-1]["loss"] if history else float("nan"),
        "classification_train_accuracy": classifier.score(x_train_noisy, y_train),
        "classification_val_accuracy": classifier.score(x_val, y_val),
    }

    if noisy_test is not None and y_test is not None:
        x_test = _encode_and_flatten_in_batches(
            denoiser_model,
            noisy_test,
            batch_size=denoise_batch_size,
            denoise_first=False,
        )
        x_test = _apply_standardization(x_test, train_mean, train_std)
        result["classification_test_accuracy"] = classifier.score(x_test, y_test)
        result["test_predictions"] = classifier.predict(x_test)

    return result
