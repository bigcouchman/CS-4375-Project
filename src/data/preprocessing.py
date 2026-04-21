from __future__ import annotations

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def select_random_subset(
    images: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same number of samples.")

    sample_count = min(max_samples, images.shape[0])
    if sample_count == images.shape[0]:
        return images, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=sample_count, replace=False)
    return images[indices], labels[indices]


def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError("images must have shape (N, H, W, C).")
    if target_size <= 0:
        raise ValueError("target_size must be > 0.")

    _, height, width, _ = images.shape
    if height == target_size and width == target_size:
        return images.astype(np.float32)

    if height == 32 and width == 32 and target_size == 16:
        # 2x2 average pooling downsample keeps denoising signal while reducing compute.
        downsampled = images.reshape(images.shape[0], 16, 2, 16, 2, images.shape[3]).mean(
            axis=(2, 4),
            dtype=np.float32,
        )
        return downsampled.astype(np.float32)

    raise ValueError(
        "Unsupported resize configuration. "
        f"Current shape={images.shape[1:3]}, target_size={target_size}."
    )
