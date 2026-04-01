from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    images: np.ndarray,
    std: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = images + rng.normal(loc=0.0, scale=std, size=images.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_salt_pepper_noise(
    images: np.ndarray,
    amount: float = 0.01,
    salt_vs_pepper: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be in [0, 1].")
    if not 0.0 <= salt_vs_pepper <= 1.0:
        raise ValueError("salt_vs_pepper must be in [0, 1].")

    rng = np.random.default_rng(seed)
    noisy = images.copy()
    mask = rng.random(images.shape)

    salt_threshold = amount * salt_vs_pepper
    pepper_threshold = amount

    noisy[mask < salt_threshold] = 1.0
    noisy[(mask >= salt_threshold) & (mask < pepper_threshold)] = 0.0

    return noisy.astype(np.float32)
