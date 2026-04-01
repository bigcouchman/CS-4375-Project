import numpy as np

from src.data.noise import add_gaussian_noise, add_salt_pepper_noise


def test_gaussian_noise_keeps_shape_and_range() -> None:
    images = np.zeros((2, 32, 32, 3), dtype=np.float32)
    noisy = add_gaussian_noise(images, std=0.2, seed=1)

    assert noisy.shape == images.shape
    assert noisy.dtype == np.float32
    assert np.min(noisy) >= 0.0
    assert np.max(noisy) <= 1.0


def test_salt_pepper_noise_keeps_shape() -> None:
    images = np.full((3, 32, 32, 3), 0.5, dtype=np.float32)
    noisy = add_salt_pepper_noise(images, amount=0.05, seed=2)

    assert noisy.shape == images.shape
    assert noisy.dtype == np.float32
