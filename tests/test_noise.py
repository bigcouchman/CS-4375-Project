import numpy as np

from src.data.noise import add_gaussian_noise, add_salt_pepper_noise, sample_gaussian_noise_std


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


def test_sample_gaussian_noise_std_returns_configured_values() -> None:
    rng = np.random.default_rng(33)
    choices = np.array([0.05, 0.1], dtype=np.float32)

    sampled = {sample_gaussian_noise_std([0.05, 0.1], rng) for _ in range(20)}

    assert all(np.any(np.isclose(value, choices, atol=1e-7)) for value in sampled)
    assert len(sampled) > 0
