import numpy as np

from src.evaluation.metrics import mse, psnr, rmse, ssim


def test_mse_and_rmse_values() -> None:
    y_true = np.array([0.0, 1.0], dtype=np.float32)
    y_pred = np.array([0.0, 0.0], dtype=np.float32)

    assert mse(y_true, y_pred) == 0.5
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.5))


def test_psnr_infinite_for_identical_arrays() -> None:
    y_true = np.zeros((4,), dtype=np.float32)
    y_pred = np.zeros((4,), dtype=np.float32)

    assert psnr(y_true, y_pred) == float("inf")


def test_ssim_is_one_for_identical_arrays() -> None:
    x = np.random.default_rng(10).random((2, 8, 8, 3), dtype=np.float32)
    assert np.isclose(ssim(x, x), 1.0, atol=1e-6)


def test_ssim_decreases_for_corrupted_images() -> None:
    rng = np.random.default_rng(11)
    clean = rng.random((2, 8, 8, 3), dtype=np.float32)
    noisy = np.clip(clean + rng.normal(0.0, 0.2, size=clean.shape).astype(np.float32), 0.0, 1.0)

    clean_ssim = ssim(clean, clean)
    noisy_ssim = ssim(clean, noisy)

    assert noisy_ssim < clean_ssim
