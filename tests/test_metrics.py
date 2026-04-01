import numpy as np

from src.evaluation.metrics import mse, psnr, rmse


def test_mse_and_rmse_values() -> None:
    y_true = np.array([0.0, 1.0], dtype=np.float32)
    y_pred = np.array([0.0, 0.0], dtype=np.float32)

    assert mse(y_true, y_pred) == 0.5
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.5))


def test_psnr_infinite_for_identical_arrays() -> None:
    y_true = np.zeros((4,), dtype=np.float32)
    y_pred = np.zeros((4,), dtype=np.float32)

    assert psnr(y_true, y_pred) == float("inf")
