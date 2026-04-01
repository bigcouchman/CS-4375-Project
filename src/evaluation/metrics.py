from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_pixel: float = 1.0) -> float:
    current_mse = mse(y_true, y_pred)
    if current_mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_pixel) - 10.0 * np.log10(current_mse))
