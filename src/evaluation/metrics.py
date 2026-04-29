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


def _ssim_1d(x: np.ndarray, y: np.ndarray, c1: float, c2: float) -> float:
    mu_x = float(np.mean(x, dtype=np.float64))
    mu_y = float(np.mean(y, dtype=np.float64))

    x_centered = x - mu_x
    y_centered = y - mu_y

    sigma_x = float(np.mean(x_centered * x_centered, dtype=np.float64))
    sigma_y = float(np.mean(y_centered * y_centered, dtype=np.float64))
    sigma_xy = float(np.mean(x_centered * y_centered, dtype=np.float64))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)

    if denominator <= 1e-12:
        return 1.0
    return float(numerator / denominator)


def ssim(y_true: np.ndarray, y_pred: np.ndarray, max_pixel: float = 1.0) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred has to have the same shapes for SSIM.")

    c1 = (0.01 * max_pixel) ** 2
    c2 = (0.03 * max_pixel) ** 2

    x = np.asarray(y_true, dtype=np.float64)
    y = np.asarray(y_pred, dtype=np.float64)

    if x.ndim == 4:
        n, _, _, c = x.shape
        scores: list[float] = []
        for sample_idx in range(n):
            for channel_idx in range(c):
                scores.append(
                    _ssim_1d(
                        x[sample_idx, :, :, channel_idx].reshape(-1),
                        y[sample_idx, :, :, channel_idx].reshape(-1),
                        c1,
                        c2,
                    )
                )
        return float(np.mean(scores, dtype=np.float64))

    if x.ndim == 3:
        _, _, c = x.shape
        scores = [
            _ssim_1d(x[:, :, channel_idx].reshape(-1), y[:, :, channel_idx].reshape(-1), c1, c2)
            for channel_idx in range(c)
        ]
        return float(np.mean(scores, dtype=np.float64))

    return _ssim_1d(x.reshape(-1), y.reshape(-1), c1, c2)
