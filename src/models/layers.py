# Defining CDAE layers
from __future__ import annotations

import numpy as np
from typing import cast


class Conv2D:
    """NumPy Conv2D layer with forward and backward passes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        seed: int | None = None,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride has to be a positive integer.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        self.weights = rng.uniform(
            low=-limit,
            high=limit,
            size=(kernel_size, kernel_size, in_channels, out_channels),
        ).astype(np.float32)
        self.bias = np.zeros(shape=(out_channels,), dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._cached_input: np.ndarray | None = None

    # Forward pass logic
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError("Conv2D needs input with shape (N, H, W, C).")

        n, h, w, _ = x.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        out_h = ((h + (2 * p) - k) // s) + 1
        out_w = ((w + (2 * p) - k) // s) + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError("Invalid output shape.")

        x_padded = np.pad(
            x,
            pad_width=((0, 0), (p, p), (p, p), (0, 0)),
            mode="constant",
        )

        output = np.zeros((n, out_h, out_w, self.out_channels), dtype=np.float32)

        for row in range(out_h):
            for col in range(out_w):
                row_start = row * s
                col_start = col * s
                patch = x_padded[:, row_start : row_start + k, col_start : col_start + k, :]
                output[:, row, col, :] = (
                    np.tensordot(patch, self.weights, axes=([1, 2, 3], [0, 1, 2]))
                    + self.bias
                )

        self._cached_input = x
        return output

    # Backward pass to apply gradient descent
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cached_input is None:
            raise RuntimeError("Conv2D.backward called before forward.")
        if grad_output.ndim != 4:
            raise ValueError("Conv2D backward needs grad_output with shape (N, H, W, C).")

        x = self._cached_input
        n, h, w, _ = x.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        out_h = ((h + (2 * p) - k) // s) + 1
        out_w = ((w + (2 * p) - k) // s) + 1

        expected_shape = (n, out_h, out_w, self.out_channels)
        if grad_output.shape != expected_shape:
            raise ValueError(
                "grad_output shape does not match Conv2D output shape. "
                f"Expected {expected_shape}, got {grad_output.shape}."
            )

        x_padded = np.pad(
            x,
            pad_width=((0, 0), (p, p), (p, p), (0, 0)),
            mode="constant",
        )
        grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)

        self.grad_weights.fill(0.0)
        self.grad_bias = np.sum(grad_output, axis=(0, 1, 2), dtype=np.float32)

        for row in range(out_h):
            for col in range(out_w):
                row_start = row * s
                col_start = col * s
                patch = x_padded[:, row_start : row_start + k, col_start : col_start + k, :]
                grad_slice = grad_output[:, row, col, :]

                self.grad_weights += np.tensordot(
                    patch,
                    grad_slice,
                    axes=([0], [0]),
                )

                grad_x_padded[:, row_start : row_start + k, col_start : col_start + k, :] += np.tensordot(
                    grad_slice,
                    self.weights,
                    axes=([1], [3]),
                )

        if p > 0:
            grad_input = grad_x_padded[:, p:-p, p:-p, :]
        else:
            grad_input = grad_x_padded

        return grad_input.astype(np.float32)

    def apply_gradients(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        grad_weights = self.grad_weights + (weight_decay * self.weights)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * self.grad_bias

# RELU activation 
class ReLU:
    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward.")
        return grad_output * self._mask

# Sigmoid activation
class Sigmoid:
    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -30.0, 30.0)
        output = 1.0 / (1.0 + np.exp(-clipped))
        self._output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("Sigmoid.backward called before forward.")
        return grad_output * self._output * (1.0 - self._output)


class NearestUpsample2D:
    """Nearest neighbor upsampling layer with backward pass."""

    def __init__(self, scale: int = 2) -> None:
        if scale <= 0:
            raise ValueError("scale has to be a positive integer.")
        self.scale = scale
        self._cached_input_shape: tuple[int, int, int, int] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError("NearestUpsample2D needs input with shape (N, H, W, C).")

        self._cached_input_shape = cast(tuple[int, int, int, int], x.shape)
        if self.scale == 1:
            return x.copy()

        upsampled = np.repeat(x, self.scale, axis=1)
        upsampled = np.repeat(upsampled, self.scale, axis=2)
        return upsampled.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cached_input_shape is None:
            raise RuntimeError("NearestUpsample2D.backward called before forward.")
        if grad_output.ndim != 4:
            raise ValueError(
                "NearestUpsample2D backward needs grad_output with shape (N, H, W, C)."
            )

        n, h, w, c = self._cached_input_shape
        expected_shape = (n, h * self.scale, w * self.scale, c)
        if grad_output.shape != expected_shape:
            raise ValueError(
                "grad_output shape does not match NearestUpsample2D output shape. "
                f"Expected {expected_shape}, got {grad_output.shape}."
            )

        if self.scale == 1:
            return grad_output.astype(np.float32)

        grad_input = grad_output.reshape(n, h, self.scale, w, self.scale, c).sum(axis=(2, 4))
        return grad_input.astype(np.float32)
