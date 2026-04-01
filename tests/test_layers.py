import numpy as np

from src.models.layers import Conv2D


def test_conv2d_forward_shape() -> None:
    layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3, padding=1, seed=123)
    x = np.random.default_rng(1).normal(size=(2, 8, 8, 3)).astype(np.float32)

    y = layer.forward(x)

    assert y.shape == (2, 8, 8, 4)


def test_conv2d_backward_and_update() -> None:
    layer = Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding=1, seed=123)
    x = np.random.default_rng(2).normal(size=(2, 6, 6, 3)).astype(np.float32)

    y = layer.forward(x)
    grad_output = np.ones_like(y, dtype=np.float32)

    grad_input = layer.backward(grad_output)

    assert grad_input.shape == x.shape
    assert layer.grad_weights.shape == layer.weights.shape
    assert layer.grad_bias.shape == layer.bias.shape

    weights_before = layer.weights.copy()
    bias_before = layer.bias.copy()
    layer.apply_gradients(learning_rate=1e-3)

    assert not np.allclose(weights_before, layer.weights)
    assert not np.allclose(bias_before, layer.bias)
