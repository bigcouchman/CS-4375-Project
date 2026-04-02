import numpy as np

from src.models.layers import Conv2D, NearestUpsample2D


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


def test_conv2d_stride_reduces_spatial_size() -> None:
    layer = Conv2D(in_channels=3, out_channels=5, kernel_size=3, padding=1, stride=2, seed=3)
    x = np.random.default_rng(4).normal(size=(2, 8, 8, 3)).astype(np.float32)

    y = layer.forward(x)

    assert y.shape == (2, 4, 4, 5)


def test_nearest_upsample_forward_backward_shapes() -> None:
    upsample = NearestUpsample2D(scale=2)
    x = np.random.default_rng(9).normal(size=(2, 4, 4, 3)).astype(np.float32)

    y = upsample.forward(x)
    grad_output = np.ones_like(y, dtype=np.float32)
    grad_input = upsample.backward(grad_output)

    assert y.shape == (2, 8, 8, 3)
    assert grad_input.shape == x.shape
    assert np.allclose(grad_input, 4.0)
