import numpy as np

from src.models.layers import Conv2D


def test_conv2d_weight_gradient_close_to_numerical() -> None:
    rng = np.random.default_rng(5)

    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1, seed=5)
    x = rng.normal(size=(1, 4, 4, 1)).astype(np.float32)
    upstream_grad = rng.normal(size=(1, 4, 4, 1)).astype(np.float32)

    _ = layer.forward(x)
    _ = layer.backward(upstream_grad)

    analytic_grad = layer.grad_weights.copy()

    epsilon = 1e-3
    numerical_grad = np.zeros_like(layer.weights)

    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            for c in range(layer.weights.shape[2]):
                for o in range(layer.weights.shape[3]):
                    original_value = layer.weights[i, j, c, o]

                    layer.weights[i, j, c, o] = original_value + epsilon
                    loss_plus = float(np.sum(layer.forward(x) * upstream_grad))

                    layer.weights[i, j, c, o] = original_value - epsilon
                    loss_minus = float(np.sum(layer.forward(x) * upstream_grad))

                    numerical_grad[i, j, c, o] = (loss_plus - loss_minus) / (2.0 * epsilon)
                    layer.weights[i, j, c, o] = original_value

    max_abs_error = float(np.max(np.abs(analytic_grad - numerical_grad)))
    assert max_abs_error < 2e-2
