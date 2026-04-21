from __future__ import annotations

from pathlib import Path

import numpy as np


class FullyConnectedDAE:
    """Fully connected denoising autoencoder implemented with NumPy."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (32, 32, 3),
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        seed: int = 42,
        skip_connection_weight: float = 0.25,
        l1_weight: float = 0.0,
    ) -> None:
        self.input_shape = tuple(int(value) for value in input_shape)
        self.input_dim = int(np.prod(self.input_shape))
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.skip_connection_weight = float(np.clip(skip_connection_weight, 0.0, 0.95))
        self.l1_weight = max(0.0, float(l1_weight))

        rng = np.random.default_rng(seed)

        self.w1 = self._xavier_init(rng, self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)

        self.w2 = self._xavier_init(rng, self.hidden_dim, self.bottleneck_dim)
        self.b2 = np.zeros((self.bottleneck_dim,), dtype=np.float32)

        self.w3 = self._xavier_init(rng, self.bottleneck_dim, self.hidden_dim)
        self.b3 = np.zeros((self.hidden_dim,), dtype=np.float32)

        self.w4 = self._xavier_init(rng, self.hidden_dim, self.input_dim)
        self.b4 = np.zeros((self.input_dim,), dtype=np.float32)

        self._cached_input_shape: tuple[int, int, int, int] | None = None
        self._cached_flat_input: np.ndarray | None = None
        self._cached_z1: np.ndarray | None = None
        self._cached_a1: np.ndarray | None = None
        self._cached_z2: np.ndarray | None = None
        self._cached_a2: np.ndarray | None = None
        self._cached_z3: np.ndarray | None = None
        self._cached_a3: np.ndarray | None = None
        self._cached_sigmoid: np.ndarray | None = None

    @staticmethod
    def _xavier_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _encode(
        self,
        noisy_images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if noisy_images.ndim != 4:
            raise ValueError("Expected noisy_images shape (N, H, W, C).")

        batch_size = int(noisy_images.shape[0])
        flattened = noisy_images.reshape(batch_size, self.input_dim).astype(np.float32)

        z1 = flattened @ self.w1 + self.b1
        a1 = self._relu(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = self._relu(z2)

        return flattened, z1, a1, z2, a2

    def encode_features(self, noisy_images: np.ndarray) -> np.ndarray:
        """Return bottleneck encoder activations for classifier features."""
        _, _, _, _, a2 = self._encode(noisy_images)
        return a2.astype(np.float32)

    def forward(self, noisy_images: np.ndarray) -> np.ndarray:
        flattened, z1, a1, z2, a2 = self._encode(noisy_images)

        z3 = a2 @ self.w3 + self.b3
        a3 = self._relu(z3)

        z4 = a3 @ self.w4 + self.b4
        decoded = self._sigmoid(z4).astype(np.float32)

        if self.skip_connection_weight > 0.0:
            reconstructed = (
                (1.0 - self.skip_connection_weight) * decoded
                + self.skip_connection_weight * flattened
            ).astype(np.float32)
            reconstructed = np.clip(reconstructed, 0.0, 1.0)
        else:
            reconstructed = decoded

        self._cached_input_shape = noisy_images.shape
        self._cached_flat_input = flattened
        self._cached_z1 = z1
        self._cached_a1 = a1
        self._cached_z2 = z2
        self._cached_a2 = a2
        self._cached_z3 = z3
        self._cached_a3 = a3
        self._cached_sigmoid = decoded

        return reconstructed.reshape(noisy_images.shape)

    @staticmethod
    def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean((predictions - targets) ** 2, dtype=np.float32))

    @staticmethod
    def mse_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (2.0 / predictions.size) * (predictions - targets)

    @staticmethod
    def mae_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(np.abs(predictions - targets), dtype=np.float32))

    @staticmethod
    def mae_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.sign(predictions - targets) / predictions.size

    def loss_and_grad(self, predictions: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        mse_value = self.mse_loss(predictions, targets)
        grad = self.mse_grad(predictions, targets)

        if self.l1_weight <= 0.0:
            return mse_value, grad

        mae_value = self.mae_loss(predictions, targets)
        grad = grad + (self.l1_weight * self.mae_grad(predictions, targets))
        return float(mse_value + (self.l1_weight * mae_value)), grad.astype(np.float32)

    def train_step(self, noisy_batch: np.ndarray, clean_batch: np.ndarray) -> tuple[float, np.ndarray]:
        reconstructed = self.forward(noisy_batch)
        loss, _ = self.loss_and_grad(reconstructed, clean_batch)
        return loss, reconstructed

    def backward_and_update(
        self,
        loss_grad: np.ndarray,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        if self._cached_input_shape is None or self._cached_flat_input is None:
            raise RuntimeError("backward_and_update called before forward.")
        if self._cached_z1 is None or self._cached_a1 is None:
            raise RuntimeError("Missing cached encoder activations.")
        if self._cached_z2 is None or self._cached_a2 is None:
            raise RuntimeError("Missing cached bottleneck activations.")
        if self._cached_z3 is None or self._cached_a3 is None or self._cached_sigmoid is None:
            raise RuntimeError("Missing cached decoder activations.")

        grad_flat = loss_grad.reshape(loss_grad.shape[0], self.input_dim).astype(np.float32)
        if self.skip_connection_weight > 0.0:
            grad_flat = grad_flat * (1.0 - self.skip_connection_weight)

        sigmoid_output = self._cached_sigmoid
        dz4 = grad_flat * sigmoid_output * (1.0 - sigmoid_output)

        dw4 = self._cached_a3.T @ dz4
        db4 = np.sum(dz4, axis=0)

        da3 = dz4 @ self.w4.T
        dz3 = da3 * self._relu_grad(self._cached_z3)

        dw3 = self._cached_a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.w3.T
        dz2 = da2 * self._relu_grad(self._cached_z2)

        dw2 = self._cached_a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * self._relu_grad(self._cached_z1)

        dw1 = self._cached_flat_input.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w4 -= learning_rate * (dw4 + weight_decay * self.w4)
        self.b4 -= learning_rate * db4

        self.w3 -= learning_rate * (dw3 + weight_decay * self.w3)
        self.b3 -= learning_rate * db3

        self.w2 -= learning_rate * (dw2 + weight_decay * self.w2)
        self.b2 -= learning_rate * db2

        self.w1 -= learning_rate * (dw1 + weight_decay * self.w1)
        self.b1 -= learning_rate * db1

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "meta_input_shape": np.asarray(self.input_shape, dtype=np.int64),
            "meta_hidden_dim": np.asarray([self.hidden_dim], dtype=np.int64),
            "meta_bottleneck_dim": np.asarray([self.bottleneck_dim], dtype=np.int64),
            "meta_skip_connection_weight": np.asarray([self.skip_connection_weight], dtype=np.float32),
            "meta_l1_weight": np.asarray([self.l1_weight], dtype=np.float32),
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy(),
            "w3": self.w3.copy(),
            "b3": self.b3.copy(),
            "w4": self.w4.copy(),
            "b4": self.b4.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        checkpoint_shape = tuple(int(x) for x in np.asarray(state["meta_input_shape"]).tolist())
        if checkpoint_shape != self.input_shape:
            raise ValueError(
                "Checkpoint input shape does not match current model. "
                f"Checkpoint={checkpoint_shape}, current={self.input_shape}."
            )

        self.w1 = np.asarray(state["w1"], dtype=np.float32).copy()
        self.b1 = np.asarray(state["b1"], dtype=np.float32).copy()
        self.w2 = np.asarray(state["w2"], dtype=np.float32).copy()
        self.b2 = np.asarray(state["b2"], dtype=np.float32).copy()
        self.w3 = np.asarray(state["w3"], dtype=np.float32).copy()
        self.b3 = np.asarray(state["b3"], dtype=np.float32).copy()
        self.w4 = np.asarray(state["w4"], dtype=np.float32).copy()
        self.b4 = np.asarray(state["b4"], dtype=np.float32).copy()

        if "meta_skip_connection_weight" in state:
            self.skip_connection_weight = float(
                np.asarray(state["meta_skip_connection_weight"]).ravel()[0]
            )

        if "meta_l1_weight" in state:
            self.l1_weight = float(np.asarray(state["meta_l1_weight"]).ravel()[0])

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_path, **self.state_dict())

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            state = {key: checkpoint[key] for key in checkpoint.files}

        self.load_state_dict(state)
