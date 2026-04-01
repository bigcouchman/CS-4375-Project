from __future__ import annotations

from pathlib import Path

import numpy as np

from .layers import Conv2D, ReLU, Sigmoid


class CDAE:
    """Convolutional Denoising Autoencoder using custom NumPy layers."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 32,
        seed: int = 42,
    ) -> None:
        self.input_channels = input_channels
        self.latent_channels = latent_channels

        self.encoder_conv = Conv2D(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            padding=1,
            seed=seed,
        )
        self.encoder_act = ReLU()

        self.bottleneck_conv = Conv2D(
            in_channels=16,
            out_channels=latent_channels,
            kernel_size=3,
            padding=1,
            seed=seed + 1,
        )
        self.bottleneck_act = ReLU()

        self.decoder_conv = Conv2D(
            in_channels=latent_channels,
            out_channels=16,
            kernel_size=3,
            padding=1,
            seed=seed + 2,
        )
        self.decoder_act = ReLU()

        self.output_conv = Conv2D(
            in_channels=16,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
            seed=seed + 3,
        )
        self.output_act = Sigmoid()

    def forward(self, noisy_images: np.ndarray) -> np.ndarray:
        x = self.encoder_conv.forward(noisy_images)
        x = self.encoder_act.forward(x)

        x = self.bottleneck_conv.forward(x)
        x = self.bottleneck_act.forward(x)

        x = self.decoder_conv.forward(x)
        x = self.decoder_act.forward(x)

        x = self.output_conv.forward(x)
        x = self.output_act.forward(x)
        return x

    @staticmethod
    def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean((predictions - targets) ** 2))

    @staticmethod
    def mse_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (2.0 / predictions.size) * (predictions - targets)

    def train_step(self, noisy_batch: np.ndarray, clean_batch: np.ndarray) -> tuple[float, np.ndarray]:
        reconstructed = self.forward(noisy_batch)
        loss = self.mse_loss(reconstructed, clean_batch)
        return loss, reconstructed

    def backward_and_update(
        self,
        loss_grad: np.ndarray,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        grad = self.output_act.backward(loss_grad)
        grad = self.output_conv.backward(grad)

        grad = self.decoder_act.backward(grad)
        grad = self.decoder_conv.backward(grad)

        grad = self.bottleneck_act.backward(grad)
        grad = self.bottleneck_conv.backward(grad)

        grad = self.encoder_act.backward(grad)
        _ = self.encoder_conv.backward(grad)

        self.encoder_conv.apply_gradients(learning_rate, weight_decay)
        self.bottleneck_conv.apply_gradients(learning_rate, weight_decay)
        self.decoder_conv.apply_gradients(learning_rate, weight_decay)
        self.output_conv.apply_gradients(learning_rate, weight_decay)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "meta_input_channels": np.asarray([self.input_channels], dtype=np.int64),
            "meta_latent_channels": np.asarray([self.latent_channels], dtype=np.int64),
            "encoder_conv.weights": self.encoder_conv.weights.copy(),
            "encoder_conv.bias": self.encoder_conv.bias.copy(),
            "bottleneck_conv.weights": self.bottleneck_conv.weights.copy(),
            "bottleneck_conv.bias": self.bottleneck_conv.bias.copy(),
            "decoder_conv.weights": self.decoder_conv.weights.copy(),
            "decoder_conv.bias": self.decoder_conv.bias.copy(),
            "output_conv.weights": self.output_conv.weights.copy(),
            "output_conv.bias": self.output_conv.bias.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        expected_input_channels = int(np.asarray(state["meta_input_channels"]).ravel()[0])
        expected_latent_channels = int(np.asarray(state["meta_latent_channels"]).ravel()[0])

        if expected_input_channels != self.input_channels:
            raise ValueError(
                "Checkpoint input_channels does not match current model. "
                f"Checkpoint={expected_input_channels}, current={self.input_channels}."
            )

        if expected_latent_channels != self.latent_channels:
            raise ValueError(
                "Checkpoint latent_channels does not match current model. "
                f"Checkpoint={expected_latent_channels}, current={self.latent_channels}."
            )

        self.encoder_conv.weights = np.asarray(state["encoder_conv.weights"], dtype=np.float32).copy()
        self.encoder_conv.bias = np.asarray(state["encoder_conv.bias"], dtype=np.float32).copy()

        self.bottleneck_conv.weights = np.asarray(
            state["bottleneck_conv.weights"], dtype=np.float32
        ).copy()
        self.bottleneck_conv.bias = np.asarray(state["bottleneck_conv.bias"], dtype=np.float32).copy()

        self.decoder_conv.weights = np.asarray(state["decoder_conv.weights"], dtype=np.float32).copy()
        self.decoder_conv.bias = np.asarray(state["decoder_conv.bias"], dtype=np.float32).copy()

        self.output_conv.weights = np.asarray(state["output_conv.weights"], dtype=np.float32).copy()
        self.output_conv.bias = np.asarray(state["output_conv.bias"], dtype=np.float32).copy()

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
