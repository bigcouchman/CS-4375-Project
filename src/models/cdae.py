from __future__ import annotations

from pathlib import Path

import numpy as np

from .layers import Conv2D, NearestUpsample2D, ReLU, Sigmoid


class CDAE:
    """Convolutional Denoising Autoencoder using custom NumPy layers."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 96,
        seed: int = 42,
        l1_weight: float = 0.0,
        skip_connection_weight: float = 0.8,
    ) -> None:
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.l1_weight = max(0.0, float(l1_weight))
        self.skip_connection_weight = float(np.clip(skip_connection_weight, 0.0, 0.95))

        self.encoder_conv = Conv2D(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            seed=seed,
        )
        self.encoder_act = ReLU()

        self.bottleneck_conv = Conv2D(
            in_channels=32,
            out_channels=latent_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            seed=seed + 1,
        )
        self.bottleneck_act = ReLU()

        self.bottleneck_refine_conv = Conv2D(
            in_channels=latent_channels,
            out_channels=latent_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            seed=seed + 5,
        )
        self.bottleneck_refine_act = ReLU()

        self.decoder_upsample = NearestUpsample2D(scale=2)

        self.decoder_conv = Conv2D(
            in_channels=latent_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            seed=seed + 2,
        )
        self.decoder_act = ReLU()

        self.refine_conv = Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            seed=seed + 4,
        )
        self.refine_act = ReLU()

        self.refine_conv_2 = Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            seed=seed + 6,
        )
        self.refine_act_2 = ReLU()

        self.output_conv = Conv2D(
            in_channels=32,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            seed=seed + 3,
        )
        self.output_act = Sigmoid()

    def encode_features(self, images: np.ndarray) -> np.ndarray:
        """Return latent encoder representation before the decoder path."""
        x = self.encoder_conv.forward(images)
        x = self.encoder_act.forward(x)

        x = self.bottleneck_conv.forward(x)
        x = self.bottleneck_act.forward(x)

        x = self.bottleneck_refine_conv.forward(x)
        x = self.bottleneck_refine_act.forward(x)
        return x.astype(np.float32)

    def forward(self, noisy_images: np.ndarray) -> np.ndarray:
        x = self.encode_features(noisy_images)

        x = self.decoder_upsample.forward(x)

        x = self.decoder_conv.forward(x)
        x = self.decoder_act.forward(x)

        x = self.refine_conv.forward(x)
        x = self.refine_act.forward(x)

        x = self.refine_conv_2.forward(x)
        x = self.refine_act_2.forward(x)

        x = self.output_conv.forward(x)
        decoded = self.output_act.forward(x)

        if self.skip_connection_weight > 0.0:
            reconstructed = (
                (1.0 - self.skip_connection_weight) * decoded
                + self.skip_connection_weight * noisy_images
            ).astype(np.float32)
            return reconstructed

        return decoded

    @staticmethod
    def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean((predictions - targets) ** 2))

    @staticmethod
    def mse_grad(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (2.0 / predictions.size) * (predictions - targets)

    @staticmethod
    def mae_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(np.abs(predictions - targets)))

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
        grad = loss_grad
        if self.skip_connection_weight > 0.0:
            grad = grad * (1.0 - self.skip_connection_weight)

        grad = self.output_act.backward(grad)
        grad = self.output_conv.backward(grad)

        grad = self.refine_act_2.backward(grad)
        grad = self.refine_conv_2.backward(grad)

        grad = self.refine_act.backward(grad)
        grad = self.refine_conv.backward(grad)

        grad = self.decoder_act.backward(grad)
        grad = self.decoder_conv.backward(grad)

        grad = self.decoder_upsample.backward(grad)

        grad = self.bottleneck_refine_act.backward(grad)
        grad = self.bottleneck_refine_conv.backward(grad)

        grad = self.bottleneck_act.backward(grad)
        grad = self.bottleneck_conv.backward(grad)

        grad = self.encoder_act.backward(grad)
        _ = self.encoder_conv.backward(grad)

        self.encoder_conv.apply_gradients(learning_rate, weight_decay)
        self.bottleneck_conv.apply_gradients(learning_rate, weight_decay)
        self.bottleneck_refine_conv.apply_gradients(learning_rate, weight_decay)
        self.decoder_conv.apply_gradients(learning_rate, weight_decay)
        self.refine_conv.apply_gradients(learning_rate, weight_decay)
        self.refine_conv_2.apply_gradients(learning_rate, weight_decay)
        self.output_conv.apply_gradients(learning_rate, weight_decay)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "meta_input_channels": np.asarray([self.input_channels], dtype=np.int64),
            "meta_latent_channels": np.asarray([self.latent_channels], dtype=np.int64),
            "meta_l1_weight": np.asarray([self.l1_weight], dtype=np.float32),
            "meta_skip_connection_weight": np.asarray([self.skip_connection_weight], dtype=np.float32),
            "encoder_conv.weights": self.encoder_conv.weights.copy(),
            "encoder_conv.bias": self.encoder_conv.bias.copy(),
            "bottleneck_conv.weights": self.bottleneck_conv.weights.copy(),
            "bottleneck_conv.bias": self.bottleneck_conv.bias.copy(),
            "bottleneck_refine_conv.weights": self.bottleneck_refine_conv.weights.copy(),
            "bottleneck_refine_conv.bias": self.bottleneck_refine_conv.bias.copy(),
            "decoder_conv.weights": self.decoder_conv.weights.copy(),
            "decoder_conv.bias": self.decoder_conv.bias.copy(),
            "refine_conv.weights": self.refine_conv.weights.copy(),
            "refine_conv.bias": self.refine_conv.bias.copy(),
            "refine_conv_2.weights": self.refine_conv_2.weights.copy(),
            "refine_conv_2.bias": self.refine_conv_2.bias.copy(),
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

        if "meta_l1_weight" in state:
            self.l1_weight = float(np.asarray(state["meta_l1_weight"]).ravel()[0])

        if "meta_skip_connection_weight" in state:
            self.skip_connection_weight = float(
                np.asarray(state["meta_skip_connection_weight"]).ravel()[0]
            )

        self.encoder_conv.weights = np.asarray(state["encoder_conv.weights"], dtype=np.float32).copy()
        self.encoder_conv.bias = np.asarray(state["encoder_conv.bias"], dtype=np.float32).copy()

        self.bottleneck_conv.weights = np.asarray(
            state["bottleneck_conv.weights"], dtype=np.float32
        ).copy()
        self.bottleneck_conv.bias = np.asarray(state["bottleneck_conv.bias"], dtype=np.float32).copy()

        if "bottleneck_refine_conv.weights" in state and "bottleneck_refine_conv.bias" in state:
            self.bottleneck_refine_conv.weights = np.asarray(
                state["bottleneck_refine_conv.weights"], dtype=np.float32
            ).copy()
            self.bottleneck_refine_conv.bias = np.asarray(
                state["bottleneck_refine_conv.bias"], dtype=np.float32
            ).copy()

        self.decoder_conv.weights = np.asarray(state["decoder_conv.weights"], dtype=np.float32).copy()
        self.decoder_conv.bias = np.asarray(state["decoder_conv.bias"], dtype=np.float32).copy()

        if "refine_conv.weights" in state and "refine_conv.bias" in state:
            self.refine_conv.weights = np.asarray(
                state["refine_conv.weights"], dtype=np.float32
            ).copy()
            self.refine_conv.bias = np.asarray(state["refine_conv.bias"], dtype=np.float32).copy()

        if "refine_conv_2.weights" in state and "refine_conv_2.bias" in state:
            self.refine_conv_2.weights = np.asarray(
                state["refine_conv_2.weights"], dtype=np.float32
            ).copy()
            self.refine_conv_2.bias = np.asarray(state["refine_conv_2.bias"], dtype=np.float32).copy()

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
