from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class SoftmaxClassifier:
    """Multiclass softmax regression implemented with NumPy."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.weights = (0.01 * rng.standard_normal((input_dim, num_classes))).astype(np.float32)
        self.bias = np.zeros((num_classes,), dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def _loss_and_grads(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weight_decay: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        sample_count = features.shape[0]

        logits = features @ self.weights + self.bias
        probabilities = self._softmax(logits)

        correct_log_probs = -np.log(probabilities[np.arange(sample_count), labels] + 1e-12)
        data_loss = float(np.mean(correct_log_probs))
        reg_loss = 0.5 * weight_decay * float(np.sum(self.weights * self.weights))
        total_loss = data_loss + reg_loss

        grad_logits = probabilities
        grad_logits[np.arange(sample_count), labels] -= 1.0
        grad_logits /= sample_count

        grad_weights = features.T @ grad_logits + (weight_decay * self.weights)
        grad_bias = np.sum(grad_logits, axis=0)

        return total_loss, grad_weights.astype(np.float32), grad_bias.astype(np.float32)

    @staticmethod
    def _iterate_minibatches(
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        indices = np.arange(features.shape[0])
        rng.shuffle(indices)

        for start in range(0, features.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            yield features[batch_indices], labels[batch_indices]

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        seed: int = 42,
        verbose: bool = False,
    ) -> list[dict[str, float]]:
        if features.ndim != 2:
            raise ValueError("features must have shape (N, D).")
        if labels.ndim != 1:
            raise ValueError("labels must have shape (N,).")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have same number of samples.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []

            for x_batch, y_batch in self._iterate_minibatches(
                features,
                labels,
                batch_size=batch_size,
                seed=seed + epoch,
            ):
                loss_value, grad_weights, grad_bias = self._loss_and_grads(
                    x_batch,
                    y_batch,
                    weight_decay,
                )
                self.weights -= learning_rate * grad_weights
                self.bias -= learning_rate * grad_bias
                epoch_losses.append(loss_value)

            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            train_accuracy = self.score(features, labels)

            history.append(
                {
                    "epoch": float(epoch),
                    "loss": mean_epoch_loss,
                    "accuracy": float(train_accuracy),
                }
            )

            if verbose:
                print(
                    f"[Classifier] Epoch {epoch:02d}/{epochs} | "
                    f"loss={mean_epoch_loss:.6f} | accuracy={train_accuracy:.4f}"
                )

        return history

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        return self._softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.argmax(probabilities, axis=1)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(features)
        return float(np.mean(predictions == labels))
