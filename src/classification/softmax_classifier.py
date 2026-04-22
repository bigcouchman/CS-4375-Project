from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


class SoftmaxClassifier:
    """Multiclass classifier implemented with NumPy.

    When ``hidden_dims`` is empty, this behaves as plain softmax regression.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] | None = None,
        dropout_rate: float = 0.0,
        input_noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1.")

        parsed_hidden_dims: tuple[int, ...] = tuple(int(dim) for dim in (hidden_dims or ()))
        if any(dim <= 0 for dim in parsed_hidden_dims):
            raise ValueError("All hidden_dims values must be > 0.")

        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError("dropout_rate must satisfy 0.0 <= dropout_rate < 1.0.")
        if input_noise_std < 0.0:
            raise ValueError("input_noise_std must be >= 0.0.")

        self.dropout_rate = float(dropout_rate)
        self.input_noise_std = float(input_noise_std)
        self.hidden_dims = parsed_hidden_dims

        layer_dims = (int(input_dim), *parsed_hidden_dims, int(num_classes))
        rng = np.random.default_rng(seed)

        self.layer_weights: list[np.ndarray] = []
        self.layer_biases: list[np.ndarray] = []

        for layer_index in range(len(layer_dims) - 1):
            fan_in = layer_dims[layer_index]
            fan_out = layer_dims[layer_index + 1]
            # He init for hidden layers, Xavier-style for output layer.
            init_scale = np.sqrt(2.0 / fan_in) if layer_index < (len(layer_dims) - 2) else np.sqrt(1.0 / fan_in)
            weights = (init_scale * rng.standard_normal((fan_in, fan_out))).astype(np.float32)
            bias = np.zeros((fan_out,), dtype=np.float32)
            self.layer_weights.append(weights)
            self.layer_biases.append(bias)

        # Compatibility aliases for existing usages/tests expecting these attributes.
        self.weights = self.layer_weights[-1]
        self.bias = self.layer_biases[-1]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def _forward_logits(self, features: np.ndarray) -> np.ndarray:
        hidden = features.astype(np.float32, copy=False)

        for weights, bias in zip(self.layer_weights[:-1], self.layer_biases[:-1]):
            hidden = np.maximum((hidden @ weights) + bias, 0.0).astype(np.float32)

        return (hidden @ self.layer_weights[-1]) + self.layer_biases[-1]

    def _forward_train(
        self,
        features: np.ndarray,
        dropout_rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        hidden = features.astype(np.float32, copy=False)
        layer_inputs: list[np.ndarray] = [hidden]
        pre_activations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray] = []
        keep_prob = 1.0 - self.dropout_rate

        for weights, bias in zip(self.layer_weights[:-1], self.layer_biases[:-1]):
            pre_activation = (hidden @ weights) + bias
            pre_activations.append(pre_activation)

            hidden = np.maximum(pre_activation, 0.0).astype(np.float32)

            if self.dropout_rate > 0.0:
                mask = (dropout_rng.random(hidden.shape) < keep_prob).astype(np.float32)
                mask /= keep_prob
                hidden = hidden * mask
            else:
                mask = np.ones_like(hidden, dtype=np.float32)

            dropout_masks.append(mask)
            layer_inputs.append(hidden)

        logits = (hidden @ self.layer_weights[-1]) + self.layer_biases[-1]
        return logits, layer_inputs, pre_activations, dropout_masks

    def _loss_and_grads(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weight_decay: float,
        dropout_rng: np.random.Generator,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        sample_count = features.shape[0]

        logits, layer_inputs, pre_activations, dropout_masks = self._forward_train(
            features,
            dropout_rng,
        )
        probabilities = self._softmax(logits)

        correct_log_probs = -np.log(probabilities[np.arange(sample_count), labels] + 1e-12)
        data_loss = float(np.mean(correct_log_probs))
        reg_loss = 0.0
        for layer_weights in self.layer_weights:
            reg_loss += 0.5 * weight_decay * float(np.sum(layer_weights * layer_weights))
        total_loss = data_loss + reg_loss

        grad_logits = probabilities
        grad_logits[np.arange(sample_count), labels] -= 1.0
        grad_logits /= sample_count

        grad_weights: list[np.ndarray] = [np.empty_like(weights) for weights in self.layer_weights]
        grad_biases: list[np.ndarray] = [np.empty_like(bias) for bias in self.layer_biases]

        output_layer_index = len(self.layer_weights) - 1
        grad_weights[output_layer_index] = (
            layer_inputs[-1].T @ grad_logits
        ) + (weight_decay * self.layer_weights[output_layer_index])
        grad_biases[output_layer_index] = np.sum(grad_logits, axis=0)

        upstream_grad = grad_logits @ self.layer_weights[output_layer_index].T

        for hidden_layer_index in range(output_layer_index - 1, -1, -1):
            if self.dropout_rate > 0.0:
                upstream_grad = upstream_grad * dropout_masks[hidden_layer_index]

            upstream_grad = upstream_grad * (pre_activations[hidden_layer_index] > 0.0)

            grad_weights[hidden_layer_index] = (
                layer_inputs[hidden_layer_index].T @ upstream_grad
            ) + (weight_decay * self.layer_weights[hidden_layer_index])
            grad_biases[hidden_layer_index] = np.sum(upstream_grad, axis=0)

            if hidden_layer_index > 0:
                upstream_grad = upstream_grad @ self.layer_weights[hidden_layer_index].T

        return (
            total_loss,
            [grad.astype(np.float32) for grad in grad_weights],
            [grad.astype(np.float32) for grad in grad_biases],
        )

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
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
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
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0.")
        if early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0.0.")

        use_validation = val_features is not None and val_labels is not None
        validated_val_features: np.ndarray | None = None
        validated_val_labels: np.ndarray | None = None
        if use_validation:
            validated_val_features = np.asarray(val_features, dtype=np.float32)
            validated_val_labels = np.asarray(val_labels, dtype=np.int64)
            if validated_val_features.ndim != 2:
                raise ValueError("val_features must have shape (N, D).")
            if validated_val_labels.ndim != 1:
                raise ValueError("val_labels must have shape (N,).")
            if validated_val_features.shape[0] != validated_val_labels.shape[0]:
                raise ValueError("val_features and val_labels must have same number of samples.")
            if validated_val_features.shape[1] != features.shape[1]:
                raise ValueError("val_features must have same feature dimension as features.")

        has_validation = (
            validated_val_features is not None and validated_val_labels is not None
        )

        best_val_accuracy = float("-inf")
        epochs_without_improvement = 0
        best_state: tuple[list[np.ndarray], list[np.ndarray]] | None = None

        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []
            dropout_rng = np.random.default_rng(seed + (epoch * 9973))

            for x_batch, y_batch in self._iterate_minibatches(
                features,
                labels,
                batch_size=batch_size,
                seed=seed + epoch,
            ):
                train_batch = x_batch
                if self.input_noise_std > 0.0:
                    feature_noise = dropout_rng.normal(
                        loc=0.0,
                        scale=self.input_noise_std,
                        size=x_batch.shape,
                    ).astype(np.float32)
                    train_batch = (x_batch + feature_noise).astype(np.float32)

                loss_value, grad_weights, grad_biases = self._loss_and_grads(
                    train_batch,
                    y_batch,
                    weight_decay,
                    dropout_rng,
                )
                for layer_index in range(len(self.layer_weights)):
                    self.layer_weights[layer_index] -= learning_rate * grad_weights[layer_index]
                    self.layer_biases[layer_index] -= learning_rate * grad_biases[layer_index]
                epoch_losses.append(loss_value)

            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            train_accuracy = self.score(features, labels)
            if has_validation:
                assert validated_val_features is not None
                assert validated_val_labels is not None
                val_accuracy = self.score(validated_val_features, validated_val_labels)
            else:
                val_accuracy = float("nan")

            history_entry = {
                "epoch": float(epoch),
                "loss": mean_epoch_loss,
                "accuracy": float(train_accuracy),
            }
            if has_validation:
                history_entry["val_accuracy"] = float(val_accuracy)

            history.append(history_entry)

            if has_validation:
                improved = (val_accuracy - best_val_accuracy) > early_stopping_min_delta
                if improved:
                    best_val_accuracy = val_accuracy
                    epochs_without_improvement = 0
                    best_state = (
                        [weights.copy() for weights in self.layer_weights],
                        [bias.copy() for bias in self.layer_biases],
                    )
                else:
                    epochs_without_improvement += 1

            if verbose:
                progress = (
                    f"[Classifier] Epoch {epoch:02d}/{epochs} | "
                    f"loss={mean_epoch_loss:.6f} | accuracy={train_accuracy:.4f}"
                )
                if has_validation:
                    progress += f" | val_accuracy={val_accuracy:.4f}"
                print(progress)

            if (
                has_validation
                and early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                if verbose:
                    print(
                        "[Classifier] Early stopping triggered: "
                        f"no val_accuracy improvement > {early_stopping_min_delta:.6f} "
                        f"for {early_stopping_patience} epochs."
                    )
                break

        if has_validation and best_state is not None:
            best_weights, best_biases = best_state
            self.layer_weights = best_weights
            self.layer_biases = best_biases
            self.weights = self.layer_weights[-1]
            self.bias = self.layer_biases[-1]

        return history

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self._forward_logits(features)
        return self._softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.argmax(probabilities, axis=1)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(features)
        return float(np.mean(predictions == labels))
