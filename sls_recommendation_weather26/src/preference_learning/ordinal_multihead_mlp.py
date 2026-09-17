#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实现带有序距离损失的共享主干四头分类MLP。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class OrdinalMLPConfig:
    hidden: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    alpha: float = 0.001
    ordinal_strength: float = 1.0
    direction_strength: float = 0.0
    epochs: int = 300
    batch_size: int = 256
    patience: int = 35
    gradient_clip: float = 5.0
    random_state: int = 2026


class OrdinalMultiHeadClassifier:
    """温度和风量使用有序损失，模式使用普通交叉熵。"""

    head_names = (
        "driver_temperature",
        "passenger_temperature",
        "wind",
        "mode",
    )
    head_values = (
        np.arange(-3.0, 3.01, 0.5),
        np.arange(-3.0, 3.01, 0.5),
        np.arange(-2.0, 4.0, 1.0),
        np.arange(0.0, 8.0, 1.0),
    )
    ordinal_heads = (True, True, True, False)

    def __init__(self, config: OrdinalMLPConfig):
        self.config = config
        self.trunk_weights_: list[np.ndarray] = []
        self.trunk_biases_: list[np.ndarray] = []
        self.head_weights_: list[np.ndarray] = []
        self.head_biases_: list[np.ndarray] = []
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.loss_history_: list[dict[str, float]] = []
        self.best_epoch_: int = -1

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(shifted, -60.0, 60.0))
        return exponential / exponential.sum(axis=1, keepdims=True)

    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        self.x_mean_ = x.mean(axis=0)
        self.x_scale_ = x.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-8] = 1.0
        return (x - self.x_mean_) / self.x_scale_

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_scale_ is None:
            raise RuntimeError("模型尚未训练")
        return (x - self.x_mean_) / self.x_scale_

    def _initialize(self, input_dim: int, rng: np.random.Generator) -> None:
        self.trunk_weights_ = []
        self.trunk_biases_ = []
        previous_dim = input_dim
        for hidden_dim in self.config.hidden:
            scale = np.sqrt(2.0 / previous_dim)
            self.trunk_weights_.append(
                rng.normal(0.0, scale, size=(previous_dim, hidden_dim))
            )
            self.trunk_biases_.append(np.zeros(hidden_dim, dtype=float))
            previous_dim = hidden_dim
        self.head_weights_ = []
        self.head_biases_ = []
        for values in self.head_values:
            scale = np.sqrt(1.0 / previous_dim)
            self.head_weights_.append(
                rng.normal(0.0, scale, size=(previous_dim, len(values)))
            )
            self.head_biases_.append(np.zeros(len(values), dtype=float))

    def _forward(self, x: np.ndarray):
        activations = [x]
        preactivations = []
        current = x
        for weight, bias in zip(self.trunk_weights_, self.trunk_biases_):
            preactivation = current @ weight + bias
            preactivations.append(preactivation)
            current = np.maximum(preactivation, 0.0)
            activations.append(current)
        logits = [
            current @ weight + bias
            for weight, bias in zip(self.head_weights_, self.head_biases_)
        ]
        probabilities = [self._softmax(value) for value in logits]
        return activations, preactivations, probabilities

    def _head_loss(
        self,
        head_index: int,
        probability: np.ndarray,
        labels: np.ndarray,
        row_weights: np.ndarray,
    ) -> float:
        normalized_weights = row_weights / max(float(row_weights.sum()), 1e-12)
        selected = probability[np.arange(len(labels)), labels]
        cross_entropy = -float(
            np.sum(normalized_weights * np.log(np.clip(selected, 1e-12, 1.0)))
        )
        if not self.ordinal_heads[head_index]:
            return cross_entropy
        values = self.head_values[head_index]
        target_values = values[labels]
        value_range = float(values.max() - values.min())
        distance = np.abs(values[None, :] - target_values[:, None]) / value_range
        expected_distance = np.sum(probability * distance, axis=1)
        ordinal_loss = float(np.sum(normalized_weights * expected_distance))
        target_sign = np.sign(target_values)
        direction_mask = np.sign(values[None, :]) == target_sign[:, None]
        correct_direction_probability = np.sum(
            probability * direction_mask, axis=1
        )
        direction_loss = -float(
            np.sum(
                normalized_weights
                * np.log(np.clip(correct_direction_probability, 1e-12, 1.0))
            )
        )
        return (
            cross_entropy
            + self.config.ordinal_strength * ordinal_loss
            + self.config.direction_strength * direction_loss
        )

    def _weighted_loss(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        row_weights: np.ndarray,
    ) -> float:
        _, _, probabilities = self._forward(x)
        task_losses = [
            self._head_loss(index, probability, labels[:, index], row_weights)
            for index, probability in enumerate(probabilities)
        ]
        regularization = sum(
            float(np.sum(weight * weight))
            for weight in self.trunk_weights_ + self.head_weights_
        )
        return float(np.mean(task_losses)) + 0.5 * self.config.alpha * regularization

    @staticmethod
    def _copy_parameters(parameters: list[np.ndarray]) -> list[np.ndarray]:
        return [value.copy() for value in parameters]

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        validation_x: np.ndarray,
        validation_labels: np.ndarray,
        row_weights: np.ndarray | None = None,
        validation_row_weights: np.ndarray | None = None,
    ) -> "OrdinalMultiHeadClassifier":
        x = np.asarray(x, dtype=float)
        labels = np.asarray(labels, dtype=int)
        validation_x = np.asarray(validation_x, dtype=float)
        validation_labels = np.asarray(validation_labels, dtype=int)
        if not np.isfinite(x).all() or not np.isfinite(validation_x).all():
            raise ValueError("MLP输入存在非有限数值")
        row_weights = (
            np.ones(len(x), dtype=float)
            if row_weights is None
            else np.asarray(row_weights, dtype=float)
        )
        validation_row_weights = (
            np.ones(len(validation_x), dtype=float)
            if validation_row_weights is None
            else np.asarray(validation_row_weights, dtype=float)
        )
        standardized_x = self._standardize_fit(x)
        standardized_validation = self._standardize(validation_x)
        rng = np.random.default_rng(self.config.random_state)
        self._initialize(x.shape[1], rng)

        parameters = (
            self.trunk_weights_
            + self.trunk_biases_
            + self.head_weights_
            + self.head_biases_
        )
        first_moment = [np.zeros_like(value) for value in parameters]
        second_moment = [np.zeros_like(value) for value in parameters]
        best_parameters = self._copy_parameters(parameters)
        best_loss = float("inf")
        stale_epochs = 0
        update_step = 0

        for epoch in range(self.config.epochs):
            order = rng.permutation(len(standardized_x))
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                batch_x = standardized_x[indices]
                batch_labels = labels[indices]
                batch_weights = row_weights[indices]
                batch_weights = batch_weights / max(float(batch_weights.sum()), 1e-12)
                activations, preactivations, probabilities = self._forward(batch_x)
                head_weight_gradients = []
                head_bias_gradients = []
                trunk_delta = np.zeros_like(activations[-1])

                for head_index, probability in enumerate(probabilities):
                    label = batch_labels[:, head_index]
                    delta = probability.copy()
                    delta[np.arange(len(label)), label] -= 1.0

                    if self.ordinal_heads[head_index]:
                        values = self.head_values[head_index]
                        targets = values[label]
                        value_range = float(values.max() - values.min())
                        distance = np.abs(values[None, :] - targets[:, None]) / value_range
                        expected_distance = np.sum(probability * distance, axis=1, keepdims=True)
                        ordinal_gradient = probability * (distance - expected_distance)
                        delta += self.config.ordinal_strength * ordinal_gradient

                        target_sign = np.sign(targets)
                        direction_mask = np.sign(values[None, :]) == target_sign[:, None]
                        correct_direction_probability = np.sum(
                            probability * direction_mask, axis=1, keepdims=True
                        )
                        direction_gradient = probability - (
                            probability
                            * direction_mask
                            / np.clip(correct_direction_probability, 1e-12, 1.0)
                        )
                        delta += self.config.direction_strength * direction_gradient

                    delta *= batch_weights[:, None] / len(self.head_names)
                    head_weight_gradients.append(
                        activations[-1].T @ delta
                        + self.config.alpha * self.head_weights_[head_index]
                    )
                    head_bias_gradients.append(delta.sum(axis=0))
                    trunk_delta += delta @ self.head_weights_[head_index].T

                trunk_weight_gradients = [
                    np.empty_like(value) for value in self.trunk_weights_
                ]
                trunk_bias_gradients = [
                    np.empty_like(value) for value in self.trunk_biases_
                ]
                delta = trunk_delta
                for layer in reversed(range(len(self.trunk_weights_))):
                    delta = delta * (preactivations[layer] > 0.0)
                    trunk_weight_gradients[layer] = (
                        activations[layer].T @ delta
                        + self.config.alpha * self.trunk_weights_[layer]
                    )
                    trunk_bias_gradients[layer] = delta.sum(axis=0)
                    if layer > 0:
                        delta = delta @ self.trunk_weights_[layer].T

                gradients = (
                    trunk_weight_gradients
                    + trunk_bias_gradients
                    + head_weight_gradients
                    + head_bias_gradients
                )
                total_norm = np.sqrt(
                    sum(float(np.sum(gradient * gradient)) for gradient in gradients)
                )
                if total_norm > self.config.gradient_clip:
                    scale = self.config.gradient_clip / max(total_norm, 1e-12)
                    gradients = [gradient * scale for gradient in gradients]

                update_step += 1
                for parameter, gradient, moment1, moment2 in zip(
                    parameters, gradients, first_moment, second_moment
                ):
                    moment1 *= 0.9
                    moment1 += 0.1 * gradient
                    moment2 *= 0.999
                    moment2 += 0.001 * gradient * gradient
                    corrected1 = moment1 / (1.0 - 0.9**update_step)
                    corrected2 = moment2 / (1.0 - 0.999**update_step)
                    parameter -= self.config.learning_rate * corrected1 / (
                        np.sqrt(corrected2) + 1e-8
                    )

            train_loss = self._weighted_loss(standardized_x, labels, row_weights)
            validation_loss = self._weighted_loss(
                standardized_validation, validation_labels, validation_row_weights
            )
            self.loss_history_.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_parameters = self._copy_parameters(parameters)
                self.best_epoch_ = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                break

        for parameter, best_parameter in zip(parameters, best_parameters):
            parameter[...] = best_parameter
        if not all(np.isfinite(value).all() for value in parameters):
            raise FloatingPointError("MLP训练后参数出现非有限数值")
        return self

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        standardized = self._standardize(np.asarray(x, dtype=float))
        return self._forward(standardized)[2]

    def export_metadata(self) -> dict:
        return {
            "configuration": asdict(self.config),
            "head_names": list(self.head_names),
            "head_values": [values.tolist() for values in self.head_values],
            "ordinal_heads": list(self.ordinal_heads),
            "best_epoch": self.best_epoch_,
            "loss_history": self.loss_history_,
        }
