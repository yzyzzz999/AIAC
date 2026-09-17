#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LabelEncodedMlp:
    """MLP classifier that trains on numeric labels and predicts original labels."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        random_state: int = 42,
        max_iter: int = 450,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.encoder = LabelEncoder()
        self.pipeline = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=8e-4,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=28,
                random_state=self.random_state,
                verbose=False,
            ),
        )
        self.classes_: np.ndarray | None = None

    def fit(self, X, y: Iterable[str], sample_weight=None):
        y_encoded = self.encoder.fit_transform(np.asarray(list(y), dtype=str))
        self.classes_ = self.encoder.classes_
        if sample_weight is not None:
            try:
                self.pipeline.fit(X, y_encoded, mlpclassifier__sample_weight=sample_weight)
                return self
            except TypeError:
                pass
        self.pipeline.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded = self.pipeline.predict(X)
        return self.encoder.inverse_transform(np.asarray(y_encoded, dtype=int))

    def predict_proba(self, X):
        probs_encoded = self.pipeline.predict_proba(X)
        trained_encoded_classes = np.asarray(self.pipeline[-1].classes_, dtype=int)
        full_probs = np.zeros((probs_encoded.shape[0], len(self.encoder.classes_)), dtype=np.float64)
        for col, encoded_cls in enumerate(trained_encoded_classes):
            full_probs[:, int(encoded_cls)] = probs_encoded[:, col]
        return full_probs
