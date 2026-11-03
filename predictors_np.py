from __future__ import annotations

import numpy as np


class PersistencePredictor:
    name = "persistence"

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)[:, -1]

    def predict(self, hist_norm: np.ndarray) -> float:
        return float(hist_norm[-1])


class LinearARPredictor:
    name = "linear_AR"

    def __init__(self):
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearARPredictor":
        A = np.c_[np.asarray(X, dtype=np.float64), np.ones(len(X))]
        self.w, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=np.float64), rcond=None)
        return self

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X @ self.w[:-1] + self.w[-1], 0.0, 1.0)

    def predict(self, hist_norm: np.ndarray) -> float:
        return float(self.predict_batch(np.asarray(hist_norm)[None, :])[0])


def mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))


def gain_pct(new: float, ref: float) -> float:
    return float("nan") if ref <= 0 else 100.0 * (1.0 - new / ref)
