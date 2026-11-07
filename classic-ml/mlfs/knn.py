"""k-nearest neighbours for classification and regression (brute-force Euclidean search).

Author: Abyn Scaria
Reference: ESL section 13.3; the curse-of-dimensionality discussion in ESL section 2.5 motivates
experiments/failure_modes.py::knn_high_dimension.
"""
import numpy as np

from .base import Classifier, Regressor, check_X, check_X_y


def pairwise_sq_dists(A, B):
    """||a - b||^2 for every pair via ||a||^2 + ||b||^2 - 2 a.b (one matrix multiply)."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


class _KNNBase:
    def __init__(self, k=5, batch_size=2048):
        self.k = k
        self.batch_size = batch_size

    def _neighbours(self, X):
        X = check_X(X)
        out = np.empty((len(X), self.k), dtype=np.int64)
        for s in range(0, len(X), self.batch_size):  # batch to bound memory at n_test x n_train
            D = pairwise_sq_dists(X[s:s + self.batch_size], self._X)
            idx = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]
            # order the k neighbours by distance, ties broken by training index (stable)
            rows = np.arange(len(idx))[:, None]
            order = np.lexsort((idx, D[rows, idx]), axis=1)
            out[s:s + self.batch_size] = idx[rows, order]
        return out


class KNNClassifier(_KNNBase, Classifier):
    """Majority vote among the k closest training points (uniform weights)."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._y = self._encode(y)
        self._X = X
        return self

    def predict_proba(self, X):
        nb = self._neighbours(X)
        K = len(self.classes_)
        votes = np.zeros((len(nb), K))
        np.add.at(votes, (np.repeat(np.arange(len(nb)), self.k), self._y[nb].ravel()), 1.0)
        return votes / self.k  # argmax picks the lowest class code on ties, like sklearn


class KNNRegressor(_KNNBase, Regressor):
    """Mean target of the k closest training points."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._X, self._y = X, y.astype(float)
        return self

    def predict(self, X):
        return self._y[self._neighbours(X)].mean(axis=1)
