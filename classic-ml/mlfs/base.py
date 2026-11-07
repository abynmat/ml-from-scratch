"""Shared building blocks: the Estimator base class, input checks, a scaler and metrics.

Author: Abyn Scaria

Every model in `mlfs` subclasses `Estimator` and exposes the same surface:

    model = SomeModel(**hyperparameters).fit(X, y)
    model.predict(X)            # labels or real values
    model.predict_proba(X)      # classifiers that have probabilities
    model.score(X, y)           # accuracy (classifiers) or R^2 (regressors)

Hyperparameters are plain keyword arguments stored on `self` under the same name, which lets
`get_params()` rebuild an identical, unfitted copy (used by the forest and the benchmarks).
"""
import inspect

import numpy as np


# ---------------------------------------------------------------------------
# Estimator base
# ---------------------------------------------------------------------------
class Estimator:
    """Minimal estimator protocol: keyword hyperparameters, fit returns self."""

    _kind = "generic"  # "classifier", "regressor", "cluster" or "transform"

    def get_params(self):
        names = [p for p in inspect.signature(self.__init__).parameters if p != "self"]
        return {n: getattr(self, n) for n in names}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                raise ValueError(f"unknown hyperparameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def clone(self):
        return type(self)(**self.get_params())

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def fit(self, X, y=None):  # pragma: no cover - interface only
        raise NotImplementedError

    def score(self, X, y):
        if self._kind == "classifier":
            return accuracy(y, self.predict(X))
        if self._kind == "regressor":
            return r2(y, self.predict(X))
        raise NotImplementedError(f"{type(self).__name__} has no default score")


class Classifier(Estimator):
    _kind = "classifier"

    def _encode(self, y):
        """Store the sorted class labels and return y as integer codes 0..K-1."""
        self.classes_, codes = np.unique(y, return_inverse=True)
        return codes

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class Regressor(Estimator):
    _kind = "regressor"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def check_X(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or inf")
    return X


def check_X_y(X, y):
    X = check_X(X)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be 1-D with one entry per row of X")
    return X, y


def as_rng(random_state):
    return random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)


class StandardScaler(Estimator):
    """z = (x - mean) / std, with std=1 for constant columns."""

    _kind = "transform"

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = check_X(X)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X):
        return (check_X(X) - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z):
    # split by sign so neither branch overflows
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    ss_res = np.sum((y_true - np.asarray(y_pred, float)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def log_loss(y_true, proba, eps=1e-15):
    """Mean negative log-likelihood. `y_true` are integer codes 0..K-1, `proba` is (n, K)."""
    proba = np.clip(np.asarray(proba, float), eps, 1 - eps)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), np.asarray(y_true)])))


def adjusted_rand_index(a, b):
    """Hubert & Arabie (1985) ARI from the contingency table. 1 = identical partitions."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = comb2(table).sum()
    sum_a = comb2(table.sum(axis=1)).sum()
    sum_b = comb2(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(len(a))
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))
