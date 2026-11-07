"""Gradient tree boosting (Friedman, 2001) for regression and binary classification.

Author: Abyn Scaria
References: ESL chapter 10, Algorithm 10.3 (gradient tree boosting); Friedman (2001) "Greedy Function
Approximation: A Gradient Boosting Machine", section 4.5, for the Newton leaf values of the
two-class logistic loss.
"""
import numpy as np

from .base import Classifier, Regressor, as_rng, check_X, check_X_y, sigmoid
from .tree import DecisionTreeRegressor


class _Boosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1,
                 subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def _new_tree(self, rng):
        return DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     random_state=rng.integers(2 ** 31))

    def _rows(self, rng, n):
        if self.subsample >= 1.0:
            return np.arange(n)
        return rng.choice(n, int(self.subsample * n), replace=False)

    def _raw(self, X):
        X = check_X(X)
        F = np.full(len(X), self.init_)
        for t in self.estimators_:
            F += self.learning_rate * t.predict(X)
        return F

    def staged_raw(self, X):
        """Yields the additive model after each stage (for learning curves)."""
        X = check_X(X)
        F = np.full(len(X), self.init_)
        for t in self.estimators_:
            F = F + self.learning_rate * t.predict(X)
            yield F


class GradientBoostingRegressor(_Boosting, Regressor):
    """Squared-error boosting: each tree fits the residual y - F, leaf value = mean residual."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = y.astype(float)
        rng = as_rng(self.random_state)
        self.init_ = float(y.mean())
        F = np.full(len(y), self.init_)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            rows = self._rows(rng, len(y))
            tree = self._new_tree(rng).fit(X[rows], (y - F)[rows])
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        return self._raw(X)


class GradientBoostingClassifier(_Boosting, Classifier):
    """Binary log-loss boosting on the log-odds F(x).

    Each stage fits a regression tree to the negative gradient r = y - sigmoid(F), then replaces
    every leaf value with a single Newton-Raphson step for the log-loss restricted to that leaf:
        gamma_leaf = sum(r) / sum(p (1 - p))
    """

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        codes = self._encode(y)
        if len(self.classes_) != 2:
            raise ValueError("this GradientBoostingClassifier is binary only")
        y = codes.astype(float)
        rng = as_rng(self.random_state)
        p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        self.init_ = float(np.log(p0 / (1 - p0)))
        F = np.full(len(y), self.init_)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            rows = self._rows(rng, len(y))
            p = sigmoid(F)
            r = y - p
            tree = self._new_tree(rng).fit(X[rows], r[rows])
            leaves = tree.apply(X[rows])
            num = np.bincount(leaves, weights=r[rows], minlength=len(tree.value_))
            den = np.bincount(leaves, weights=(p * (1 - p))[rows], minlength=len(tree.value_))
            tree.value_[:, 0] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def decision_function(self, X):
        return self._raw(X)

    def predict_proba(self, X):
        p = sigmoid(self._raw(X))
        return np.column_stack([1 - p, p])
