"""Random forests (Breiman, 2001): bagged CART trees with a random feature subset per split.

Author: Abyn Scaria
Reference: ESL chapter 15; Breiman (2001) "Random Forests", Machine Learning 45(1).
"""
import numpy as np

from .base import Classifier, Regressor, as_rng, check_X, check_X_y
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class _Forest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _grow(self, X, y, tree_cls, criterion):
        rng = as_rng(self.random_state)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # each tree sees a bootstrap sample and a random feature subset at every split
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = tree_cls(criterion=criterion, max_depth=self.max_depth,
                            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                            random_state=rng.integers(2 ** 31))
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)


class RandomForestClassifier(_Forest, Classifier):
    """Averages the class-probability vectors of bootstrapped gini trees (soft voting)."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        codes = self._encode(y)
        self._grow(X, codes, DecisionTreeClassifier, "gini")
        return self

    def predict_proba(self, X):
        X = check_X(X)
        K = len(self.classes_)
        P = np.zeros((len(X), K))
        for t in self.estimators_:
            # a bootstrap sample can miss a class; map the tree's classes back to global codes
            P[:, t.classes_] += t.predict_proba(X)
        return P / len(self.estimators_)


class RandomForestRegressor(_Forest, Regressor):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features=1.0,
                 bootstrap=True, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_leaf, max_features, bootstrap,
                         random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._grow(X, y.astype(float), DecisionTreeRegressor, "mse")
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)
