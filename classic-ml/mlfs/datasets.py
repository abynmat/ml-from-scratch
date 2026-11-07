"""Public datasets with fixed, reproducible train/test splits.

Author: Abyn Scaria

Every experiment in this repo gets its data from `load(name)`, so mlfs and scikit-learn always see
the *same* rows in the *same* order. The loaders wrap scikit-learn's bundled toy datasets
(iris, wine, breast_cancer, digits, diabetes), California housing (downloaded once by
scikit-learn and cached locally; we use a 5,000-row random subset with log1p on four heavy-tailed
columns) and two synthetic generators.
"""
from dataclasses import dataclass

import numpy as np
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

from .base import StandardScaler

CLASSIFICATION = ("iris", "wine", "breast_cancer", "digits", "moons", "blobs")
REGRESSION = ("diabetes", "california")


@dataclass
class Split:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list

    @property
    def task(self):
        return "regression" if self.name in REGRESSION else "classification"


def _raw(name, seed):
    if name == "iris":
        d = skd.load_iris()
    elif name == "wine":
        d = skd.load_wine()
    elif name == "breast_cancer":
        d = skd.load_breast_cancer()
    elif name == "digits":
        d = skd.load_digits()
    elif name == "diabetes":
        d = skd.load_diabetes()
    elif name == "california":
        d = skd.fetch_california_housing()
        idx = np.random.default_rng(seed).choice(len(d.target), 5000, replace=False)
        X = d.data[idx].copy()
        # AveRooms, AveBedrms, Population and AveOccup are heavy-tailed ratios/counts (one block
        # has AveOccup > 1000). Without log1p a single test row gives OLS a test R^2 of -3.7 in
        # both libraries; with it, OLS reaches ~0.65. Same transform is applied for everyone.
        X[:, 2:6] = np.log1p(X[:, 2:6])
        return X, d.target[idx], list(d.feature_names)
    elif name == "moons":
        X, y = skd.make_moons(n_samples=500, noise=0.25, random_state=seed)
        return X, y, ["x1", "x2"]
    elif name == "blobs":
        X, y = skd.make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=seed)
        return X, y, ["x1", "x2"]
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return d.data, d.target, list(getattr(d, "feature_names", [f"f{i}" for i in range(d.data.shape[1])]))


def load(name, test_size=0.25, seed=0, scale=True):
    """Return a `Split`. With `scale=True` features are standardised using train-set statistics."""
    X, y, names = _raw(name, seed)
    strat = y if name not in REGRESSION else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    return Split(name, Xtr, Xte, ytr, yte, names)
