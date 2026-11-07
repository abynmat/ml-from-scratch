"""Principal component analysis by the singular value decomposition of the centred data.

Author: Abyn Scaria
References: Bishop, PRML, section 12.1; ESL section 14.5.1.

If Xc = U S V^T then the principal axes are the rows of V^T and the variance captured by axis k is
S_k^2 / (n - 1). Singular vectors are only defined up to sign; we flip each axis so its largest-
magnitude loading is positive, which makes results deterministic and comparable across libraries.
"""
import numpy as np

from .base import Estimator, check_X


class PCA(Estimator):
    _kind = "transform"

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        X = check_X(X)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]
        k = min(n, d) if self.n_components is None else self.n_components
        var = S ** 2 / (n - 1)
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        self.singular_values_ = S[:k]
        return self

    def transform(self, X):
        Z = (check_X(X) - self.mean_) @ self.components_.T
        return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_
