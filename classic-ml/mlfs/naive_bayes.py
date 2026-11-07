"""Gaussian and multinomial naive Bayes.

Author: Abyn Scaria
References: Murphy, "Machine Learning: A Probabilistic Perspective", section 3.5; Manning, Raghavan &
Schuetze, "Introduction to Information Retrieval", ch. 13 (multinomial NB with Laplace smoothing).

Both models compute a joint log-likelihood log p(c) + sum_j log p(x_j | c) and normalise with
log-sum-exp. The "naive" assumption is that features are independent given the class; see
experiments/failure_modes.py::naive_bayes_correlated for what happens when they are not.
"""
import numpy as np

from .base import Classifier, check_X, check_X_y, logsumexp


class _NBBase(Classifier):
    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(check_X(X))
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))


class GaussianNB(_NBBase):
    """x_j | c ~ N(mu_cj, var_cj). `var_smoothing` adds a fraction of the largest feature variance
    to every variance so a constant feature in one class does not give a zero variance."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        codes = self._encode(y)
        K = len(self.classes_)
        eps = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.array([X[codes == c].mean(axis=0) for c in range(K)])
        self.var_ = np.array([X[codes == c].var(axis=0) for c in range(K)]) + eps
        self.class_log_prior_ = np.log(np.bincount(codes, minlength=K) / len(codes))
        return self

    def _joint_log_likelihood(self, X):
        # sum_j log N(x_j; mu, var) for each class, vectorised to (n, K)
        const = -0.5 * np.sum(np.log(2 * np.pi * self.var_), axis=1)
        quad = -0.5 * (((X[:, None, :] - self.theta_[None]) ** 2) / self.var_[None]).sum(axis=2)
        return self.class_log_prior_ + const + quad


class MultinomialNB(_NBBase):
    """x is a vector of non-negative counts; p(x | c) is multinomial with Laplace/Lidstone
    smoothing alpha on the per-class feature counts."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(X < 0):
            raise ValueError("MultinomialNB needs non-negative features (counts)")
        codes = self._encode(y)
        K = len(self.classes_)
        counts = np.array([X[codes == c].sum(axis=0) for c in range(K)]) + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(np.bincount(codes, minlength=K) / len(codes))
        return self

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_
