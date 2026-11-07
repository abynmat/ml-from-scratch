"""Gaussian mixture model with full covariances, fitted by expectation-maximisation.

Author: Abyn Scaria
References: Bishop, PRML, section 9.2.2 (EM for Gaussian mixtures); Dempster, Laird & Rubin (1977).

All densities are evaluated in log space through a Cholesky factor of each covariance, and the
E-step normalises with log-sum-exp, so components far from a point do not underflow to 0/0.
"""
import numpy as np

from .base import Estimator, as_rng, check_X, logsumexp
from .kmeans import KMeans


class GaussianMixture(Estimator):
    """init_params: "kmeans" (responsibilities from a k-means labelling) or "random".
    `reg_covar` is added to every covariance diagonal to keep it positive definite.
    EM only finds a local optimum, so `n_init` restarts are run and the highest likelihood kept."""

    _kind = "cluster"

    def __init__(self, n_components=1, max_iter=100, tol=1e-3, reg_covar=1e-6, init_params="kmeans",
                 n_init=1, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_params = init_params
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_X(X)
        rng = as_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            self._fit_once(X, rng)
            if best is None or self.lower_bound_curve_[-1] > best[0]:
                best = (self.lower_bound_curve_[-1], self.weights_, self.means_, self.covariances_,
                        self.lower_bound_curve_, self.converged_, self.n_iter_)
        (_, self.weights_, self.means_, self.covariances_, self.lower_bound_curve_, self.converged_,
         self.n_iter_) = best
        return self

    def _fit_once(self, X, rng):
        n = len(X)
        K = self.n_components
        if self.init_params == "kmeans":
            labels = KMeans(K, n_init=1, random_state=rng).fit(X).labels_
            R = np.eye(K)[labels]
        else:
            R = rng.random((n, K))
            R /= R.sum(axis=1, keepdims=True)
        self._m_step(X, R)
        self.lower_bound_curve_ = []
        prev = -np.inf
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            log_resp, ll = self._e_step(X)
            self._m_step(X, np.exp(log_resp))
            self.lower_bound_curve_.append(ll)
            if abs(ll - prev) < self.tol:
                self.converged_ = True
                break
            prev = ll
        self.n_iter_ = it

    # E-step: log r_ik = log pi_k + log N(x_i | mu_k, S_k) - log sum_k(...)
    def _e_step(self, X):
        wlp = self._weighted_log_prob(X)
        norm = logsumexp(wlp, axis=1, keepdims=True)
        return wlp - norm, float(norm.mean())

    # M-step: closed-form weighted means, covariances and mixing weights
    def _m_step(self, X, R):
        n, d = X.shape
        Nk = R.sum(axis=0) + 10 * np.finfo(float).eps
        self.weights_ = Nk / n
        self.means_ = (R.T @ X) / Nk[:, None]
        cov = np.empty((len(Nk), d, d))
        for k in range(len(Nk)):
            diff = X - self.means_[k]
            cov[k] = (R[:, k, None] * diff).T @ diff / Nk[k] + self.reg_covar * np.eye(d)
        self.covariances_ = cov

    def _weighted_log_prob(self, X):
        n, d = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            L = np.linalg.cholesky(self.covariances_[k])
            z = np.linalg.solve(L, (X - self.means_[k]).T)  # L z = (x - mu)
            maha = np.sum(z * z, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
        return out + np.log(self.weights_)

    def predict_proba(self, X):
        return np.exp(self._e_step(check_X(X))[0])

    def predict(self, X):
        return np.argmax(self._weighted_log_prob(check_X(X)), axis=1)

    def score(self, X, y=None):
        """Mean log-likelihood per sample (same definition as sklearn)."""
        return self._e_step(check_X(X))[1]

    def bic(self, X):
        X = check_X(X)
        d = X.shape[1]
        n_params = self.n_components * (d + d * (d + 1) / 2) + self.n_components - 1
        return -2 * self.score(X) * len(X) + n_params * np.log(len(X))
