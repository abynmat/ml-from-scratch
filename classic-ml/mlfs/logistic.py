"""Binary and multinomial logistic regression (Newton-Raphson / IRLS and gradient descent).

Author: Abyn Scaria
References: Bishop, PRML, section 4.3.3 (IRLS for logistic regression); ESL section 4.4.
Penalty matches sklearn's default: 0.5 * (1/C) * ||w||^2 on the weights, intercept unpenalised.
"""
import numpy as np

from .base import Classifier, check_X, check_X_y, sigmoid, softmax


class LogisticRegression(Classifier):
    """Logistic regression with an L2 penalty.

    solver="newton": Newton-Raphson (IRLS) with the exact Hessian
                     H = X^T diag(p(1-p)) X + (1/C) I. Binary only; with K>2 classes the model
                     falls back to gradient descent on the softmax (multinomial) likelihood.
    solver="gd":     full-batch gradient descent on the penalised negative log-likelihood.
    """

    def __init__(self, C=1.0, solver="newton", n_iter=100, lr=0.5, tol=1e-8):
        self.C = C
        self.solver = solver
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol

    # -- public -----------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = self._encode(y)
        Xb = _add_bias(X)
        self._multiclass = len(self.classes_) > 2
        if self._multiclass:
            self._fit_multinomial(Xb, y)
        else:
            self._fit_binary(Xb, y.astype(float))
        return self

    def decision_function(self, X):
        Xb = _add_bias(check_X(X))
        return Xb @ self._W if self._multiclass else Xb @ self._w

    def predict_proba(self, X):
        z = self.decision_function(X)
        if self._multiclass:
            return softmax(z)
        p = sigmoid(z)
        return np.column_stack([1 - p, p])

    # -- binary -----------------------------------------------------------
    def _fit_binary(self, Xb, y):
        n, d = Xb.shape
        w = np.zeros(d)
        reg = np.ones(d) / self.C
        reg[-1] = 0.0  # bias column last; do not penalise it
        self.loss_curve_ = []
        for _ in range(self.n_iter):
            p = sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) + reg * w
            self.loss_curve_.append(self._nll_binary(Xb, y, w, reg))
            if self.solver == "newton":
                s = np.clip(p * (1 - p), 1e-9, None)
                H = (Xb * s[:, None]).T @ Xb + np.diag(reg)
                step = np.linalg.solve(H, grad)
            elif self.solver == "gd":
                step = self.lr * grad / n
            else:
                raise ValueError("solver must be 'newton' or 'gd'")
            w_new = w - step
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new
        self._w = w

    def _nll_binary(self, Xb, y, w, reg):
        z = Xb @ w
        # log(1+e^z) computed stably
        ll = np.sum(np.logaddexp(0, z) - y * z)
        return float(ll + 0.5 * np.sum(reg * w * w))

    # -- multinomial ------------------------------------------------------
    def _fit_multinomial(self, Xb, y):
        n, d = Xb.shape
        K = len(self.classes_)
        Y = np.eye(K)[y]
        W = np.zeros((d, K))
        reg = np.ones(d) / self.C
        reg[-1] = 0.0
        self.loss_curve_ = []
        for _ in range(self.n_iter):
            P = softmax(Xb @ W)
            grad = Xb.T @ (P - Y) + reg[:, None] * W
            self.loss_curve_.append(float(-np.sum(Y * np.log(np.clip(P, 1e-12, 1)))
                                          + 0.5 * np.sum(reg[:, None] * W * W)))
            W_new = W - self.lr * grad / n
            if np.max(np.abs(W_new - W)) < self.tol:
                W = W_new
                break
            W = W_new
        self._W = W


def _add_bias(X):
    return np.column_stack([X, np.ones(len(X))])
