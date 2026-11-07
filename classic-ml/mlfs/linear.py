"""Linear regression (closed form and gradient descent), ridge and lasso.

Author: Abyn Scaria
References: Hastie, Tibshirani & Friedman, ESL 2nd ed., sections 3.2 (least squares) and 3.4
(shrinkage); Friedman, Hastie & Tibshirani (2010) "Regularization Paths for Generalized Linear
Models via Coordinate Descent" for the lasso update.

Objectives are written to match scikit-learn so coefficients can be compared one-to-one:
    LinearRegression  min ||y - Xw - b||^2
    Ridge             min ||y - Xw - b||^2 + alpha ||w||^2
    Lasso             min (1 / 2n) ||y - Xw - b||^2 + alpha ||w||_1
The intercept b is never penalised. We handle it by centring X and y, solving for w, then
recovering b = mean(y) - mean(X) @ w.
"""
import numpy as np

from .base import Regressor, check_X, check_X_y


class _CenteredLinear(Regressor):
    def _center(self, X, y):
        self._x_mean = X.mean(axis=0)
        self._y_mean = y.mean()
        return X - self._x_mean, y - self._y_mean

    def _finish(self):
        self.intercept_ = float(self._y_mean - self._x_mean @ self.coef_)

    def predict(self, X):
        return check_X(X) @ self.coef_ + self.intercept_


class LinearRegression(_CenteredLinear):
    """Ordinary least squares.

    solver="closed": solve the least-squares problem with `lstsq` (SVD based, so it also copes with
                     rank-deficient X; equivalent to the normal equations X^T X w = X^T y).
    solver="gd":     full-batch gradient descent on the mean squared error. Converges only if
                     lr < 2 / lambda_max(X^T X / n), which is why feature scaling matters
                     (see experiments/failure_modes.py).
    """

    def __init__(self, solver="closed", lr=0.1, n_iter=2000, tol=1e-10):
        self.solver = solver
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xc, yc = self._center(X, y.astype(float))
        if self.solver == "closed":
            self.coef_ = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        elif self.solver == "gd":
            n = len(yc)
            w = np.zeros(X.shape[1])
            self.loss_curve_ = []
            for _ in range(self.n_iter):
                resid = Xc @ w - yc
                loss = float(resid @ resid / n)
                self.loss_curve_.append(loss)
                if not np.isfinite(loss):  # diverged: keep the curve so the failure is visible
                    break
                w -= self.lr * (2.0 / n) * (Xc.T @ resid)
                if len(self.loss_curve_) > 1 and abs(self.loss_curve_[-2] - loss) < self.tol * max(loss, 1e-12):
                    break
            self.coef_ = w
        else:
            raise ValueError("solver must be 'closed' or 'gd'")
        self._finish()
        return self


class Ridge(_CenteredLinear):
    """L2-penalised least squares.

    solver="closed": w = (X^T X + alpha I)^{-1} X^T y on centred data.
    solver="cd":     cyclic coordinate descent; each coordinate has the exact minimiser
                     w_j = x_j^T r_j / (x_j^T x_j + alpha), where r_j is the residual without feature j.
    """

    def __init__(self, alpha=1.0, solver="closed", max_iter=1000, tol=1e-8):
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xc, yc = self._center(X, y.astype(float))
        d = X.shape[1]
        if self.solver == "closed":
            self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ yc)
        elif self.solver == "cd":
            self.coef_, self.n_iter_ = _coordinate_descent(Xc, yc, l1=0.0, l2=self.alpha,
                                                           max_iter=self.max_iter, tol=self.tol)
        else:
            raise ValueError("solver must be 'closed' or 'cd'")
        self._finish()
        return self


class Lasso(_CenteredLinear):
    """L1-penalised least squares by cyclic coordinate descent with soft-thresholding."""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xc, yc = self._center(X, y.astype(float))
        n = len(yc)
        # (1/2n)||r||^2 + alpha|w|_1 has the same minimiser as (1/2)||r||^2 + n*alpha|w|_1
        self.coef_, self.n_iter_ = _coordinate_descent(Xc, yc, l1=n * self.alpha, l2=0.0,
                                                       max_iter=self.max_iter, tol=self.tol)
        self._finish()
        return self


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def _coordinate_descent(X, y, l1, l2, max_iter, tol):
    """Minimise 0.5||y - Xw||^2 + l1 ||w||_1 + 0.5 * l2 ||w||^2 one coordinate at a time.

    With l1 = 0, l2 = alpha this is half of sklearn's ridge objective (same minimiser).
    Setting the derivative in w_j to zero gives w_j = S(rho_j, l1) / (x_j^T x_j + l2),
    where rho_j = x_j^T (y - X w + x_j w_j) and S is soft-thresholding.
    We keep the full residual r = y - Xw up to date so each coordinate update is O(n).
    """
    n, d = X.shape
    w = np.zeros(d)
    col_sq = np.einsum("ij,ij->j", X, X)
    r = y.copy()
    for it in range(1, max_iter + 1):
        max_change = 0.0
        for j in range(d):
            if col_sq[j] == 0:
                continue
            old = w[j]
            rho = X[:, j] @ r + col_sq[j] * old  # x_j^T (residual with feature j removed)
            new = soft_threshold(rho, l1) / (col_sq[j] + l2)
            if new != old:
                r -= X[:, j] * (new - old)
                w[j] = new
                max_change = max(max_change, abs(new - old))
        if max_change <= tol * max(1.0, np.max(np.abs(w))):
            return w, it
    return w, max_iter
