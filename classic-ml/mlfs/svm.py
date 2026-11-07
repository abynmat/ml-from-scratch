"""Binary soft-margin SVM trained with Sequential Minimal Optimisation.

Author: Abyn Scaria
References:
  Platt (1998) "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector
  Machines" - the idea of optimising two multipliers at a time analytically.
  Fan, Chen & Lin (2005) "Working Set Selection Using Second Order Information for Training SVM",
  JMLR 6 - the working-set rule used below (their WSS 3, the rule LIBSVM uses). Platt's original
  heuristic loops are replaced by it because it has a clean stopping criterion.

Dual problem (labels y in {-1, +1}, Q_ij = y_i y_j K(x_i, x_j)):
    min_a  0.5 a^T Q a - 1^T a     s.t.  0 <= a_i <= C,  y^T a = 0
Decision function: f(x) = sum_i a_i y_i K(x_i, x) + b.
"""
import numpy as np

from .base import Classifier, check_X, check_X_y
from .knn import pairwise_sq_dists


def kernel_matrix(A, B, kernel, gamma, degree=3, coef0=0.0):
    if kernel == "linear":
        return A @ B.T
    if kernel == "rbf":
        return np.exp(-gamma * pairwise_sq_dists(A, B))
    if kernel == "poly":
        return (gamma * A @ B.T + coef0) ** degree
    raise ValueError("kernel must be 'linear', 'rbf' or 'poly'")


class SVC(Classifier):
    """Binary SVM. `gamma="scale"` uses 1 / (n_features * X.var()), the same default as sklearn.

    After fitting, `n_iter_` is the number of SMO pair updates and `converged_` says whether the
    KKT gap fell below `tol` before `max_iter` (see the SMO-vs-C failure-mode experiment).
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0, tol=1e-3,
                 max_iter=100_000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        codes = self._encode(y)
        if len(self.classes_) != 2:
            raise ValueError("this SVC is binary only")
        y = np.where(codes == 1, 1.0, -1.0)
        self._gamma = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)
        K = kernel_matrix(X, X, self.kernel, self._gamma, self.degree, self.coef0)
        a, b, self.n_iter_, self.converged_ = _smo(K, y, float(self.C), self.tol, self.max_iter)
        sv = a > 1e-8
        self.support_ = np.where(sv)[0]
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (a * y)[sv]
        self.intercept_ = b
        return self

    def decision_function(self, X):
        Kx = kernel_matrix(check_X(X), self.support_vectors_, self.kernel, self._gamma,
                           self.degree, self.coef0)
        return Kx @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def predict_proba(self, X):  # not calibrated; only here so the shared API works
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])


def _smo(K, y, C, tol, max_iter, tau=1e-12):
    n = len(y)
    a = np.zeros(n)
    G = -np.ones(n)  # gradient of the dual objective, Q a - 1, at a = 0
    diag = np.diag(K).copy()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # I_up: indices whose -y_t G_t can still increase; I_low: can still decrease
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))
        low = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        score = -y * G
        i = int(np.argmax(np.where(up, score, -np.inf)))
        m = score[i]
        M = np.min(np.where(low, score, np.inf))
        if m - M < tol:  # maximal KKT violation below tolerance -> optimal
            converged = True
            break
        # second-order choice of j (Fan et al. 2005, WSS 3): maximise the objective decrease
        bgap = m - score
        cand = low & (bgap > 0)
        quad = diag[i] + diag - 2.0 * K[i]
        quad = np.where(quad > 0, quad, tau)
        j = int(np.argmin(np.where(cand, -(bgap ** 2) / quad, np.inf)))
        # move along a_i += y_i t, a_j -= y_j t (keeps y^T a fixed); unconstrained optimum t*
        t = bgap[j] / quad[j]
        # clip t so both multipliers stay inside the box [0, C]
        t_i = C - a[i] if y[i] > 0 else a[i]
        t_j = a[j] if y[j] > 0 else C - a[j]
        t = min(t, t_i, t_j)
        da_i, da_j = y[i] * t, -y[j] * t
        a[i] += da_i
        a[j] += da_j
        # G = Q a - 1 changes by Q[:, i] da_i + Q[:, j] da_j, with Q[:, k] = y * y_k * K[:, k]
        G += y * (y[i] * da_i * K[:, i] + y[j] * da_j * K[:, j])
    # bias: for free SVs (0 < a < C) the KKT condition gives b = -y_t G_t exactly
    free = (a > 1e-8) & (a < C - 1e-8)
    score = -y * G
    if free.any():
        b = float(np.mean(score[free]))
    else:
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))
        low = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        b = float(0.5 * (np.max(score[up]) + np.min(score[low])))
    return a, b, it, converged
