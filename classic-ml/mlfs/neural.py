"""Rosenblatt perceptron and a one-hidden-layer MLP trained with Adam, both in NumPy.

Author: Abyn Scaria
References: Rosenblatt (1958); Bishop, PRML, section 5.3 (backpropagation); Kingma & Ba (2015)
"Adam: A Method for Stochastic Optimization".
"""
import numpy as np

from .base import Classifier, as_rng, check_X, check_X_y, softmax


class Perceptron(Classifier):
    """Binary perceptron: on every misclassified point, w <- w + lr * y_i x_i (y in {-1, +1}).
    Converges only if the data are linearly separable; otherwise it cycles, so we cap the epochs."""

    def __init__(self, max_epochs=50, lr=1.0, shuffle=True, random_state=None):
        self.max_epochs = max_epochs
        self.lr = lr
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        codes = self._encode(y)
        if len(self.classes_) != 2:
            raise ValueError("Perceptron is binary only")
        s = np.where(codes == 1, 1.0, -1.0)
        rng = as_rng(self.random_state)
        w, b = np.zeros(X.shape[1]), 0.0
        self.errors_ = []
        for _ in range(self.max_epochs):
            mistakes = 0
            for i in (rng.permutation(len(X)) if self.shuffle else range(len(X))):
                if s[i] * (X[i] @ w + b) <= 0:
                    w += self.lr * s[i] * X[i]
                    b += self.lr * s[i]
                    mistakes += 1
            self.errors_.append(mistakes)
            if mistakes == 0:
                break
        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X):
        return check_X(X) @ self.coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def predict_proba(self, X):  # hard 0/1 "probabilities", for API symmetry only
        p = (self.decision_function(X) > 0).astype(float)
        return np.column_stack([1 - p, p])


class MLPClassifier(Classifier):
    """input -> Linear -> ReLU (or tanh) -> Linear -> softmax, cross-entropy + L2 (alpha), Adam."""

    def __init__(self, hidden=64, activation="relu", alpha=1e-4, lr=1e-3, batch_size=200,
                 epochs=200, tol=1e-4, patience=10, random_state=None):
        self.hidden = hidden
        self.activation = activation
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.tol = tol
        self.patience = patience
        self.random_state = random_state

    def _act(self, z):
        return np.maximum(z, 0) if self.activation == "relu" else np.tanh(z)

    def _act_grad(self, h):
        # derivative written in terms of the activation output h
        return (h > 0).astype(float) if self.activation == "relu" else 1.0 - h * h

    def _forward(self, X):
        H = self._act(X @ self.W1 + self.b1)
        return H, softmax(H @ self.W2 + self.b2)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        codes = self._encode(y)
        n, d = X.shape
        K = len(self.classes_)
        Y = np.eye(K)[codes]
        rng = as_rng(self.random_state)
        # Glorot-uniform initialisation
        lim1, lim2 = np.sqrt(6 / (d + self.hidden)), np.sqrt(6 / (self.hidden + K))
        self.W1 = rng.uniform(-lim1, lim1, (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.uniform(-lim2, lim2, (self.hidden, K))
        self.b2 = np.zeros(K)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1_, b2_, eps, step = 0.9, 0.999, 1e-8, 0
        self.loss_curve_ = []
        best, stall = np.inf, 0
        bs = min(self.batch_size, n)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                Xb, Yb = X[idx], Y[idx]
                H, P = self._forward(Xb)
                nb = len(idx)
                epoch_loss += -np.sum(Yb * np.log(np.clip(P, 1e-12, 1)))
                # backprop: dL/dlogits = (P - Y) / nb for mean cross-entropy
                dZ2 = (P - Yb) / nb
                gW2 = H.T @ dZ2 + self.alpha * self.W2 / nb
                gb2 = dZ2.sum(axis=0)
                dZ1 = (dZ2 @ self.W2.T) * self._act_grad(H)
                gW1 = Xb.T @ dZ1 + self.alpha * self.W1 / nb
                gb1 = dZ1.sum(axis=0)
                step += 1
                for p, g, mi, vi in zip(params, (gW1, gb1, gW2, gb2), m, v):
                    mi *= b1_
                    mi += (1 - b1_) * g
                    vi *= b2_
                    vi += (1 - b2_) * g * g
                    mhat = mi / (1 - b1_ ** step)
                    vhat = vi / (1 - b2_ ** step)
                    p -= self.lr * mhat / (np.sqrt(vhat) + eps)  # in-place, so self.W1 etc. update
            loss = epoch_loss / n + 0.5 * self.alpha * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)) / n
            self.loss_curve_.append(float(loss))
            # same early-stopping rule as sklearn: stop after `patience` epochs without tol gain
            if loss > best - self.tol:
                stall += 1
                if stall >= self.patience:
                    break
            else:
                stall = 0
            best = min(best, loss)
        return self

    def predict_proba(self, X):
        return self._forward(check_X(X))[1]
