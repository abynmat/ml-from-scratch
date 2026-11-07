"""k-means clustering (Lloyd's algorithm) with k-means++ or uniform-random seeding.

Author: Abyn Scaria
References: Lloyd (1982) "Least squares quantization in PCM"; Arthur & Vassilvitskii (2007)
"k-means++: The Advantages of Careful Seeding", SODA. ESL section 14.3.6.
"""
import numpy as np

from .base import Estimator, as_rng, check_X
from .knn import pairwise_sq_dists


def kmeans_plusplus(X, k, rng):
    """D^2 seeding: the first centre is a random point, each next centre is drawn with probability
    proportional to its squared distance from the nearest centre already chosen."""
    n = len(X)
    centres = [X[rng.integers(n)]]
    d2 = pairwise_sq_dists(X, centres[0][None])[:, 0]
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        c = X[rng.choice(n, p=probs)]
        centres.append(c)
        d2 = np.minimum(d2, pairwise_sq_dists(X, c[None])[:, 0])
    return np.array(centres)


class KMeans(Estimator):
    """init: "k-means++" or "random" (k distinct data points chosen uniformly).
    The best of `n_init` restarts (lowest inertia) is kept."""

    _kind = "cluster"

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_X(X)
        rng = as_rng(self.random_state)
        # sklearn scales the tolerance by the mean feature variance; do the same
        tol = self.tol * np.mean(X.var(axis=0))
        best = None
        for _ in range(self.n_init):
            if self.init == "k-means++":
                C = kmeans_plusplus(X, self.n_clusters, rng)
            elif self.init == "random":
                C = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
            else:
                raise ValueError("init must be 'k-means++' or 'random'")
            C, labels, inertia, n_iter = _lloyd(X, C, self.max_iter, tol)
            if best is None or inertia < best[2]:
                best = (C, labels, inertia, n_iter)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        return np.argmin(pairwise_sq_dists(check_X(X), self.cluster_centers_), axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


def _lloyd(X, C, max_iter, tol):
    """Alternate (1) assign each point to its nearest centre, (2) move centres to cluster means."""
    k = len(C)
    for it in range(1, max_iter + 1):
        D = pairwise_sq_dists(X, C)
        labels = np.argmin(D, axis=1)
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)
        newC = C.copy()
        nonempty = counts > 0
        newC[nonempty] = sums[nonempty] / counts[nonempty, None]
        # an empty cluster is re-seeded at the point currently farthest from its centre
        for c in np.where(~nonempty)[0]:
            far = int(np.argmax(D[np.arange(len(X)), labels]))
            newC[c] = X[far]
        shift = np.sum((newC - C) ** 2)
        C = newC
        if shift <= tol:
            break
    D = pairwise_sq_dists(X, C)
    labels = np.argmin(D, axis=1)
    inertia = float(D[np.arange(len(X)), labels].sum())
    return C, labels, inertia, it
