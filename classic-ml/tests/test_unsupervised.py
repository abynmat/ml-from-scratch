"""k-means, Gaussian mixtures, PCA and the metrics vs scikit-learn."""
import _harness  # noqa: F401
from _harness import close

import numpy as np
from sklearn import cluster, decomposition, metrics, mixture

import mlfs
from mlfs.datasets import load

I = load("iris")
G = load("digits")
BL = load("blobs")


def test_kmeans_inertia_within_half_percent():
    for name, data, k in (("iris", I, 3), ("blobs", BL, 4), ("digits", G, 10)):
        m = mlfs.KMeans(k, n_init=10, random_state=0).fit(data.X_train)
        s = cluster.KMeans(k, n_init=10, random_state=0).fit(data.X_train)
        assert m.inertia_ <= s.inertia_ * 1.005, (name, m.inertia_, s.inertia_)


def test_kmeans_plusplus_beats_random_init_on_average():
    # many small, well separated clusters: the regime where uniform seeding often puts two seeds
    # in one cluster and Lloyd's algorithm cannot move one of them out
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(1500, centers=15, cluster_std=0.5, center_box=(-20, 20), random_state=3)
    pp = [mlfs.KMeans(15, init="k-means++", n_init=1, random_state=s).fit(X).inertia_ for s in range(20)]
    rnd = [mlfs.KMeans(15, init="random", n_init=1, random_state=s).fit(X).inertia_ for s in range(20)]
    assert np.mean(pp) < np.mean(rnd), (np.mean(pp), np.mean(rnd))


def test_gmm_log_likelihood_close_to_sklearn():
    m = mlfs.GaussianMixture(3, random_state=0).fit(I.X_train)
    s = mixture.GaussianMixture(3, random_state=0).fit(I.X_train)
    # EM finds local optima; require the same train likelihood within 0.15 nats / sample
    assert abs(m.score(I.X_train) - s.score(I.X_train)) <= 0.15
    lb = np.array(m.lower_bound_curve_)
    assert np.all(np.diff(lb) >= -1e-8), "EM log-likelihood must be non-decreasing"


def test_gmm_responsibilities_sum_to_one():
    m = mlfs.GaussianMixture(4, random_state=1).fit(BL.X_train)
    close(m.predict_proba(BL.X_test).sum(axis=1), 1.0, 1e-10, "resp sum")


def test_pca_components_and_variance():
    m = mlfs.PCA(10).fit(G.X_train)
    s = decomposition.PCA(10).fit(G.X_train)
    close(np.abs(m.components_), np.abs(s.components_), 1e-8, "|components|")
    close(m.explained_variance_ratio_, s.explained_variance_ratio_, 1e-10, "EVR")
    close(np.abs(m.transform(G.X_test)), np.abs(s.transform(G.X_test)), 1e-7, "|scores|")


def test_pca_full_rank_reconstruction_is_exact():
    m = mlfs.PCA().fit(I.X_train)
    close(m.inverse_transform(m.transform(I.X_test)), I.X_test, 1e-10, "reconstruction")


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 4, 300), rng.integers(0, 5, 300)
    close(mlfs.adjusted_rand_index(a, b), metrics.adjusted_rand_score(a, b), 1e-12, "ARI")
    p = rng.dirichlet(np.ones(4), 300)
    close(mlfs.log_loss(a, p), metrics.log_loss(a, p, labels=range(4)), 1e-12, "log loss")
    y, yh = rng.normal(size=100), rng.normal(size=100)
    close(mlfs.r2(y, yh), metrics.r2_score(y, yh), 1e-12, "r2")


if __name__ == "__main__":
    raise SystemExit(_harness.run(globals()))
