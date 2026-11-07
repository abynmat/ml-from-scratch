"""Failure-mode study: six controlled experiments where a textbook implementation breaks.

Author: Abyn Scaria

Each experiment changes one thing, measures what goes wrong, and saves a figure in figures/ and
its raw numbers in results/failure_modes.json.

    python3 experiments/failure_modes.py
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", message=".*encountered in.*")  # GD divergence is intentional

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import mlfs  # noqa: E402
from mlfs.datasets import load  # noqa: E402

FIG = os.path.join(ROOT, "figures")
OUT = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Gradient descent on unscaled vs standardised features
# ---------------------------------------------------------------------------
def gd_scaling():
    raw = load("california", scale=False)
    std = load("california", scale=True)
    exact = mlfs.LinearRegression().fit(raw.X_train, raw.y_train)
    floor = mlfs.mse(raw.y_train, exact.predict(raw.X_train))

    def lam_max(X):
        Xc = X - X.mean(0)
        ev = np.linalg.eigvalsh(Xc.T @ Xc / len(X))
        return ev.max(), ev.max() / ev.min()

    lr_raw, cond_raw = lam_max(raw.X_train)
    lr_std, cond_std = lam_max(std.X_train)
    runs = {
        "standardised, lr=0.1": (std, 0.1),
        "raw, lr=0.1": (raw, 0.1),
        "raw, lr=0.9/lambda_max": (raw, 0.9 / lr_raw),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    res = {"cond_raw": float(cond_raw), "cond_std": float(cond_std), "ols_train_mse": floor}
    for label, (d, lr) in runs.items():
        m = mlfs.LinearRegression("gd", lr=lr, n_iter=5000, tol=0).fit(d.X_train, d.y_train)
        curve = np.array(m.loss_curve_)
        finite = np.isfinite(curve) & (curve < 1e6)
        ax.semilogy(np.where(finite)[0], curve[finite] - floor + 1e-12, label=label)
        res[label] = dict(lr=float(lr), steps=len(curve), final_excess_mse=float(curve[-1] - floor),
                          diverged=bool(not np.isfinite(curve[-1]) or curve[-1] > 1e6))
        if res[label]["diverged"]:
            ax.annotate(f"{label}: diverges (loss > 1e6 after {int(finite.sum())} steps)",
                        xy=(0.25, 0.55), xycoords="axes fraction", fontsize=8, color="C1")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("train MSE minus OLS optimum (log)")
    ax.set_title(f"GD on California housing: condition number {cond_raw:,.0f} (raw) "
                 f"vs {cond_std:.0f} (standardised)", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "fail_gd_scaling.png")
    OUT["gd_scaling"] = res


# ---------------------------------------------------------------------------
# 2. SMO iterations as C grows (and on unscaled data)
# ---------------------------------------------------------------------------
def smo_vs_C():
    std = load("breast_cancer", scale=True)
    raw = load("breast_cancer", scale=False)
    Cs = np.logspace(-2, 3, 11)
    rows = []
    for kernel in ("linear", "rbf"):
        for scaled, d in (("standardised", std), ("raw", raw)):
            if kernel == "linear" and scaled == "raw":
                continue  # raw linear kernel values ~1e6: hits the iteration cap for every C
            for C in Cs:
                t0 = time.perf_counter()
                m = mlfs.SVC(C=C, kernel=kernel, max_iter=200_000).fit(d.X_train, d.y_train)
                rows.append(dict(kernel=kernel, data=scaled, C=float(C), iters=m.n_iter_,
                                 converged=bool(m.converged_), n_sv=int(len(m.support_)),
                                 acc=m.score(d.X_test, d.y_test), sec=time.perf_counter() - t0))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for (kernel, scaled), mk in ((("linear", "standardised"), "o-"), (("rbf", "standardised"), "s-"),
                                 (("rbf", "raw"), "^--")):
        r = [x for x in rows if x["kernel"] == kernel and x["data"] == scaled]
        a1.loglog([x["C"] for x in r], [x["iters"] for x in r], mk, label=f"{kernel}, {scaled}")
        a2.semilogx([x["C"] for x in r], [x["acc"] for x in r], mk, label=f"{kernel}, {scaled}")
    a1.set_xlabel("C")
    a1.set_ylabel("SMO pair updates until KKT gap < 1e-3")
    a1.set_title("SMO work grows with C (fewer multipliers stay at the bound)", fontsize=9)
    a1.legend(fontsize=8)
    a2.set_xlabel("C")
    a2.set_ylabel("test accuracy")
    a2.set_title("...while accuracy saturates or drops", fontsize=9)
    save(fig, "fail_smo_C.png")
    OUT["smo_vs_C"] = rows


# ---------------------------------------------------------------------------
# 3. kNN as irrelevant dimensions are added
# ---------------------------------------------------------------------------
def knn_dimension():
    X2, y = make_moons(1000, noise=0.2, random_state=0)
    rng = np.random.default_rng(0)
    dims = [0, 2, 5, 10, 20, 50, 100, 200, 500]
    rows = []
    for extra in dims:
        X = np.hstack([X2, rng.normal(0, 1, (len(X2), extra))]) if extra else X2
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        knn = mlfs.KNNClassifier(15).fit(Xtr, ytr)
        tree = mlfs.DecisionTreeClassifier(min_samples_leaf=5).fit(Xtr, ytr)
        D = np.sqrt(mlfs.knn.pairwise_sq_dists(Xte[:200], Xtr))
        contrast = float(np.mean((D.max(1) - D.min(1)) / D.min(1)))
        rows.append(dict(noise_dims=extra, total_dims=2 + extra, knn_acc=knn.score(Xte, yte),
                         tree_acc=tree.score(Xte, yte), relative_contrast=contrast))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    td = [r["total_dims"] for r in rows]
    a1.semilogx(td, [r["knn_acc"] for r in rows], "o-", label="kNN (k=15)")
    a1.semilogx(td, [r["tree_acc"] for r in rows], "s-", label="CART (min_leaf=5)")
    a1.axhline(0.5, color="gray", ls=":", lw=1)
    a1.set_xlabel("dimensions (2 informative + Gaussian noise)")
    a1.set_ylabel("test accuracy")
    a1.set_title("kNN degrades towards chance; a tree can ignore noise features", fontsize=9)
    a1.legend(fontsize=8)
    a2.loglog(td, [r["relative_contrast"] for r in rows], "o-", color="C2")
    a2.set_xlabel("dimensions")
    a2.set_ylabel("(d_max - d_min) / d_min")
    a2.set_title("Distances concentrate: nearest is barely nearer than farthest", fontsize=9)
    save(fig, "fail_knn_dimension.png")
    OUT["knn_dimension"] = rows


# ---------------------------------------------------------------------------
# 4. k-means: uniform-random seeding vs k-means++
# ---------------------------------------------------------------------------
def kmeans_init():
    X, _ = make_blobs(1500, centers=15, cluster_std=0.5, center_box=(-20, 20), random_state=3)
    seeds = range(100)
    res = {}
    for init in ("random", "k-means++"):
        res[init] = [mlfs.KMeans(15, init=init, n_init=1, random_state=s).fit(X).inertia_ for s in seeds]
    best = min(min(v) for v in res.values())
    summary = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), best=float(best),
                       frac_within_1pct_of_best=float(np.mean(np.array(v) <= best * 1.01)),
                       frac_within_25pct_of_best=float(np.mean(np.array(v) <= best * 1.25)))
               for k, v in res.items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(best * 0.98, max(max(v) for v in res.values()) * 1.02, 40)
    for k, v in res.items():
        a1.hist(v, bins=bins, alpha=0.6, label=f"{k}: median {summary[k]['median']:.0f}, "
                                              f"{100 * summary[k]['frac_within_25pct_of_best']:.0f}% within 25% of best")
    a1.set_xlabel("final inertia (100 single-restart runs)")
    a1.set_ylabel("runs")
    a1.set_title("15 well-separated blobs, n_init=1", fontsize=9)
    a1.legend(fontsize=8)
    worst = int(np.argmax(res["random"]))
    m = mlfs.KMeans(15, init="random", n_init=1, random_state=worst).fit(X)
    a2.scatter(X[:, 0], X[:, 1], c=m.labels_, s=4, cmap="tab20")
    a2.scatter(*m.cluster_centers_.T, c="k", marker="x", s=60)
    a2.set_title(f"Worst random-init run: centres pile up in some blobs, others merge",
                 fontsize=9)
    save(fig, "fail_kmeans_init.png")
    OUT["kmeans_init"] = summary


# ---------------------------------------------------------------------------
# 5. Naive Bayes with correlated (duplicated) features
# ---------------------------------------------------------------------------
def naive_bayes_correlated():
    d = load("breast_cancer")
    rows = []
    for r in (1, 2, 3, 5, 8, 12):
        Xtr, Xte = np.tile(d.X_train, r), np.tile(d.X_test, r)  # each feature repeated r times
        nb = mlfs.GaussianNB().fit(Xtr, d.y_train)
        lr = mlfs.LogisticRegression(C=1.0).fit(Xtr, d.y_train)
        p_nb, p_lr = nb.predict_proba(Xte), lr.predict_proba(Xte)
        rows.append(dict(copies=r, nb_acc=nb.score(Xte, d.y_test), nb_logloss=mlfs.log_loss(d.y_test, p_nb),
                         nb_frac_conf_gt_0999=float(np.mean(p_nb.max(1) > 0.999)),
                         lr_acc=lr.score(Xte, d.y_test), lr_logloss=mlfs.log_loss(d.y_test, p_lr)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    c = [x["copies"] for x in rows]
    a1.plot(c, [x["nb_logloss"] for x in rows], "o-", label="Gaussian NB")
    a1.plot(c, [x["lr_logloss"] for x in rows], "s-", label="logistic regression")
    a1.set_xlabel("copies of every feature")
    a1.set_ylabel("test log-loss")
    a1.set_title("NB counts the same evidence r times -> overconfident", fontsize=9)
    a1.legend(fontsize=8)
    a2.plot(c, [x["nb_frac_conf_gt_0999"] for x in rows], "o-", color="C3")
    a2.set_xlabel("copies of every feature")
    a2.set_ylabel("fraction of test points with max p > 0.999")
    a2.set_title("Accuracy barely moves; calibration collapses", fontsize=9)
    save(fig, "fail_naive_bayes_correlated.png")
    OUT["naive_bayes_correlated"] = rows


# ---------------------------------------------------------------------------
# 6. Decision tree overfitting vs min_samples_leaf
# ---------------------------------------------------------------------------
def tree_overfit():
    d = load("california")
    rows = []
    for leaf in (1, 2, 5, 10, 20, 50, 100, 200, 400):
        m = mlfs.DecisionTreeRegressor(min_samples_leaf=leaf).fit(d.X_train, d.y_train)
        rows.append(dict(min_samples_leaf=leaf, train_r2=m.score(d.X_train, d.y_train),
                         test_r2=m.score(d.X_test, d.y_test), leaves=m.n_leaves_, depth=m.depth()))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    L = [r["min_samples_leaf"] for r in rows]
    ax.semilogx(L, [r["train_r2"] for r in rows], "o-", label="train R2")
    ax.semilogx(L, [r["test_r2"] for r in rows], "s-", label="test R2")
    for r in rows[::2]:
        ax.annotate(f"{r['leaves']} leaves", (r["min_samples_leaf"], r["test_r2"]), fontsize=7,
                    textcoords="offset points", xytext=(0, -12), ha="center")
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("R2")
    ax.set_title("CART on California housing: the unpruned tree memorises the training set", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "fail_tree_overfit.png")
    OUT["tree_overfit"] = rows


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for fn in (gd_scaling, smo_vs_C, knn_dimension, kmeans_init, naive_bayes_correlated, tree_overfit):
        t0 = time.time()
        fn()
        print(f"{fn.__name__:26s} done in {time.time() - t0:5.1f}s")
    with open(os.path.join(ROOT, "results", "failure_modes.json"), "w") as f:
        json.dump(OUT, f, indent=2)
    print(json.dumps(OUT, indent=1)[:6000])
