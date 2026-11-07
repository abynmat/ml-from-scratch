"""Qualitative figures: decision boundaries on make_moons and unsupervised views of real data.

Author: Abyn Scaria

    python3 experiments/visualize.py  -> figures/decision_boundaries.png, figures/unsupervised.png
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", message=".*encountered in matmul")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

import mlfs  # noqa: E402
from mlfs.datasets import load  # noqa: E402

FIG = os.path.join(ROOT, "figures")


def decision_boundaries():
    d = load("moons")
    models = [
        ("Logistic regression (Newton)", mlfs.LogisticRegression()),
        ("kNN, k=15", mlfs.KNNClassifier(15)),
        ("Gaussian naive Bayes", mlfs.GaussianNB()),
        ("CART, depth 6", mlfs.DecisionTreeClassifier(max_depth=6)),
        ("Random forest, 100 trees", mlfs.RandomForestClassifier(100, min_samples_leaf=3, random_state=0)),
        ("Gradient boosting, 100 x depth 3", mlfs.GradientBoostingClassifier(100, random_state=0)),
        ("SVM RBF (SMO), C=1", mlfs.SVC(1.0, "rbf")),
        ("MLP, 32 tanh units", mlfs.MLPClassifier(32, activation="tanh", lr=0.01, epochs=500, random_state=0)),
    ]
    X = np.vstack([d.X_train, d.X_test])
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 250),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (name, m) in zip(axes.ravel(), models):
        m.fit(d.X_train, d.y_train)
        if isinstance(m, mlfs.SVC):
            z = m.decision_function(grid)
            ax.contourf(xx, yy, (z > 0).reshape(xx.shape), levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.35)
            ax.contour(xx, yy, z.reshape(xx.shape), levels=[-1, 0, 1], colors="k",
                       linestyles=[":", "-", ":"], linewidths=0.8)
        else:
            p = m.predict_proba(grid)[:, 1].reshape(xx.shape)
            ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.45)
            ax.contour(xx, yy, p, levels=[0.5], colors="k", linewidths=0.8)
        ax.scatter(d.X_test[:, 0], d.X_test[:, 1], c=d.y_test, cmap="RdBu", edgecolor="k", s=14, lw=0.4)
        ax.set_title(f"{name}\ntest acc {m.score(d.X_test, d.y_test):.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("make_moons (noise 0.25): all eight models from mlfs, test points shown", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "decision_boundaries.png"), dpi=150)
    plt.close(fig)


def _ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for k in (1, 2):
        ax.add_patch(Ellipse(mean, 2 * k * np.sqrt(vals[-1]), 2 * k * np.sqrt(vals[0]), angle=angle,
                             fill=False, color=color, lw=1.2, alpha=0.9 / k))


def unsupervised():
    dg = load("digits", scale=False)  # raw pixels: standardising near-constant pixels creates outliers
    X = np.vstack([dg.X_train, dg.X_test])
    y = np.concatenate([dg.y_train, dg.y_test])
    pca = mlfs.PCA().fit(X)
    Z = pca.transform(X)[:, :2]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ax = axes[0]
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5)
    ax.set_title("PCA of digits (64-D -> 2-D), coloured by true digit", fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, ticks=range(10))
    ax = axes[1]
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(np.arange(1, len(cum) + 1), cum, "-")
    for t in (0.5, 0.8, 0.95):
        k = int(np.searchsorted(cum, t) + 1)
        ax.axhline(t, color="gray", ls=":", lw=0.8)
        ax.annotate(f"{k} PCs -> {t:.0%}", (k, t), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("Digits: how many components are needed", fontsize=9)

    ir = load("iris")
    Xi = np.vstack([ir.X_train, ir.X_test])
    yi = np.concatenate([ir.y_train, ir.y_test])
    Zi = mlfs.PCA(2).fit_transform(Xi)
    km = mlfs.KMeans(3, random_state=0).fit(Zi)
    ax = axes[2]
    ax.scatter(Zi[:, 0], Zi[:, 1], c=km.labels_, cmap="Set1", s=12)
    ax.scatter(*km.cluster_centers_.T, marker="X", c="k", s=90)
    ax.set_title(f"k-means++ (k=3) on iris PCA plane\nARI vs species "
                 f"{mlfs.adjusted_rand_index(yi, km.labels_):.2f}", fontsize=9)
    gm = mlfs.GaussianMixture(3, n_init=5, random_state=0).fit(Zi)
    lab = gm.predict(Zi)
    ax = axes[3]
    colors = plt.cm.Set1(np.arange(3) / 8)
    ax.scatter(Zi[:, 0], Zi[:, 1], c=colors[lab], s=12)
    for k in range(3):
        _ellipse(ax, gm.means_[k], gm.covariances_[k], colors[k])
    ax.set_title(f"Gaussian mixture (EM, full cov) on the same plane\nARI vs species "
                 f"{mlfs.adjusted_rand_index(yi, lab):.2f} (best of 5 EM restarts)", fontsize=9)
    for a in axes[2:]:
        a.set_xlabel("PC1")
        a.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "unsupervised.png"), dpi=150)
    plt.close(fig)
    gm1 = [mlfs.GaussianMixture(3, n_init=1, random_state=s).fit(Zi) for s in range(5)]
    return dict(gmm_single_restart_aris=[round(mlfs.adjusted_rand_index(yi, g.predict(Zi)), 3) for g in gm1],
                kmeans_ari=mlfs.adjusted_rand_index(yi, km.labels_), gmm_ari=mlfs.adjusted_rand_index(yi, lab),
                digits_pcs_for_80pct=int(np.searchsorted(cum, 0.8) + 1))


if __name__ == "__main__":
    import json
    decision_boundaries()
    stats = unsupervised()
    with open(os.path.join(ROOT, "results", "visualize.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(stats)
