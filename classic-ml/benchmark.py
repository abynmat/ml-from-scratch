"""Parity benchmark: every mlfs estimator vs its scikit-learn counterpart on the same split.

Author: Abyn Scaria

For each row we fit both libraries with the same hyperparameters on the same train split and
report: the task metric on the test split for each, the agreement between the two models'
predictions, and fit wall-clock (best of up to 3 repeats for fits under one second).

    python3 benchmark.py          -> results/parity.csv, figures/parity.png
"""
import csv
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy 2 + Accelerate noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Maximum iterations.*")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import (cluster, decomposition, ensemble, linear_model, mixture, naive_bayes,  # noqa: E402
                     neighbors, neural_network, svm, tree)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import mlfs  # noqa: E402
from mlfs.datasets import load  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
HERE = os.path.dirname(os.path.abspath(__file__))


def timed_fit(make, X, y):
    """Fit a fresh model; repeat fast fits and keep the minimum time to reduce noise."""
    best, model = np.inf, None
    for rep in range(3):
        m = make()
        t0 = time.perf_counter()
        m.fit(X, y) if y is not None else m.fit(X)
        dt = time.perf_counter() - t0
        if dt < best:
            best, model = dt, m
        if dt > 1.0:
            break
    return model, best


# (algorithm label, dataset, metric, mlfs factory, sklearn factory, hyperparameter note)
CASES = [
    ("OLS (closed form)", "california", "R2", lambda: mlfs.LinearRegression(),
     lambda: linear_model.LinearRegression(), "-"),
    ("OLS (gradient descent)", "california", "R2", lambda: mlfs.LinearRegression("gd", lr=0.1, n_iter=5000),
     lambda: linear_model.LinearRegression(), "lr=0.1, 5000 steps; sklearn has no batch GD, ref = exact OLS"),
    ("Ridge (coordinate descent)", "diabetes", "R2", lambda: mlfs.Ridge(1.0, solver="cd"),
     lambda: linear_model.Ridge(1.0), "alpha=1"),
    ("Lasso (coordinate descent)", "diabetes", "R2", lambda: mlfs.Lasso(0.1),
     lambda: linear_model.Lasso(0.1), "alpha=0.1"),
    ("Logistic (Newton/IRLS)", "breast_cancer", "accuracy", lambda: mlfs.LogisticRegression(1.0, "newton"),
     lambda: linear_model.LogisticRegression(C=1.0), "C=1; sklearn: lbfgs"),
    ("Logistic (gradient descent)", "breast_cancer", "accuracy",
     lambda: mlfs.LogisticRegression(1.0, "gd", n_iter=5000), lambda: linear_model.LogisticRegression(C=1.0),
     "C=1, lr=0.5"),
    ("Logistic multinomial (GD)", "wine", "accuracy", lambda: mlfs.LogisticRegression(1.0, "gd", n_iter=3000),
     lambda: linear_model.LogisticRegression(C=1.0), "C=1, 3 classes"),
    ("kNN classifier", "digits", "accuracy", lambda: mlfs.KNNClassifier(5),
     lambda: neighbors.KNeighborsClassifier(5), "k=5"),
    ("kNN regressor", "california", "R2", lambda: mlfs.KNNRegressor(10),
     lambda: neighbors.KNeighborsRegressor(10), "k=10"),
    ("Gaussian NB", "wine", "accuracy", lambda: mlfs.GaussianNB(), lambda: naive_bayes.GaussianNB(), "-"),
    ("Multinomial NB", "digits_raw", "accuracy", lambda: mlfs.MultinomialNB(1.0),
     lambda: naive_bayes.MultinomialNB(alpha=1.0), "alpha=1, raw pixel counts"),
    ("CART classifier (gini)", "breast_cancer", "accuracy", lambda: mlfs.DecisionTreeClassifier("gini"),
     lambda: tree.DecisionTreeClassifier(criterion="gini", random_state=0), "unpruned"),
    ("CART classifier (entropy)", "wine", "accuracy", lambda: mlfs.DecisionTreeClassifier("entropy", max_depth=4),
     lambda: tree.DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0), "max_depth=4"),
    ("CART regressor (MSE)", "california", "R2", lambda: mlfs.DecisionTreeRegressor(max_depth=8),
     lambda: tree.DecisionTreeRegressor(max_depth=8, random_state=0), "max_depth=8"),
    ("Random forest", "breast_cancer", "accuracy", lambda: mlfs.RandomForestClassifier(100, random_state=0),
     lambda: ensemble.RandomForestClassifier(100, random_state=0, n_jobs=1), "100 trees, sqrt features"),
    ("Gradient boosting regressor", "california", "R2",
     lambda: mlfs.GradientBoostingRegressor(100, 0.1, 3, random_state=0),
     lambda: ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                criterion="squared_error", random_state=0),
     "100 x depth 3, lr=0.1"),
    ("Gradient boosting classifier", "breast_cancer", "accuracy",
     lambda: mlfs.GradientBoostingClassifier(100, 0.1, 3, random_state=0),
     lambda: ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                 criterion="squared_error", random_state=0),
     "100 x depth 3, Newton leaves"),
    ("SVM linear (SMO)", "breast_cancer", "accuracy", lambda: mlfs.SVC(1.0, "linear"),
     lambda: svm.SVC(C=1.0, kernel="linear"), "C=1"),
    ("SVM RBF (SMO)", "breast_cancer", "accuracy", lambda: mlfs.SVC(1.0, "rbf"),
     lambda: svm.SVC(C=1.0, kernel="rbf"), "C=1, gamma=scale"),
    ("Perceptron", "breast_cancer", "accuracy", lambda: mlfs.Perceptron(50, random_state=0),
     lambda: linear_model.Perceptron(max_iter=50, random_state=0), "50 epochs"),
    ("MLP (1 hidden layer, Adam)", "digits", "accuracy", lambda: mlfs.MLPClassifier(64, random_state=0),
     lambda: neural_network.MLPClassifier((64,), max_iter=200, random_state=0), "64 ReLU, alpha=1e-4"),
    ("k-means (++ init)", "digits", "ARI vs labels", lambda: mlfs.KMeans(10, n_init=10, random_state=0),
     lambda: cluster.KMeans(10, n_init=10, random_state=0), "k=10, 10 restarts"),
    ("Gaussian mixture (EM)", "iris", "test log-lik / sample", lambda: mlfs.GaussianMixture(3, random_state=0),
     lambda: mixture.GaussianMixture(3, random_state=0), "3 full-cov components"),
    ("PCA (SVD)", "digits", "explained var (10 PCs)", lambda: mlfs.PCA(10),
     lambda: decomposition.PCA(10), "10 components"),
]


def get_data(name):
    if name == "digits_raw":
        return load("digits", scale=False)
    return load(name)


def evaluate(label, metric, m, s, data):
    """Return (metric_mlfs, metric_sklearn, agreement)."""
    Xte, yte = data.X_test, data.y_test
    if metric == "R2":
        pm, ps = m.predict(Xte), s.predict(Xte)
        # agreement for regressors: 1 - mean|diff| / std(y)
        return mlfs.r2(yte, pm), mlfs.r2(yte, ps), 1 - np.mean(np.abs(pm - ps)) / np.std(yte)
    if metric == "accuracy":
        pm, ps = m.predict(Xte), s.predict(Xte)
        return mlfs.accuracy(yte, pm), mlfs.accuracy(yte, ps), mlfs.accuracy(pm, ps)
    if metric == "ARI vs labels":
        lm, ls = m.predict(Xte), s.predict(Xte)
        return (mlfs.adjusted_rand_index(yte, lm), mlfs.adjusted_rand_index(yte, ls),
                mlfs.adjusted_rand_index(lm, ls))
    if metric == "test log-lik / sample":
        return m.score(Xte), s.score(Xte), mlfs.adjusted_rand_index(m.predict(Xte), s.predict(Xte))
    if metric.startswith("explained var"):
        cos = np.abs(np.sum(m.components_ * s.components_, axis=1)).min()  # 1 = same axes
        return float(m.explained_variance_ratio_.sum()), float(s.explained_variance_ratio_.sum()), float(cos)
    raise ValueError(metric)


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    cache, rows = {}, []
    for label, dname, metric, make_m, make_s, note in CASES:
        data = cache.setdefault(dname, get_data(dname))
        sup = metric in ("R2", "accuracy")
        y = data.y_train if sup else None
        m, tm = timed_fit(make_m, data.X_train, y)
        s, ts = timed_fit(make_s, data.X_train, y)
        vm, vs, agree = evaluate(label, metric, m, s, data)
        rows.append(dict(algorithm=label, dataset=dname, n_train=len(data.X_train), metric=metric,
                         mlfs=round(vm, 4), sklearn=round(vs, 4), diff=round(vm - vs, 4),
                         agreement=round(agree, 4), mlfs_fit_s=round(tm, 4), sklearn_fit_s=round(ts, 4),
                         slowdown=round(tm / max(ts, 1e-6), 2), hyperparameters=note))
        print(f"{label:32s} {dname:14s} {metric:22s} mlfs={vm:8.4f} sk={vs:8.4f} agree={agree:6.3f} "
              f"t={tm:7.3f}s / {ts:7.3f}s")
    path = os.path.join(HERE, "results", "parity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)
    plot(rows)


def plot(rows):
    unit = [r for r in rows if 0 <= r["sklearn"] <= 1.0001]  # metrics on a 0-1 scale
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 0.38 * len(unit) + 1.5),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    yy = np.arange(len(unit))
    a1.barh(yy - 0.2, [r["mlfs"] for r in unit], 0.4, label="mlfs (this repo)", color="#3b75af")
    a1.barh(yy + 0.2, [r["sklearn"] for r in unit], 0.4, label="scikit-learn", color="#ef8636")
    a1.set_yticks(yy)
    a1.set_yticklabels([f"{r['algorithm']}  [{r['dataset']}, {r['metric']}]" for r in unit], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0, 1.05)
    a1.set_xlabel("test metric (higher is better)")
    a1.legend(loc="lower right", fontsize=8)
    a1.set_title("Test metric on identical splits")
    a2.barh(yy, [max(r["slowdown"], 0.05) for r in unit], color="#6a9f58")  # clip so tiny ratios stay visible
    a2.axvline(1, color="k", lw=0.8)
    a2.set_xscale("log")
    a2.set_yticks(yy)
    a2.set_yticklabels([])
    a2.invert_yaxis()
    a2.set_xlabel("fit time mlfs / sklearn (log scale, >1 = mlfs slower)")
    a2.set_title("Fit wall-clock ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "parity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
