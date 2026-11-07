"""Complexity study: fit (or query) time vs n on log-log axes, compared with the textbook order.

Author: Abyn Scaria

For each algorithm we time mlfs and scikit-learn on synthetic data (d = 20 features), fit a line
to log(time) vs log(n) over the larger half of the grid (small n is dominated by Python overhead),
and compare the slope with the theoretical exponent.

    python3 experiments/timing.py   -> results/timing.csv, figures/timing.png
"""
import csv
import os
import sys
import time
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
from sklearn import linear_model, neighbors, svm, tree  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402

import mlfs  # noqa: E402

D = 20


def data(n, seed=0):
    X, y = make_classification(n, D, n_informative=10, random_state=seed)
    return X, y


def t_fit(make, X, y, reps=3):
    best = np.inf
    for _ in range(reps):
        m = make()
        t0 = time.perf_counter()
        m.fit(X, y)
        best = min(best, time.perf_counter() - t0)
        if best > 1.0:
            break
    return best


def t_query(make, X, y, reps=3):
    """kNN: 'fit' is storing the data; the cost is answering n queries against n points."""
    best = np.inf
    m = make().fit(X, y)
    for _ in range(reps):
        t0 = time.perf_counter()
        m.predict(X)
        best = min(best, time.perf_counter() - t0)
        if best > 1.0:
            break
    return best


STUDIES = [
    # name, theory string, theoretical exponent, ns, mlfs factory, sklearn factory, timer
    ("OLS closed form (fit)", "O(n d^2)", 1.0, [1000, 2000, 4000, 8000, 16000, 32000, 64000],
     lambda: mlfs.LinearRegression(), lambda: linear_model.LinearRegression(), t_fit),
    ("CART gini (fit)", "O(d n log n) per level", 1.0, [500, 1000, 2000, 4000, 8000, 16000],
     lambda: mlfs.DecisionTreeClassifier(max_depth=8), lambda: tree.DecisionTreeClassifier(max_depth=8), t_fit),
    ("kNN (n queries vs n points)", "O(n^2 d)", 2.0, [500, 1000, 2000, 4000, 8000, 16000],
     lambda: mlfs.KNNClassifier(5), lambda: neighbors.KNeighborsClassifier(5, algorithm="brute"), t_query),
    ("SVM RBF via SMO (fit)", "O(n^2) to O(n^3)", 2.0, [250, 500, 1000, 2000, 4000],
     lambda: mlfs.SVC(1.0, "rbf"), lambda: svm.SVC(C=1.0, kernel="rbf", cache_size=1000), t_fit),
]


def slope(ns, ts):
    k = len(ns) // 2
    return float(np.polyfit(np.log(ns[k:]), np.log(ts[k:]), 1)[0])


def main():
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (name, theory, p, ns, mk_m, mk_s, timer) in zip(axes, STUDIES):
        tm, ts = [], []
        for n in ns:
            X, y = data(n)
            tm.append(timer(mk_m, X, y))
            ts.append(timer(mk_s, X, y))
            print(f"{name:30s} n={n:6d}  mlfs={tm[-1]:8.4f}s  sklearn={ts[-1]:8.4f}s", flush=True)
            rows.append(dict(study=name, n=n, mlfs_s=round(tm[-1], 5), sklearn_s=round(ts[-1], 5)))
        sm, ss = slope(ns, tm), slope(ns, ts)
        rows.append(dict(study=name, n="slope", mlfs_s=round(sm, 2), sklearn_s=round(ss, 2)))
        ax.loglog(ns, tm, "o-", label=f"mlfs (slope {sm:.2f})")
        ax.loglog(ns, ts, "s--", label=f"sklearn (slope {ss:.2f})")
        ref = tm[-1] * (np.array(ns, float) / ns[-1]) ** p
        ax.loglog(ns, ref, ":", color="gray", label=f"n^{p:g} reference")
        ax.set_title(f"{name}\ntheory: {theory}", fontsize=9)
        ax.set_xlabel("n")
        ax.set_ylabel("seconds")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "timing.png"), dpi=150)
    with open(os.path.join(ROOT, "results", "timing.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
