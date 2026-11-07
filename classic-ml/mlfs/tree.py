"""CART decision trees (Breiman, Friedman, Olshen & Stone, 1984) for classification and regression.

Author: Abyn Scaria
Reference: ESL section 9.2.

Split search: for every candidate feature we sort the node's samples once, then compute the
impurity of *every* threshold with cumulative sums (class counts for gini/entropy, sum and sum of
squares for MSE). That makes one node O(d * n log n) instead of the naive O(d * n^2).

The fitted tree is stored as flat arrays (feature, threshold, left, right, value), and prediction
walks all samples down the tree at once, one level per loop iteration.
"""
import numpy as np

from .base import Classifier, Regressor, as_rng, check_X, check_X_y

LEAF = -1


def _gini(counts, n):
    p = counts / n[:, None]
    return 1.0 - np.sum(p * p, axis=1)


def _entropy(counts, n):
    p = counts / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -np.sum(p * logp, axis=1)


class _Tree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    # -- building ---------------------------------------------------------
    def _n_features_per_split(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(d, int(mf))

    def _build(self, X, targets):
        """`targets` is a one-hot matrix (classification) or a column of y (regression)."""
        self._rng = as_rng(self.random_state)
        self._mtry = self._n_features_per_split(X.shape[1])
        self.feature_, self.threshold_, self.left_, self.right_, self.value_ = [], [], [], [], []
        self.n_node_samples_ = []
        max_depth = np.inf if self.max_depth is None else self.max_depth
        # explicit stack instead of recursion: (sample indices, depth, parent id, is_left)
        stack = [(np.arange(len(X)), 0, -1, False)]
        while stack:
            idx, depth, parent, is_left = stack.pop()
            node = self._new_node(targets[idx])
            if parent >= 0:
                (self.left_ if is_left else self.right_)[parent] = node
            n = len(idx)
            if depth >= max_depth or n < self.min_samples_split or n < 2 * self.min_samples_leaf:
                continue
            split = self._best_split(X[idx], targets[idx])
            if split is None:
                continue
            f, thr = split
            go_left = X[idx, f] <= thr
            self.feature_[node], self.threshold_[node] = f, thr
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))
        for name in ("feature_", "left_", "right_", "n_node_samples_"):
            setattr(self, name, np.array(getattr(self, name), dtype=np.int64))
        self.threshold_ = np.array(self.threshold_)
        self.value_ = np.array(self.value_)
        del self._rng

    def _new_node(self, t):
        self.feature_.append(LEAF)
        self.threshold_.append(0.0)
        self.left_.append(LEAF)
        self.right_.append(LEAF)
        self.value_.append(t.mean(axis=0))
        self.n_node_samples_.append(len(t))
        return len(self.feature_) - 1

    def _node_impurity_sum(self, t):
        """n * impurity of a node (what a split tries to reduce)."""
        n = np.array([len(t)], float)
        if self.criterion == "mse":
            return float(np.sum((t - t.mean(axis=0)) ** 2))
        fn = _gini if self.criterion == "gini" else _entropy
        return float(n[0] * fn(t.sum(axis=0, keepdims=True), n)[0])

    def _best_split(self, X, t):
        n, d = X.shape
        parent = self._node_impurity_sum(t)
        if parent <= 1e-12:  # pure node
            return None
        feats = self._rng.permutation(d)[:self._mtry] if self._mtry < d else np.arange(d)
        best_score, best = parent - 1e-12, None
        leaf = self.min_samples_leaf
        n_left = np.arange(1, n, dtype=float)  # split after position i -> i+1 samples on the left
        n_right = n - n_left
        for f in feats:
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]
            ts = t[order]
            if self.criterion == "mse":
                cs = np.cumsum(ts[:, 0])[:-1]
                cs2 = np.cumsum(ts[:, 0] ** 2)[:-1]
                tot, tot2 = ts[:, 0].sum(), (ts[:, 0] ** 2).sum()
                # SSE = sum(y^2) - (sum y)^2 / n for each side
                score = (cs2 - cs ** 2 / n_left) + ((tot2 - cs2) - (tot - cs) ** 2 / n_right)
            else:
                left = np.cumsum(ts, axis=0)[:-1]
                right = ts.sum(axis=0) - left
                fn = _gini if self.criterion == "gini" else _entropy
                score = n_left * fn(left, n_left) + n_right * fn(right, n_right)
            valid = xs[:-1] < xs[1:]  # cannot split between equal values
            if leaf > 1:
                valid[: leaf - 1] = False
                valid[n - leaf:] = False
            if not valid.any():
                continue
            score = np.where(valid, score, np.inf)
            i = int(np.argmin(score))
            if score[i] < best_score:
                best_score = score[i]
                best = (int(f), 0.5 * (xs[i] + xs[i + 1]))
        return best

    # -- inference --------------------------------------------------------
    def apply(self, X):
        """Leaf id for each sample."""
        X = check_X(X)
        node = np.zeros(len(X), dtype=np.int64)
        active = self.feature_[node] != LEAF
        while active.any():
            a = np.where(active)[0]
            nd = node[a]
            go_left = X[a, self.feature_[nd]] <= self.threshold_[nd]
            node[a] = np.where(go_left, self.left_[nd], self.right_[nd])
            active[a] = self.feature_[node[a]] != LEAF
        return node

    @property
    def n_leaves_(self):
        return int(np.sum(self.feature_ == LEAF))

    def depth(self):
        depths = np.zeros(len(self.feature_), dtype=int)
        for i in range(len(self.feature_)):
            for c in (self.left_[i], self.right_[i]):
                if c != LEAF:
                    depths[c] = depths[i] + 1
        return int(depths.max())


class DecisionTreeClassifier(_Tree, Classifier):
    """criterion: "gini" or "entropy". Leaves store class proportions."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        codes = self._encode(y)
        self._build(X, np.eye(len(self.classes_))[codes])
        return self

    def predict_proba(self, X):
        return self.value_[self.apply(X)]


class DecisionTreeRegressor(_Tree, Regressor):
    """criterion: "mse". Leaves store the mean target; `value_` may be overwritten (gradient boosting)."""

    def __init__(self, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, max_features,
                         random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.criterion != "mse":
            raise ValueError("regression trees use criterion='mse'")
        self._build(X, y.astype(float)[:, None])
        return self

    def predict(self, X):
        return self.value_[self.apply(X), 0]
