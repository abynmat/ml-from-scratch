"""CART, random forest and gradient boosting vs scikit-learn."""
import _harness  # noqa: F401
from _harness import close

import numpy as np
from sklearn import ensemble as ske
from sklearn import tree as skt

import mlfs
from mlfs.datasets import load

B = load("breast_cancer")
D = load("diabetes")


def test_unpruned_tree_fits_training_set_perfectly():
    m = mlfs.DecisionTreeClassifier().fit(B.X_train, B.y_train)
    assert m.score(B.X_train, B.y_train) == 1.0


def test_tree_structure_matches_sklearn_when_no_ties():
    # with all features considered there is no randomness; the same greedy CART should give
    # the same number of leaves and depth (exact thresholds can differ only on tied gains)
    for crit in ("gini", "entropy"):
        m = mlfs.DecisionTreeClassifier(criterion=crit, max_depth=4).fit(B.X_train, B.y_train)
        s = skt.DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=0).fit(B.X_train, B.y_train)
        assert m.depth() == s.get_depth(), crit
        assert abs(m.n_leaves_ - s.get_n_leaves()) <= 1, crit
        assert abs(m.score(B.X_test, B.y_test) - s.score(B.X_test, B.y_test)) <= 0.03, crit


def test_regression_tree_root_split_matches():
    m = mlfs.DecisionTreeRegressor(max_depth=1).fit(D.X_train, D.y_train)
    s = skt.DecisionTreeRegressor(max_depth=1).fit(D.X_train, D.y_train)
    assert m.feature_[0] == s.tree_.feature[0]
    # sklearn casts X to float32 inside trees, so its thresholds carry float32 rounding
    close(m.threshold_[0], s.tree_.threshold[0], 1e-6, "root threshold")
    close(m.predict(D.X_test), s.predict(D.X_test), 1e-9, "stump predictions")


def test_min_samples_leaf_is_respected():
    m = mlfs.DecisionTreeRegressor(min_samples_leaf=15).fit(D.X_train, D.y_train)
    leaves = m.feature_ == -1
    assert m.n_node_samples_[leaves].min() >= 15


def test_random_forest_close_to_sklearn():
    m = mlfs.RandomForestClassifier(n_estimators=60, random_state=0).fit(B.X_train, B.y_train)
    s = ske.RandomForestClassifier(n_estimators=60, random_state=0).fit(B.X_train, B.y_train)
    assert abs(m.score(B.X_test, B.y_test) - s.score(B.X_test, B.y_test)) <= 0.03


def test_gradient_boosting_regressor_close_to_sklearn():
    m = mlfs.GradientBoostingRegressor(n_estimators=100, max_depth=2).fit(D.X_train, D.y_train)
    s = ske.GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0).fit(D.X_train, D.y_train)
    assert abs(m.score(D.X_test, D.y_test) - s.score(D.X_test, D.y_test)) <= 0.03


def test_gradient_boosting_classifier_newton_leaves():
    m = mlfs.GradientBoostingClassifier(n_estimators=100).fit(B.X_train, B.y_train)
    s = ske.GradientBoostingClassifier(n_estimators=100, random_state=0).fit(B.X_train, B.y_train)
    # same init (log-odds of the base rate) and same Newton leaf formula
    p0 = s.init_.predict_proba(B.X_train[:1])[0, 1]  # sklearn's prior model
    close(m.init_, np.log(p0 / (1 - p0)), 1e-12, "log-odds init")
    assert np.mean(m.predict(B.X_test) == s.predict(B.X_test)) >= 0.97
    # training log-loss must decrease monotonically-ish for a convex loss with small steps
    losses = [mlfs.log_loss(B.y_train, np.column_stack([1 - p, p]))
              for p in (1 / (1 + np.exp(-F)) for F in m.staged_raw(B.X_train))]
    assert losses[-1] < losses[0] * 0.2


if __name__ == "__main__":
    raise SystemExit(_harness.run(globals()))
