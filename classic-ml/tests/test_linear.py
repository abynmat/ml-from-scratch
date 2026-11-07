"""Linear, ridge, lasso and logistic regression vs scikit-learn."""
import _harness  # noqa: F401  (sets sys.path)
from _harness import close

import numpy as np
from sklearn import linear_model as sk

import mlfs
from mlfs.datasets import load

D = load("diabetes")
B = load("breast_cancer")


def test_ols_closed_form_matches_coefficients():
    m = mlfs.LinearRegression().fit(D.X_train, D.y_train)
    s = sk.LinearRegression().fit(D.X_train, D.y_train)
    close(m.coef_, s.coef_, 1e-8, "coef")
    close(m.intercept_, s.intercept_, 1e-8, "intercept")


def test_ols_gradient_descent_reaches_closed_form_predictions():
    m = mlfs.LinearRegression(solver="gd", lr=0.1, n_iter=20000).fit(D.X_train, D.y_train)
    s = sk.LinearRegression().fit(D.X_train, D.y_train)
    # coefficients of correlated features converge slowly; predictions converge fast
    close(m.score(D.X_test, D.y_test), s.score(D.X_test, D.y_test), 1e-3, "test R2")


def test_ridge_closed_and_cd_match_sklearn():
    s = sk.Ridge(alpha=3.0).fit(D.X_train, D.y_train)
    for solver in ("closed", "cd"):
        m = mlfs.Ridge(alpha=3.0, solver=solver, tol=1e-12, max_iter=100000).fit(D.X_train, D.y_train)
        close(m.coef_, s.coef_, 1e-6, f"ridge {solver} coef")


def test_lasso_matches_sklearn_and_is_sparse():
    m = mlfs.Lasso(alpha=0.5, tol=1e-12, max_iter=100000).fit(D.X_train, D.y_train)
    s = sk.Lasso(alpha=0.5, tol=1e-12, max_iter=100000).fit(D.X_train, D.y_train)
    close(m.coef_, s.coef_, 1e-6, "lasso coef")
    assert np.sum(m.coef_ == 0) == np.sum(s.coef_ == 0) > 0, "same number of exact zeros"


def test_logistic_newton_matches_lbfgs():
    m = mlfs.LogisticRegression(C=1.0, solver="newton").fit(B.X_train, B.y_train)
    s = sk.LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(B.X_train, B.y_train)
    close(m._w[:-1], s.coef_[0], 1e-4, "logistic coef")
    close(m.predict_proba(B.X_test), s.predict_proba(B.X_test), 1e-5, "probabilities")


def test_logistic_gd_agrees_on_labels():
    m = mlfs.LogisticRegression(solver="gd", n_iter=5000, lr=0.5).fit(B.X_train, B.y_train)
    s = sk.LogisticRegression().fit(B.X_train, B.y_train)
    assert np.mean(m.predict(B.X_test) == s.predict(B.X_test)) >= 0.99


def test_multinomial_logistic_on_wine():
    W = load("wine")
    m = mlfs.LogisticRegression(solver="gd", n_iter=3000).fit(W.X_train, W.y_train)
    s = sk.LogisticRegression().fit(W.X_train, W.y_train)
    assert abs(m.score(W.X_test, W.y_test) - s.score(W.X_test, W.y_test)) <= 0.03


if __name__ == "__main__":
    raise SystemExit(_harness.run(globals()))
