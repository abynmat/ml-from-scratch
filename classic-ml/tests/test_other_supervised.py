"""kNN, naive Bayes, SVM (SMO), perceptron and MLP vs scikit-learn."""
import _harness  # noqa: F401
from _harness import close

import numpy as np
from sklearn import linear_model, naive_bayes, neighbors, neural_network, svm

import mlfs
from mlfs.datasets import load

B = load("breast_cancer")
G = load("digits")
GR = load("digits", scale=False)  # raw pixel counts for multinomial NB
W = load("wine")
D = load("diabetes")


def test_knn_classifier_identical_predictions():
    for k in (1, 5, 15):
        m = mlfs.KNNClassifier(k).fit(G.X_train, G.y_train)
        s = neighbors.KNeighborsClassifier(k).fit(G.X_train, G.y_train)
        assert np.mean(m.predict(G.X_test) == s.predict(G.X_test)) >= 0.995, k


def test_knn_regressor_identical_predictions():
    m = mlfs.KNNRegressor(7).fit(D.X_train, D.y_train)
    s = neighbors.KNeighborsRegressor(7).fit(D.X_train, D.y_train)
    close(m.predict(D.X_test), s.predict(D.X_test), 1e-9, "knn regression")


def test_gaussian_nb_probabilities():
    m = mlfs.GaussianNB().fit(W.X_train, W.y_train)
    s = naive_bayes.GaussianNB().fit(W.X_train, W.y_train)
    close(m.predict_proba(W.X_test), s.predict_proba(W.X_test), 1e-9, "GNB proba")


def test_multinomial_nb_probabilities():
    m = mlfs.MultinomialNB(alpha=0.5).fit(GR.X_train, GR.y_train)
    s = naive_bayes.MultinomialNB(alpha=0.5).fit(GR.X_train, GR.y_train)
    close(m.predict_proba(GR.X_test), s.predict_proba(GR.X_test), 1e-9, "MNB proba")


def test_svm_smo_matches_libsvm_dual_solution():
    for kernel, C in (("linear", 1.0), ("rbf", 1.0), ("rbf", 10.0)):
        m = mlfs.SVC(C=C, kernel=kernel, tol=1e-3).fit(B.X_train, B.y_train)
        s = svm.SVC(C=C, kernel=kernel, tol=1e-3).fit(B.X_train, B.y_train)
        assert m.converged_, kernel
        assert abs(len(m.support_) - s.n_support_.sum()) <= 2, (kernel, C)
        # both solvers stop at the same KKT tolerance, so decision values agree to ~tol
        close(m.decision_function(B.X_test), s.decision_function(B.X_test), 2e-2, f"{kernel} C={C}")
        assert np.mean(m.predict(B.X_test) == s.predict(B.X_test)) >= 0.99


def test_svm_separates_moons_with_rbf():
    M = load("moons")
    m = mlfs.SVC(C=1.0, kernel="rbf").fit(M.X_train, M.y_train)
    assert m.score(M.X_test, M.y_test) >= 0.9


def test_perceptron_learns_breast_cancer():
    m = mlfs.Perceptron(random_state=0).fit(B.X_train, B.y_train)
    s = linear_model.Perceptron(random_state=0).fit(B.X_train, B.y_train)
    assert m.score(B.X_test, B.y_test) >= s.score(B.X_test, B.y_test) - 0.04


def test_mlp_close_to_sklearn_on_digits():
    m = mlfs.MLPClassifier(hidden=64, random_state=0).fit(G.X_train, G.y_train)
    s = neural_network.MLPClassifier((64,), random_state=0, max_iter=200).fit(G.X_train, G.y_train)
    assert abs(m.score(G.X_test, G.y_test) - s.score(G.X_test, G.y_test)) <= 0.02


def test_api_clone_and_params_roundtrip():
    m = mlfs.SVC(C=3.0, kernel="linear")
    c = m.clone()
    assert c.get_params() == m.get_params() and c is not m
    c.set_params(C=0.5)
    assert c.C == 0.5 and m.C == 3.0


if __name__ == "__main__":
    raise SystemExit(_harness.run(globals()))
