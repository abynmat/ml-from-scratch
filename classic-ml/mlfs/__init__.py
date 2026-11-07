"""mlfs: classic machine learning in plain NumPy with one fit/predict API.

Author: Abyn Scaria
"""
from .base import (StandardScaler, accuracy, adjusted_rand_index, log_loss, mse, r2)
from .boosting import GradientBoostingClassifier, GradientBoostingRegressor
from .forest import RandomForestClassifier, RandomForestRegressor
from .gmm import GaussianMixture
from .kmeans import KMeans
from .knn import KNNClassifier, KNNRegressor
from .linear import Lasso, LinearRegression, Ridge
from .logistic import LogisticRegression
from .naive_bayes import GaussianNB, MultinomialNB
from .neural import MLPClassifier, Perceptron
from .pca import PCA
from .svm import SVC
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "StandardScaler", "accuracy", "adjusted_rand_index", "log_loss", "mse", "r2",
    "LinearRegression", "Ridge", "Lasso", "LogisticRegression", "KNNClassifier", "KNNRegressor",
    "GaussianNB", "MultinomialNB", "DecisionTreeClassifier", "DecisionTreeRegressor",
    "RandomForestClassifier", "RandomForestRegressor", "GradientBoostingClassifier",
    "GradientBoostingRegressor", "SVC", "KMeans", "GaussianMixture", "PCA", "Perceptron",
    "MLPClassifier",
]
