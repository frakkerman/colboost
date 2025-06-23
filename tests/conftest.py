import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from gurobipy import Env


@pytest.fixture
def sample_dataset():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    y = 2 * y - 1
    return X, y


@pytest.fixture
def dataset_and_preds():
    X, y = make_classification(n_samples=20, n_features=4, random_state=42)
    y = 2 * y - 1
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    return preds, y