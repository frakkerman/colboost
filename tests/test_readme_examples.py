import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from colboost.ensemble import EnsembleClassifier


def test_fit_and_compute_margins():
    X, y = make_classification(n_samples=200, n_features=20, random_state=0)
    y = 2 * y - 1  # Convert labels from {0, 1} to {-1, +1}

    model = EnsembleClassifier(solver="lp_boost", max_iter=10)
    model.fit(X, y)

    # Ensure model is fitted
    assert len(model.learners) > 0
    assert model.weights is not None

    # Score should be between 0 and 1
    score = model.score(X, y)
    assert 0.0 <= score <= 1.0

    # Compute margins and check shape
    margins = model.compute_margins(X, y)
    assert margins.shape == (len(y),)


def test_reweight_ensemble_with_adaboost():
    X, y = make_classification(n_samples=200, n_features=20, random_state=42)
    y = 2 * y - 1  # Convert labels to {-1, +1}

    ada = AdaBoostClassifier(n_estimators=10, random_state=0)
    ada.fit(X, y)

    model = EnsembleClassifier(solver="lp_boost")
    model.reweight_ensemble(X, y, learners=ada.estimators_)

    # Ensure correct number of learners
    assert len(model.learners) == 10
    assert model.weights is not None
    assert np.all(model.weights >= 0)

    # Check accuracy range
    acc = model.score(X, y)
    assert 0.0 <= acc <= 1.0
