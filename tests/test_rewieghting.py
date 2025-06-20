import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from colboost.ensemble import EnsembleClassifier


def test_fit_weights_only_executes():
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    y = 2 * y - 1

    # Pre-train AdaBoost
    boost = AdaBoostClassifier(n_estimators=5)
    boost.fit(X, (y + 1) // 2)

    # Reweight ensemble
    model = EnsembleClassifier()
    model.fit_weights_only(X, y, learners=boost.estimators_)

    assert len(model.learners) == 5
    assert len(model.weights) == 5
    assert np.all(np.array(model.weights) >= 0)


def test_fit_weights_only_predict_shape():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    y = 2 * y - 1

    boost = AdaBoostClassifier(n_estimators=7)
    boost.fit(X, (y + 1) // 2)

    model = EnsembleClassifier()
    model.fit_weights_only(X, y, learners=boost.estimators_)

    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({-1, 0, 1})


def test_fit_weights_only_margin_computation():
    X, y = make_classification(n_samples=30, n_features=4, random_state=1)
    y = 2 * y - 1

    boost = AdaBoostClassifier(n_estimators=3)
    boost.fit(X, (y + 1) // 2)

    model = EnsembleClassifier()
    model.fit_weights_only(X, y, learners=boost.estimators_)

    margins = model.compute_margins(X, y)
    assert margins.shape == (30,)
    assert np.issubdtype(margins.dtype, np.floating)


def test_fit_weights_only_with_random_forest():
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    y = 2 * y - 1

    rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    rf.fit(X, (y + 1) // 2)
    learners = list(rf.estimators_)

    model = EnsembleClassifier(max_iter=1)
    model.fit_weights_only(X, y, learners)

    assert len(model.weights) == len(learners)
    assert all(w >= 0 for w in model.weights)