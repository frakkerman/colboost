import numpy as np
import pytest
from sklearn.datasets import make_classification
from colboost.ensemble import EnsembleClassifier
from sklearn.tree import ExtraTreeClassifier

def test_initial_state():
    model = EnsembleClassifier()
    assert model.learners == []
    assert model.weights == []


def test_single_iteration_fit():
    X, y = make_classification(n_samples=30, n_features=4, random_state=1)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=1)
    model.fit(X, y)
    assert len(model.learners) == 1
    assert len(model.weights) == 1


def test_prediction_shape():
    X, y = make_classification(n_samples=50, n_features=4, random_state=3)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_deterministic_output():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    y = 2 * y - 1
    model1 = EnsembleClassifier(max_iter=3)
    model2 = EnsembleClassifier(max_iter=3)
    model1.fit(X, y)
    model2.fit(X, y)
    np.testing.assert_array_almost_equal(model1.weights, model2.weights)


def test_fit_predict():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    y = 2 * y - 1  # convert to -1/+1

    model = EnsembleClassifier(max_iter=50)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({-1, 0, 1})


def test_not_fitted():
    model = EnsembleClassifier()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((10, 4)))


def test_learner_weight_alignment():
    X, y = make_classification(n_samples=50, n_features=4, random_state=1)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=5)
    model.fit(X, y)
    assert len(model.learners) == len(model.weights)
    assert all(w >= 0 for w in model.weights)


def test_early_stopping_by_objective():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    y = 2 * y - 1  # Convert to -1/+1

    model = EnsembleClassifier(max_iter=10, obj_check=2, obj_eps=1e-5)
    model.fit(X, y)

    # Should stop before reaching max_iter if objective does not improve
    assert len(model.learners) < model.max_iter, (
        "Early stopping by objective did not trigger."
    )
    assert model.weights is not None
    assert all(w >= 0 for w in model.weights)


def test_invalid_labels():
    X = np.random.randn(10, 3)
    y = np.array([0.1] * 10)  # Not -1/+1
    model = EnsembleClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_margin_sign_agrees_with_prediction():
    X, y = make_classification(n_samples=50, n_features=4, random_state=4)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=4)
    model.fit(X, y)

    margins = model.compute_margins(X, y)
    preds = model.predict(X)
    assert np.all(np.sign(margins) == preds)

def test_train_accuracies_logged():
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    assert len(model.train_accuracies_) == len(model.learners)

def test_objective_values_logged():
    X, y = make_classification(n_samples=60, n_features=4, random_state=1)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    assert len(model.objective_values_) == len(model.learners)

def test_margin_computation():
    X, y = make_classification(n_samples=40, n_features=4, random_state=2)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    margins = model.compute_margins(X, y)
    assert margins.shape == (len(y),)
    assert np.issubdtype(margins.dtype, np.floating)

def test_fit_with_extra_tree_classifier():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    y = 2 * y - 1

    base_learner = ExtraTreeClassifier(max_depth=1, random_state=0)
    model = EnsembleClassifier(max_iter=3, base_estimator=base_learner)
    model.fit(X, y)

    assert len(model.learners) > 0
    assert all(isinstance(clf, ExtraTreeClassifier) for clf in model.learners)

def test_use_crb_predictions():
    X, y = make_classification(n_samples=50, n_features=6, n_informative=2, n_redundant=1, random_state=0)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=2, use_crb=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)

def test_objective_epsilon_threshold():
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    y = 2 * y - 1
    model = EnsembleClassifier(max_iter=10, obj_check=1, obj_eps=1e10)
    model.fit(X, y)
    assert len(model.learners) == 2
