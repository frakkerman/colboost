import numpy as np
import pytest
from colboost.ensemble import EnsembleClassifier


def test_initial_state():
    model = EnsembleClassifier()
    assert model.learners == []
    assert model.weights == []


def test_single_iteration_fit(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=1)
    model.fit(X, y)
    assert len(model.learners) == 1
    assert len(model.weights) == 1


def test_prediction_shape(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_not_fitted():
    model = EnsembleClassifier()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 4)))


def test_train_objective_logging(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    assert len(model.train_accuracies_) == len(model.learners)
    assert len(model.objective_values_) == len(model.learners)


def test_margin_sign_prediction_agreement(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    margins = model.compute_margins(X, y)
    preds = model.predict(X)
    expected_preds = y * np.where(margins >= 0, 1, -1)
    assert np.all(expected_preds == preds)


def test_early_stopping(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=10, obj_check=2, obj_eps=1e-5)
    model.fit(X, y)
    assert len(model.learners) < model.max_iter


def test_invalid_labels():
    X = np.random.randn(10, 3)
    y = np.array([0.1] * 10)
    model = EnsembleClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_score_with_fitted_model(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=3)
    model.fit(X, y)
    score = model.score(X, y)
    assert 0.0 <= score <= 1.0


def test_model_attributes_after_training(sample_dataset):
    X, y = sample_dataset
    model = EnsembleClassifier(max_iter=3, solver="lp_boost")
    model.fit(X, y)

    # Check basic attribute existence and types
    assert isinstance(model.learners, list)
    assert isinstance(model.weights, np.ndarray)
    assert isinstance(model.objective_values_, list)
    assert isinstance(model.solve_times_, list)
    assert isinstance(model.train_accuracies_, list)
    assert isinstance(model.n_iter_, int)
    assert isinstance(model.model_name_, str)

    # Check expected lengths
    assert len(model.learners) == model.n_iter_
    assert len(model.weights) == model.n_iter_
    assert len(model.objective_values_) == model.n_iter_
    assert len(model.solve_times_) == model.n_iter_
    assert len(model.train_accuracies_) == model.n_iter_
