import numpy as np
from colboost.ensemble import EnsembleClassifier
from unittest.mock import patch


def test_custom_gurobi_env(dataset_and_preds):
    preds, y = dataset_and_preds
    solver = EnsembleClassifier(max_iter=1)
    solver.fit(np.random.randn(len(y), 4), y)
    assert solver.weights is not None


def test_solver_params_passed():
    X, y = np.random.randn(20, 4), np.ones(20)
    with patch("colboost.solvers.lp_boost.LPBoost.solve") as mock:
        mock.return_value = (np.ones(20) / 20, 0, np.array([1.0]), 1.0, 0.1)
        model = EnsembleClassifier(
            max_iter=1, gurobi_time_limit=10, gurobi_num_threads=3
        )
        model.fit(X, y)
        kwargs = mock.call_args[1]
        assert kwargs["time_limit"] == 10
        assert kwargs["num_threads"] == 3


def test_dual_constraint_early_stop():
    X, y = np.random.randn(20, 4), np.ones(20)
    model = EnsembleClassifier(max_iter=5, check_dual_const=True)
    model.fit(X, y)
    assert len(model.learners) == 1
