import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from gurobipy import Env
from colboost.ensemble import EnsembleClassifier
from colboost.solvers import get_solver
from colboost.utils.predictions import create_predictions

from unittest.mock import patch


def test_solver_with_custom_gurobi_env():
    # Create a simple binary classification dataset
    X, y = make_classification(n_samples=20, n_features=4, random_state=42)
    y = 2 * y - 1  # Convert to -1/+1 for LPBoost

    # Train a weak learner
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = create_predictions(clf, X, use_crb=False)

    # Initialize Gurobi environment
    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # Get solver
    solver = get_solver("lp_boost")

    # Call solve() with all required named arguments
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=1e-2,
    )

    # Assertions
    assert weights is not None, "Solver failed to return weights"
    assert len(weights) == 1, "Only one weak learner, so one weight expected"
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0

def test_gurobi_parameters_are_passed():
    X, y = make_classification(n_samples=30, n_features=4, random_state=1)
    y = 2 * y - 1

    with patch("colboost.solvers.lp_boost.LPBoost.solve") as mock_solve:
        mock_solve.return_value = (
            np.ones(len(y)) / len(y),
            0.0,
            np.array([1.0]),
            0.0,
            0.1
        )
        model = EnsembleClassifier(max_iter=1, gurobi_time_limit=123, gurobi_num_threads=7)
        model.fit(X, y)

        kwargs = mock_solve.call_args[1]
        assert kwargs["time_limit"] == 123
        assert kwargs["num_threads"] == 7


def test_check_dual_constraint_triggers_stop():
    X = np.random.randn(20, 4)
    y = np.ones(20)
    model = EnsembleClassifier(max_iter=5, check_dual_const=True)
    model.fit(X, y)
    assert len(model.learners) == 1

def test_tradeoff_hyperparam_affects_solution():
    X, y = make_classification(n_samples=50, n_features=4, random_state=1)
    y = 2 * y - 1
    model1 = EnsembleClassifier(max_iter=3, tradeoff_hyperparam=1e-1)
    model2 = EnsembleClassifier(max_iter=3, tradeoff_hyperparam=1e-4)
    model1.fit(X, y)
    model2.fit(X, y)
    assert not np.allclose(model1.weights, model2.weights)
