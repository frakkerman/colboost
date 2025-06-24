import numpy as np
from colboost.ensemble import EnsembleClassifier
from colboost.solvers.solver import SolveResult
from unittest.mock import patch


def test_custom_gurobi_env(dataset_and_preds):
    preds, y = dataset_and_preds
    solver = EnsembleClassifier(max_iter=1)
    solver.fit(np.random.randn(len(y), 4), y)
    assert solver.weights is not None


def test_solver_params_passed():
    X, y = np.random.randn(20, 4), np.ones(20)

    with patch("colboost.solvers.nm_boost.NMBoost.solve") as mock_solve:
        mock_solve.return_value = SolveResult(
            alpha=np.ones(20) / 20,
            beta=0,
            weights=np.array([1.0]),
            obj_val=1.0,
            solve_time=0.1,
        )
        model = EnsembleClassifier(
            solver="nm_boost",
            max_iter=1,
            gurobi_time_limit=10,
            gurobi_num_threads=3,
        )
        model.fit(X, y)
        # Check that the solve method was called
        assert mock_solve.called, (
            "NMBoost.solve was not called by EnsembleClassifier"
        )
        # Retrieve keyword arguments safely
        kwargs = mock_solve.call_args.kwargs
        assert kwargs["time_limit"] == 10
        assert kwargs["num_threads"] == 3


def test_dual_constraint_early_stop():
    X, y = np.random.randn(20, 4), np.ones(20)

    with patch("colboost.solvers.lp_boost.LPBoost.solve") as mock:
        mock.return_value = (
            np.array([0.5, 0.5]),  # optim_weights
            0.0,  # beta
            np.array([1.0]),  # objval
            1.0,  # time
            0.01,  # solve_time
        )

        model = EnsembleClassifier(max_iter=5, check_dual_const=True)

        # Patch np.dot to trigger early stopping after 1st iteration
        with patch("numpy.dot", side_effect=[-1.0]):  # force dual_sum <= beta
            model.fit(X, y)

        assert len(model.learners) == 0
