import numpy as np
from colboost.solvers import (
    CGBoost,
    LPBoost,
    MDBoost,
    ERLPBoost,
    QRLPBoost,
    NMBoost,
)

solvers = [CGBoost, LPBoost, MDBoost, ERLPBoost, QRLPBoost, NMBoost]

import pytest


@pytest.mark.parametrize("SolverClass", solvers)
def test_solver_variants(SolverClass, dataset_and_preds):
    preds, y = dataset_and_preds
    solver = SolverClass()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=1.0,
    )
    assert weights is not None and weights[0] >= 0
    assert np.isfinite(obj_val) and solve_time >= 0
