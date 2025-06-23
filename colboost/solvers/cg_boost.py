import numpy as np
import logging
from typing import Optional, Tuple
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver

logger = logging.getLogger(__name__)


class CGBoost(Solver):
    """
    Implements CGBoost (Bi et al., 2004), L2-regularized margin formulation.

    Reference:
    Bi, Jinbo, Zhang, Tong, & Bennett, Kristin P. (2004).
    Column-generation boosting methods for mixture of kernels.
    SIGKDD International Conference on Knowledge Discovery and Data Mining.
    https://doi.org/10.1145/1014052.1014113
    """

    def __init__(self):
        super().__init__()

    def solve(
        self,
        predictions: list[np.ndarray],
        y_train: np.ndarray,
        hyperparam: float,
        time_limit: int,
        num_threads: int,
        seed: int,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[float],
        Optional[float],
    ]:
        forest_size = len(predictions)
        data_size = len(y_train)

        with Model(env=self.env) as model:
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = time_limit
            model.Params.Threads = num_threads
            model.Params.Seed = seed

            weights = model.addVars(
                forest_size,
                lb=0.0,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name="w",
            )

            xi = model.addVars(
                data_size,
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name="xi",
            )

            acc_constraints = [
                model.addConstr(
                    sum(
                        y_train[i] * predictions[j][i] * weights[j]
                        for j in range(forest_size)
                    )
                    + xi[i]
                    >= 1,
                    name=f"acc_{i}",
                )
                for i in range(data_size)
            ]

            model.setObjective(
                0.5 * sum(weights[j] * weights[j] for j in range(forest_size))
                + hyperparam * sum(xi[i] for i in range(data_size)),
                GRB.MINIMIZE,
            )

            model.optimize()

            if model.status == GRB.OPTIMAL:
                lp_weights = np.array(
                    [weights[j].X for j in range(forest_size)]
                )
                alpha = np.array(
                    [max(0, acc_constraints[i].Pi) for i in range(data_size)]
                )
                beta = max(
                    np.dot(alpha * y_train, preds) for preds in predictions
                )
                obj_val = model.ObjVal
                solve_time = model.Runtime

                return alpha, beta, lp_weights, obj_val, solve_time

            logger.warning("Gurobi failed to find an optimal solution.")
            return None, None, None, None, None
