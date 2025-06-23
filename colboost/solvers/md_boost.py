import numpy as np
import logging
from typing import Optional, Tuple
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver

logger = logging.getLogger(__name__)


class MDBoost(Solver):
    """
    Implements MDBoost (Margin Distribution Boosting, Shen & Li 2009).

    Reference:
    Shen, Chunhua and Hanxi Li.
    Boosting Through Optimization of Margin Distributions.
    IEEE Transactions on Neural Networks 21(4), 659–666 (2009).
    https://doi.org/10.1109/TNN.2010.2040484
    """

    def __init__(self):
        super().__init__()
        self.use_identity_approx = True

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
                forest_size, lb=0.0, vtype=GRB.CONTINUOUS, name="w"
            )
            rho = model.addVars(
                data_size, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho"
            )

            margin_constraints = [
                model.addConstr(
                    rho[i]
                    == sum(
                        y_train[i] * predictions[j][i] * weights[j]
                        for j in range(forest_size)
                    ),
                    name=f"margin_{i}",
                )
                for i in range(data_size)
            ]

            wsum_constraint = model.addConstr(
                sum(weights[j] for j in range(forest_size)) == hyperparam,
                name="sum_weights",
            )

            if getattr(self, "use_identity_approx", True):
                logger.info(
                    "Using identity matrix approximation for variance."
                )
                quadratic_term = 0.5 * sum(
                    rho[i] * rho[i] for i in range(data_size)
                )
            else:
                A = np.ones((data_size, data_size)) * (-1 / (data_size - 1))
                np.fill_diagonal(A, 1)
                rho_vec = np.array([rho[i] for i in range(data_size)])
                quadratic_term = 0.5 * np.dot(rho_vec.T, np.dot(A, rho_vec))

            linear_term = sum(rho[i] for i in range(data_size))
            model.setObjective(linear_term - quadratic_term, GRB.MAXIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                lp_weights = np.array(
                    [weights[j].X for j in range(forest_size)]
                )
                alpha = np.array(
                    [
                        max(0, margin_constraints[i].Pi)
                        for i in range(data_size)
                    ]
                )
                beta = wsum_constraint.Pi
                obj_val = model.ObjVal
                solve_time = model.Runtime

                return alpha, beta, lp_weights, obj_val, solve_time

            logger.warning("Gurobi failed to find an optimal solution.")
            return None, None, None, None, None
