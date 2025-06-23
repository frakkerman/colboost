import numpy as np
import logging
from typing import Optional, Tuple
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver

logger = logging.getLogger(__name__)


class LPBoost(Solver):
    """
    Implements LPBoost (Demiriz et al., 2002), soft-margin variant.

    Reference:
    Demiriz, A., Bennett, K.P., & Shawe-Taylor, J.
    Linear Programming Boosting via Column Generation.
    Machine Learning, 46, 225–254 (2002).
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
        seed: int
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[float],
        Optional[float],
    ]:
        """
        Solves the LPBoost optimization problem to determine optimal ensemble weights.

        Parameters
        ----------
        predictions : list of np.ndarray
            List of base learner predictions on the training set.
        y_train : np.ndarray
            Target values, must be -1/+1.
        hyperparam : float
            Regularization parameter controlling the trade-off with slack.
        time_limit : int
            Maximum allowed time (in seconds) for the solver.
        num_threads : int
            Number of threads to use for Gurobi.
        seed : int
            Random seed for Gurobi.

        Returns
        -------
        alpha : np.ndarray or None
            Dual values for the margin constraints (used as sample weights).
        beta : float or None
            Placeholder (always 0.0 for LPBoost).
        weights : np.ndarray or None
            Optimized weights for the weak learners.
        obj_val : float or None
            Final objective value of the solved LP.
        solve_time : float or None
            Runtime in seconds.
        """
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
            xi = model.addVars(
                data_size, lb=0.0, vtype=GRB.CONTINUOUS, name="xi"
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
                sum(weights[j] for j in range(forest_size))
                + hyperparam * sum(xi[i] for i in range(data_size)),
                GRB.MINIMIZE,
            )

            model.optimize()

            if model.status == GRB.OPTIMAL:
                lp_weights = np.array(
                    [weights[j].X for j in range(forest_size)]
                )
                alpha = np.array(
                    [acc_constraints[i].Pi for i in range(data_size)]
                )
                beta = max(
                    np.dot(alpha * y_train, preds) for preds in predictions
                )
                obj_val = model.ObjVal
                solve_time = model.Runtime

                if np.all(alpha == 0):
                    logger.warning("All dual values (alphas) are zero.")

                return alpha, beta, lp_weights, obj_val, solve_time

            logger.warning("Gurobi failed to find an optimal solution.")
            return None, None, None, None, None
