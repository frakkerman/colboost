import numpy as np
import math
import logging
from typing import Optional, Tuple
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver

logger = logging.getLogger(__name__)


class ERLPBoost(Solver):
    """
    Implements Entropy Regularized LPBoost (Warmuth et al., 2008).

    Reference:
    Warmuth, M.K., Glocer, K.A., Vishwanathan, S.V.N. (2008).
    Entropy Regularized LPBoost. Lecture Notes in Computer Science, vol 5254.
    https://doi.org/10.1007/978-3-540-87987-9_23
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
        data_size = len(y_train)

        dist = np.full(data_size, 1 / data_size)
        ln_n = math.log(data_size)
        half_tol = 1e-4  # could be exposed as arg
        eta = max(0.5, ln_n / half_tol)
        max_iter = int(max(4.0 / half_tol, (8.0 * ln_n / (half_tol**2))))

        gamma_hat = 1.0
        total_solve_time = 0.0
        objval = None
        margin_constraints = []

        with Model(env=self.env) as model:
            model.Params.OutputFlag = 0
            model.Params.Threads = num_threads
            model.Params.Seed = seed
            model.Params.NumericFocus = 3
            model.Params.TimeLimit = time_limit

            gamma = model.addVar(
                lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma"
            )
            dist_vars = model.addVars(
                data_size,
                lb=0.0,
                ub=1.0 / hyperparam,
                vtype=GRB.CONTINUOUS,
                name="dist",
            )

            model.addConstr(
                sum(dist_vars[i] for i in range(data_size)) == 1,
                name="sum_to_1",
            )

            for iteration in range(max_iter):
                for j, preds in enumerate(predictions):
                    margin_expr = sum(
                        dist_vars[i] * y_train[i] * preds[i]
                        for i in range(data_size)
                    )
                    margin_constraints.append(
                        model.addConstr(
                            margin_expr <= gamma, name=f"margin_{j}"
                        )
                    )

                EPSILON = 1e-9
                entropy = sum(
                    dist_vars[i]
                    * (
                        math.log(dist[i] + EPSILON)
                        + (dist_vars[i] - dist[i]) / (dist[i] + EPSILON)
                    )
                    for i in range(data_size)
                )

                model.setObjective(gamma + entropy / eta, GRB.MINIMIZE)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    logger.warning(
                        "Gurobi failed to find an optimal solution."
                    )
                    return None, None, None, None, None

                dist_new = np.array([dist_vars[i].X for i in range(data_size)])
                total_solve_time += model.Runtime
                objval = model.ObjVal

                edges = [
                    sum(
                        dist_new[i] * y_train[i] * preds[i]
                        for i in range(data_size)
                    )
                    for preds in predictions
                ]
                gamma_star = max(edges) + float(entropy.getValue()) / eta
                gamma_hat = min(gamma_hat, objval)
                if gamma_hat - gamma_star <= half_tol:
                    break

                dist = dist_new

            model.optimize()

            alpha = np.array([dist_vars[i].X for i in range(data_size)])
            lp_weights = np.abs(np.array([c.Pi for c in margin_constraints]))
            beta = 0.0

            return alpha, beta, lp_weights, objval, total_solve_time
