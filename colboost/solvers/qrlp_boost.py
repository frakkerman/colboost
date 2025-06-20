import numpy as np
import math
import logging
from typing import Optional, Tuple
from gurobipy import GRB, Model, Env
from colboost.solvers.solver import Solver

logger = logging.getLogger(__name__)


class QRLPBoost(Solver):
    """
    Implements QRLPBoost: a method inspired by ERLP-Boost.

    Reference: Custom boosting variant.
    """

    def solve(
        self,
        predictions: list[np.ndarray],
        y_train: np.ndarray,
        hyperparam: float,
        time_limit: int,
        num_threads: int,
        seed: int,
        env: Optional[Env] = None,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[float],
        Optional[float],
    ]:
        data_size = len(y_train)
        forest_size = len(predictions)

        dist = np.full(data_size, 1 / data_size)
        weights = np.zeros(forest_size)
        gamma = float("inf")

        ln_n_sample = math.log(data_size)
        half_tol = 1e-4
        eta = max(0.5, ln_n_sample / half_tol)

        with Model(env=env) as model:
            model.Params.OutputFlag = 0
            model.Params.Threads = num_threads
            model.Params.Seed = seed

            gamma_var = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma")
            dist_vars = model.addVars(
                data_size, lb=0.0, ub=1.0 / hyperparam, vtype=GRB.CONTINUOUS, name="dist"
            )

            sum_constraint = model.addConstr(
                sum(dist_vars[i] for i in range(data_size)) == 1,
                name="sum_dist_is_1",
            )

            margin_constraints = [
                model.addConstr(
                    sum(dist_vars[i] * y_train[i] * predictions[j][i] for i in range(data_size))
                    <= gamma_var,
                    name=f"margin_{j}",
                )
                for j in range(forest_size)
            ]

            total_solve_time = 0.0
            while True:
                reg_term = sum(
                    (np.log(dist[i]) if dist[i] > 0 else 0) * dist_vars[i]
                    + (dist_vars[i] * dist_vars[i] / (2 * dist[i]))
                    for i in range(data_size)
                )
                model.setObjective(gamma_var + (1 / eta) * reg_term, GRB.MINIMIZE)

                model.optimize()

                if model.status != GRB.OPTIMAL:
                    logger.warning("Gurobi failed to find an optimal solution.")
                    return None, None, None, None, None

                dist_new = np.array([dist_vars[i].X for i in range(data_size)])
                objval = model.ObjVal
                total_solve_time += model.Runtime

                if (
                    np.any(dist_new <= 0)
                    or abs(gamma - objval) < 2 * half_tol
                ):
                    break

                dist = dist_new
                gamma = objval

                weights = np.array([abs(constr.Pi) for constr in margin_constraints])

            alpha = np.array([dist_vars[i].X for i in range(data_size)])
            beta = sum_constraint.Pi
            obj_val = model.ObjVal

            return alpha, beta, weights, obj_val, total_solve_time
