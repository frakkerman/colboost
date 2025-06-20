import numpy as np
import logging
from typing import Optional, Tuple
from gurobipy import GRB, Model, Env
from colboost.solvers.solver import Solver

logger = logging.getLogger(__name__)


class NMBoost(Solver):
    """
    Implements NMBoost: a method optimizing the negative margin.

    Reference: Custom boosting variant focused on minimizing negative margin components.
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
        forest_size = len(predictions)
        data_size = len(y_train)

        with Model(env=env) as model:
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = time_limit
            model.Params.Threads = num_threads
            model.Params.Seed = seed

            weights = model.addVars(
                forest_size, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="weights"
            )

            rhoi = model.addVars(
                data_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho"
            )
            rhonegi = model.addVars(
                data_size, lb=-GRB.INFINITY, ub=0.0, vtype=GRB.CONTINUOUS, name="rhoneg"
            )

            acc_constraints = []
            neg_margin_constraints = []

            for i in range(data_size):
                expr = sum(y_train[i] * predictions[j][i] * weights[j] for j in range(forest_size))
                acc_constraints.append(model.addConstr(expr >= rhoi[i], name=f"acc_{i}"))
                neg_margin_constraints.append(
                    model.addConstr(rhonegi[i] <= rhoi[i] - (1 / forest_size), name=f"neg_margin_{i}")
                )

            wsum_constraint = model.addConstr(
                sum(weights[j] for j in range(forest_size)) == 1.0, name="weight_sum"
            )

            model.setObjective(
                sum(rhonegi[i] + hyperparam * rhoi[i] for i in range(data_size)),
                GRB.MAXIMIZE,
            )

            model.optimize()

            if model.status == GRB.OPTIMAL:
                lp_weights = np.array([weights[j].X for j in range(forest_size)])
                alpha = np.array([
                    abs(acc_constraints[i].Pi) + neg_margin_constraints[i].Pi
                    for i in range(data_size)
                ])
                beta = wsum_constraint.Pi
                obj_val = model.ObjVal
                solve_time = model.Runtime

                return alpha, beta, lp_weights, obj_val, solve_time

            logger.warning("Gurobi failed to find an optimal solution.")
            return None, None, None, None, None
