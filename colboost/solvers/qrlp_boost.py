import numpy as np
import math
import logging
from typing import List
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver, SolveResult

logger = logging.getLogger("colboost.solver")

class QRLPBoost(Solver):
    """
    Implements QRLPBoost: a method inspired by ERLP-Boost.

    Reference: Custom boosting variant.
    """

    def __init__(self):
        super().__init__()

    def solve(
        self,
        predictions: List[np.ndarray],
        y_train: np.ndarray,
        hyperparam: float,
        time_limit: int,
        num_threads: int,
        seed: int,
    ) -> SolveResult:
        """
        Solves the QRLPBoost optimization problem to determine optimal ensemble weights.

        Parameters
        ----------
        predictions : List[np.ndarray]
            Predictions of each base learner on the training set.
        y_train : np.ndarray
            True labels (-1 or +1) for training set.
        hyperparam : float
            trade-off parameter..
        time_limit : int
            Maximum solver runtime in seconds.
        num_threads : int
            Number of threads to use.
        seed : int
            Random seed for Gurobi.

        Returns
        -------
        SolveResult
            Structured result containing alpha, beta, weights, objective value, and solve time.
        """
        data_size = len(y_train)
        forest_size = len(predictions)

        dist = np.full(data_size, 1 / data_size)
        gamma = float("inf")

        ln_n_sample = math.log(data_size)
        half_tol = 0.99/2.0
        eta = max(0.5, ln_n_sample / half_tol)

        with Model(env=self.env) as model:
            self.set_gurobi_params(model, time_limit, num_threads, seed)

            gamma_var, dist_vars = self._add_variables(
                model, data_size, hyperparam
            )
            sum_constraint = self._add_sum_constraint(
                model, dist_vars, data_size
            )
            margin_constraints = self._add_margin_constraints(
                model, predictions, y_train, dist_vars, gamma_var
            )

            total_solve_time = 0.0
            weights = np.zeros(forest_size)

            while True:
                reg_term = self._compute_regularization(
                    dist_vars, dist, data_size
                )
                model.setObjective(
                    gamma_var + (1 / eta) * reg_term, GRB.MINIMIZE
                )
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    logger.warning(
                        "Gurobi failed to find an optimal solution."
                    )
                    return SolveResult(None, None, None, None, None)

                dist_new = np.array([dist_vars[i].X for i in range(data_size)])
                obj_val = model.ObjVal
                total_solve_time += model.Runtime

                if (
                    np.any(dist_new <= 0)
                    or abs(gamma - obj_val) < 2 * half_tol
                ):
                    break

                dist = dist_new
                gamma = obj_val

                weights = np.array([abs(c.Pi) for c in margin_constraints])

            alpha = np.array([dist_vars[i].X for i in range(data_size)])
            beta = sum_constraint.Pi

            return SolveResult(
                alpha=alpha,
                beta=beta,
                weights=weights,
                obj_val=model.ObjVal,
                solve_time=total_solve_time,
            )

    def _add_variables(self, model, data_size: int, hyperparam: float):
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
        return gamma, dist_vars

    def _add_sum_constraint(self, model, dist_vars, data_size: int):
        return model.addConstr(
            sum(dist_vars[i] for i in range(data_size)) == 1.0,
            name="sum_dist_is_1",
        )

    def _add_margin_constraints(
        self, model, predictions, y_train, dist_vars, gamma_var
    ):
        constraints = []
        for j, preds in enumerate(predictions):
            margin_expr = sum(
                dist_vars[i] * y_train[i] * preds[i]
                for i in range(len(y_train))
            )
            constraints.append(
                model.addConstr(margin_expr <= gamma_var, name=f"margin_{j}")
            )
        return constraints

    def _compute_regularization(self, dist_vars, dist, data_size: int):
        return sum(
            (np.log(dist[i]) if dist[i] > 0 else 0) * dist_vars[i]
            + (dist_vars[i] * dist_vars[i] / (2 * dist[i]))
            for i in range(data_size)
        )
