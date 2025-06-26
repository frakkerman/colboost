import numpy as np
import math
import logging
from typing import List
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver, SolveResult

logger = logging.getLogger("colboost.solver")


class ERLPBoost(Solver):
    """
    Implements Entropy Regularized LPBoost (Warmuth et al., 2008).

    Reference:
    Warmuth, M.K., Glocer, K.A., Vishwanathan, S.V.N. (2008).
    Entropy Regularized LPBoost. Lecture Notes in Computer Science, vol 5254.
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
        Solves the ERLPBoost optimization problem to determine optimal ensemble weights.

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
        dist = np.full(data_size, 1 / data_size)

        ln_n = math.log(data_size)
        half_tol = 0.99 / 2.0
        eta = max(0.5, ln_n / half_tol)
        max_iter = int(max(4.0 / half_tol, (8.0 * ln_n / (half_tol**2))))

        gamma_hat = 1.0
        total_solve_time = 0.0

        with Model(env=self.env) as model:
            self.set_gurobi_params(model, time_limit, num_threads, seed)
            model.Params.NumericFocus = 3

            gamma, dist_vars = self._add_variables(
                model, data_size, hyperparam
            )
            self._add_normalization_constraint(model, dist_vars)

            for iteration in range(max_iter):
                margin_constraints = []
                self._add_margin_constraints(
                    model,
                    predictions,
                    y_train,
                    dist_vars,
                    gamma,
                    margin_constraints,
                )

                entropy_expr = self._compute_entropy(dist_vars, dist)
                model.setObjective(gamma + entropy_expr / eta, GRB.MINIMIZE)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    logger.warning(
                        f"Gurobi failed to find an optimal solution (status: {model.status})"
                    )
                    return SolveResult(None, None, None, None, None)

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
                gamma_star = max(edges) + float(entropy_expr.getValue()) / eta
                gamma_hat = min(gamma_hat, objval)

                if gamma_hat - gamma_star <= half_tol:
                    break

                dist = dist_new

            alpha = np.array([dist_vars[i].X for i in range(data_size)])
            weights = np.abs(np.array([c.Pi for c in margin_constraints]))
            beta = 0.0

            return SolveResult(
                alpha=alpha,
                beta=beta,
                weights=weights,
                obj_val=objval,
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

    def _add_normalization_constraint(self, model, dist_vars):
        model.addConstr(
            sum(dist_vars[i] for i in dist_vars) == 1, name="sum_to_1"
        )

    def _add_margin_constraints(
        self, model, predictions, y_train, dist_vars, gamma, constraints
    ):
        data_size = len(y_train)
        for j, preds in enumerate(predictions):
            margin_expr = sum(
                dist_vars[i] * y_train[i] * preds[i] for i in range(data_size)
            )
            constraints.append(
                model.addConstr(margin_expr <= gamma, name=f"margin_{j}")
            )

    def _compute_entropy(self, dist_vars, dist: np.ndarray):
        EPSILON = 1e-9
        return sum(
            dist_vars[i]
            * (
                math.log(dist[i] + EPSILON)
                + (dist_vars[i] - dist[i]) / (dist[i] + EPSILON)
            )
            for i in dist_vars
        )
