import numpy as np
import logging
from typing import List
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver, SolveResult

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
        predictions: List[np.ndarray],
        y_train: np.ndarray,
        hyperparam: float,
        time_limit: int,
        num_threads: int,
        seed: int,
    ) -> SolveResult:
        """
        Solves the CGBoost optimization problem to determine optimal ensemble weights.

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
        forest_size = len(predictions)
        data_size = len(y_train)

        with Model(env=self.env) as model:
            self.set_gurobi_params(model, time_limit, num_threads, seed)

            weights, slack_vars = self._add_variables(
                model, forest_size, data_size
            )
            acc_constraints = self._add_constraints(
                model, predictions, y_train, weights, slack_vars
            )
            self._set_objective(model, weights, slack_vars, hyperparam)

            model.optimize()
            return self._extract_solution(
                model,
                weights,
                acc_constraints,
                predictions=predictions,
                y_train=y_train,
                constraint_type="clipped"
            )

    def _add_variables(self, model, forest_size: int, data_size: int):
        weights = model.addVars(
            forest_size,
            lb=0.0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="w",
        )
        slack = model.addVars(
            data_size, lb=0.0, vtype=GRB.CONTINUOUS, name="xi"
        )
        return weights, slack

    def _add_constraints(self, model, predictions, y_train, weights, slack):
        forest_size = len(predictions)
        data_size = len(y_train)

        return [
            model.addConstr(
                sum(
                    y_train[i] * predictions[j][i] * weights[j]
                    for j in range(forest_size)
                )
                + slack[i]
                >= 1,
                name=f"acc_{i}",
            )
            for i in range(data_size)
        ]

    def _set_objective(self, model, weights, slack, C: float):
        model.setObjective(
            0.5 * sum(weights[j] * weights[j] for j in weights)
            + C * sum(slack[i] for i in slack),
            GRB.MINIMIZE,
        )
