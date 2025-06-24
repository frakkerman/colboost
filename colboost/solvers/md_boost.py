import numpy as np
import logging
from typing import List
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver, SolveResult

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
        predictions: List[np.ndarray],
        y_train: np.ndarray,
        hyperparam: float,
        time_limit: int,
        num_threads: int,
        seed: int,
    ) -> SolveResult:
        """
        Solves the MDBoost optimization problem to determine optimal ensemble weights.

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

            weights, rho = self._add_variables(model, forest_size, data_size)
            margin_constraints = self._add_constraints(
                model, predictions, y_train, weights, rho
            )
            sum_constraint = self._add_weight_sum_constraint(
                model, weights, hyperparam
            )
            self._set_objective(model, rho, data_size)

            model.optimize()
            return self._extract_solution(
                model,
                weights,
                margin_constraints,
                constraint_type="clipped",
                sum_constraint=sum_constraint
            )

    def _add_variables(self, model, forest_size: int, data_size: int):
        weights = model.addVars(
            forest_size, lb=0.0, vtype=GRB.CONTINUOUS, name="w"
        )
        rho = model.addVars(
            data_size, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho"
        )
        return weights, rho

    def _add_constraints(self, model, predictions, y_train, weights, rho):
        forest_size = len(predictions)
        data_size = len(y_train)

        return [
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

    def _add_weight_sum_constraint(self, model, weights, total_weight):
        return model.addConstr(
            sum(weights[j] for j in weights) == total_weight,
            name="sum_weights",
        )

    def _set_objective(self, model, rho, data_size: int):
        if getattr(self, "use_identity_approx", True):
            logger.info("Using identity matrix approximation for variance.")
            quad_term = 0.5 * sum(rho[i] * rho[i] for i in rho)
        else:
            logger.warning(
                "Non-identity approximation is not supported in Gurobi expressions."
            )
            raise NotImplementedError(
                "Non-identity matrix approximation is not implemented."
            )

        linear_term = sum(rho[i] for i in rho)
        model.setObjective(linear_term - quad_term, GRB.MAXIMIZE)
