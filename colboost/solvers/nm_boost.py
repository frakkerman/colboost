import numpy as np
from typing import List
from gurobipy import GRB, Model
from colboost.solvers.solver import Solver, SolveResult


class NMBoost(Solver):
    """
    Implements NMBoost: a method optimizing the negative margin.

    Reference: Custom boosting variant focused on minimizing negative margin components.
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
        Solves the NMBoost optimization problem to determine optimal ensemble weights.

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

            weights, rhoi, rhonegi = self._add_variables(
                model, forest_size, data_size
            )
            acc_constraints, neg_margin_constraints = self._add_constraints(
                model, predictions, y_train, weights, rhoi, rhonegi
            )
            sum_constraint = self._add_weight_sum_constraint(model, weights)
            self._set_objective(model, rhoi, rhonegi, hyperparam)

            model.optimize()
            return self._extract_solution(
                model,
                weights,
                acc_constraints,
                constraint_type="nm_boost",
                neg_constraints=neg_margin_constraints,
                sum_constraint=sum_constraint
            )

    def _add_variables(self, model, forest_size: int, data_size: int):
        weights = model.addVars(
            forest_size,
            lb=0.0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="weights",
        )
        rhoi = model.addVars(
            data_size, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho"
        )
        rhonegi = model.addVars(
            data_size,
            lb=-GRB.INFINITY,
            ub=0.0,
            vtype=GRB.CONTINUOUS,
            name="rhoneg",
        )
        return weights, rhoi, rhonegi

    def _add_constraints(
        self, model, predictions, y_train, weights, rhoi, rhonegi
    ):
        forest_size = len(predictions)
        data_size = len(y_train)

        acc_constraints = []
        neg_margin_constraints = []

        for i in range(data_size):
            expr = sum(
                y_train[i] * predictions[j][i] * weights[j]
                for j in range(forest_size)
            )
            acc_constraints.append(
                model.addConstr(expr >= rhoi[i], name=f"acc_{i}")
            )
            neg_margin_constraints.append(
                model.addConstr(
                    rhonegi[i] <= rhoi[i] - (1 / forest_size),
                    name=f"neg_margin_{i}",
                )
            )

        return acc_constraints, neg_margin_constraints

    def _add_weight_sum_constraint(self, model, weights):
        return model.addConstr(
            sum(weights[j] for j in weights) == 1.0, name="weight_sum"
        )

    def _set_objective(self, model, rhoi, rhonegi, hyperparam: float):
        model.setObjective(
            sum(rhonegi[i] + hyperparam * rhoi[i] for i in rhoi),
            GRB.MAXIMIZE,
        )
