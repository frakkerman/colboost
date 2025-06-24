import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass
import logging
from typing import List
from gurobipy import GRB, Env


@dataclass
class SolveResult:
    alpha: Optional[np.ndarray]
    beta: Optional[float]
    weights: Optional[np.ndarray]
    obj_val: Optional[float]
    solve_time: Optional[float]


class Solver(ABC):
    """
    Abstract base class for solvers assigning ensemble weights.
    Provides shared validation and utility methods.
    """

    def __init__(self):
        self.env = Env(params={"LogFile": ""})
        self.logger = logging.getLogger("colboost.solver")

    def __del__(self):
        if hasattr(self, "env"):
            try:
                self.env.dispose()
            except Exception:
                pass

    @abstractmethod
    def solve(self, args: object, data_train: object, env: object) -> Any:
        raise NotImplementedError(
            "Subclasses must implement the solve() method."
        )

    def _extract_solution(
        self,
        model,
        weights,
        constraints,
        *,
        predictions: Optional[List[np.ndarray]] = None,
        y_train: Optional[np.ndarray] = None,
        constraint_type: str = "non_clipped",
        sum_constraint=None,
        neg_constraints=None,
    ) -> SolveResult:
        if model.status != GRB.OPTIMAL:
            self.logger.warning(
                f"Gurobi failed to find an optimal solution (status: {model.status})"
            )
            return SolveResult(None, None, None, None, None)

        forest_size = len(weights)
        lp_weights = np.array([weights[j].X for j in range(forest_size)])

        # Alpha extraction
        if constraint_type == "non_clipped":
            alpha = np.array(
                [constraints[i].Pi for i in range(len(constraints))]
            )
        elif constraint_type == "clipped":
            alpha = np.array(
                [max(0, constraints[i].Pi) for i in range(len(constraints))]
            )
        elif constraint_type == "nm_boost" and neg_constraints is not None:
            alpha = np.array(
                [
                    abs(constraints[i].Pi) + neg_constraints[i].Pi
                    for i in range(len(constraints))
                ]
            )
        else:
            raise ValueError(f"Unsupported constraint_type: {constraint_type}")

        # Beta calculation
        if predictions is not None and y_train is not None:
            predictions_matrix = np.vstack(predictions).T
            beta = np.max((alpha * y_train) @ predictions_matrix)
        elif sum_constraint is not None:
            beta = sum_constraint.Pi
        else:
            beta = 0.0

        if np.all(alpha == 0):
            self.logger.warning(
                "All dual values (alpha) are zero – possible degeneracy or infeasibility."
            )

        return SolveResult(
            alpha=alpha,
            beta=beta,
            weights=lp_weights,
            obj_val=model.ObjVal,
            solve_time=model.Runtime,
        )

    def set_gurobi_params(
        self,
        model,
        time_limit: int,
        num_threads: int,
        seed: int,
        verbose: bool = False,
    ):
        """
        Applies standard Gurobi parameters.
        """
        model.Params.OutputFlag = int(verbose)
        model.Params.TimeLimit = time_limit
        model.Params.Threads = num_threads
        model.Params.Seed = seed
