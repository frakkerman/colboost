from abc import ABC, abstractmethod
from typing import Any
from gurobipy import Env

class Solver(ABC):
    """
    Abstract base class for solvers assigning ensemble weights.
    """

    def __init__(self):
        self.env = Env(params={"LogFile": ""})

    def __del__(self):
        if hasattr(self, "env"):
            try:
                self.env.dispose()
            except Exception:
                pass

    @abstractmethod
    def solve(self, args: object, data_train: object, env: object) -> Any:
        """
        Run the solver and return updated weights, duals, objective, and timing.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement the solve() method."
        )
