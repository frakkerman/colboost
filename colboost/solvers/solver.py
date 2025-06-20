from abc import ABC, abstractmethod


class Solver(ABC):
    """
    Abstract base class for solvers assigning ensemble weights.
    """

    @abstractmethod
    def solve(self, args, data_train, env):
        """
        Run the solver and return updated weights, duals, objective, and timing.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement the solve() method."
        )
