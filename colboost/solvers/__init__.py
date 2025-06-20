from colboost.solvers.cg_boost import CGBoost
from colboost.solvers.lp_boost import LPBoost
from colboost.solvers.md_boost import MDBoost
from colboost.solvers.qrlp_boost import QRLPBoost
from colboost.solvers.nm_boost import NMBoost
from colboost.solvers.erlp_boost import ERLPBoost
from colboost.solvers.solver import Solver


def get_solver(solver_type: str) -> Solver:
    """
    Return Solver object corresponding to input type.

    Parameters
    ----------
    solver_type : str
        Type of solver (e.g., "lp_boost").

    Returns
    -------
    Solver
        Instance of the corresponding Solver subclass.

    Raises
    ------
    ValueError
        If solver_type is not recognized.
    """
    weigher_map = {
        "cg_boost": CGBoost,
        "erlp_boost": ERLPBoost,
        "lp_boost": LPBoost,
        "md_boost": MDBoost,
        "qrlp_boost": QRLPBoost,
        "nm_boost": NMBoost,
    }

    if solver_type in weigher_map:
        return weigher_map[solver_type]()

    raise ValueError(
        f"Solver '{solver_type}' not recognized. Available: {list(weigher_map)}"
    )
