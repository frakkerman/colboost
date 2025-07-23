import logging
import numpy as np
from tqdm import trange
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from colboost.solvers import get_solver
from colboost.utils.predictions import create_predictions
from colboost.utils.validate import validate_base_learner, validate_inputs


logger = logging.getLogger("colboost.ensemble")


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using column generation and LP-based solvers like LPBoost.
    Parameters
    ----------
    solver : str, default="nm_boost"
        Which formulation to use. Options: "nm_boost", "cg_boost", "erlp_boost", "lp_boost", "md_boost", "qrlp_boost"

    base_estimator : object, optional
        Optional base estimator (defaults to CART decision tree if not provided).

    max_depth : int, default=1
        Maximum depth of individual trees (only relevant when using default, `base_estimator=None`)

    max_iter : int, default=100
        Maximum number of boosting iterations.

    use_crb : bool, default=False
        Whether to use confidence rated boosting, using soft-voting (only applicable for tree-based `base_estimator`).

    check_dual_const : bool, default=True
        Whether to check dual feasibility in each iteration.

    early_stopping : bool, default=True
        Stop boosting early if no improvement is observed.

    acc_eps : float, default=1e-4
        Tolerance for accuracy-based stopping criteria.

    acc_check_interval : int, default=5
        How often (in iterations) to check accuracy for early stopping.

    gurobi_time_limit : int, default=60
        Time limit (in seconds) for each Gurobi solve.

    gurobi_num_threads : int, default=1
        Number of threads Gurobi uses.

    tradeoff_hyperparam : float, default=1e-2
        Trade-off parameter for regularization.

    seed : int, default=1
        Random seed for reproducibility.
    """

    def __init__(
        self,
        solver="nm_boost",
        base_estimator=None,
        max_depth=1,
        max_iter=100,
        use_crb=False,
        check_dual_const=True,
        early_stopping=True,
        acc_eps=1e-4,
        acc_check_interval=5,
        gurobi_time_limit=60,
        gurobi_num_threads=1,
        tradeoff_hyperparam=1e-2,
        seed=1,
    ):
        self.solver = solver
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.use_crb = use_crb
        self.check_dual_const = check_dual_const
        self.acc_eps = acc_eps
        self.acc_check_interval = acc_check_interval
        self.early_stopping = early_stopping
        self.gurobi_time_limit = gurobi_time_limit
        self.gurobi_num_threads = gurobi_num_threads
        self.seed = seed
        self.tradeoff_hyperparam = tradeoff_hyperparam

        self.learners = []
        self.weights = []
        self.objective_values_ = []
        self.solve_times_ = []
        self.train_accuracies_ = []

    def fit(self, X, y):
        """
        Fit the ensemble model to the training data using column generation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values, must be -1/+1.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.solver = get_solver(self.solver)
        X, y = validate_inputs(X, y)
        self.learners = []
        self.weights = None
        beta = 0.0

        sample_weights = np.ones(len(y)) / len(y)
        prev_obj = float("inf")
        pred_matrix = []

        progress = trange(self.max_iter, desc="Boosting Progress")
        for it in progress:
            if self.base_estimator is not None:
                clf = clone(self.base_estimator)
            else:
                clf = DecisionTreeClassifier(max_depth=self.max_depth)
            validate_base_learner(clf, self.use_crb)

            clf.fit(X, y, sample_weight=sample_weights)
            preds = create_predictions(clf, X, self.use_crb)
            pred_matrix.append(preds)

            dual_sum = np.dot(sample_weights * y, preds)
            if dual_sum <= beta and self.check_dual_const:
                logger.info(
                    "Optimal solution (according to dual criterion) found. Stopping."
                )
                break

            result = self.solver.solve(
                predictions=pred_matrix,
                y_train=y,
                hyperparam=self.tradeoff_hyperparam,
                time_limit=self.gurobi_time_limit,
                num_threads=self.gurobi_num_threads,
                seed=self.seed,
            )

            if result.alpha is None or result.beta is None:
                logger.warning("No feasible solution found. Stopping.")
                break

            self.objective_values_.append(result.obj_val)
            self.solve_times_.append(result.solve_time)

            train_preds = np.sign(
                np.dot(result.weights, np.array(pred_matrix))
            )
            acc = np.mean(train_preds == y)
            self.train_accuracies_.append(acc)

            sample_weights = result.alpha
            beta = result.beta
            self.learners.append(clf)
            self.weights = result.weights

            if (
                self.early_stopping
                and len(self.train_accuracies_) >= 2 * self.acc_check_interval
            ):
                recent_avg = np.mean(
                    self.train_accuracies_[-self.acc_check_interval :]
                )
                prev_avg = np.mean(
                    self.train_accuracies_[
                        -2 * self.acc_check_interval : -self.acc_check_interval
                    ]
                )
                delta_acc = recent_avg - prev_avg

                if delta_acc < self.acc_eps:
                    progress.close()
                    logger.info(
                        f"Early stopping at iteration {it + 1}: Δacc={delta_acc:.6f} < obj_eps={self.acc_eps}"
                    )
                    break

            progress.set_postfix(
                {
                    "train acc": f"{acc:.3f}",
                }
            )

        self.n_iter_ = len(self.learners)
        self.classes_ = np.unique(y)
        return self

    def reweight_ensemble(self, X, y, learners):
        """
        Determine weights for an existing ensemble of pre-trained learners.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input samples.
        y : array-like, shape (n_samples,)
            Target labels in {-1, +1}.
        learners : list
            List of pre-trained classifiers.

        Returns
        -------
        self : object
            The estimator with updated weights.
        """
        if not learners:
            raise ValueError("List of learners must be non-empty.")
        validate_base_learner(learners[0], self.use_crb)
        X, y = validate_inputs(X, y)

        self.solver = get_solver(self.solver)

        self.learners = learners
        pred_matrix = [
            create_predictions(clf, X, self.use_crb) for clf in learners
        ]

        result = self.solver.solve(
            predictions=pred_matrix,
            y_train=y,
            hyperparam=self.tradeoff_hyperparam,
            time_limit=self.gurobi_time_limit,
            num_threads=self.gurobi_num_threads,
            seed=self.seed,
        )

        if result.weights is None:
            raise RuntimeError("Solver failed to reweight the ensemble.")

        self.weights = result.weights
        self.objective_values_ = [result.obj_val]
        self.solve_times_ = [result.solve_time]

        # Compute and store training accuracy
        train_preds = np.sign(np.dot(result.weights, np.array(pred_matrix)))
        acc = np.mean(train_preds == y)
        self.train_accuracies_ = [acc]

        self.n_iter_ = 1
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Predict class labels for input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (-1 or +1).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.learners:
            raise RuntimeError("Model has not been fitted yet.")
        pred_matrix = np.array(
            [create_predictions(clf, X, self.use_crb) for clf in self.learners]
        )
        aggregated = np.dot(self.weights, pred_matrix)
        return np.where(aggregated >= 0, 1, -1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def compute_margins(self, X, y):
        """
        Computes margin distribution y * f(x) for input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        margins : np.ndarray
            Margin values for each sample.
        """
        if not self.learners or self.weights is None:
            raise RuntimeError(
                "Model must be fitted before computing margins."
            )

        y = np.asarray(y)
        pred_matrix = np.array(
            [create_predictions(clf, X, self.use_crb) for clf in self.learners]
        )
        aggregated = np.dot(self.weights, pred_matrix)
        return y * aggregated

    @property
    def model_name_(self):
        if hasattr(self.solver, "__class__"):
            return self.solver.__class__.__name__
        return str(self.solver)
