import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from gurobipy import Env
from colboost.solvers import get_solver
from colboost.utils.predictions import create_predictions


logger = logging.getLogger(__name__)


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using column generation and LP-based solvers like LPBoost.
    """

    def __init__(
        self,
        solver="lp_boost",
        base_estimator=None,
        max_depth=1,
        max_iter=50,
        use_crb=False,
        check_dual_const=False,
        obj_eps=1e-4,
        obj_check=5,
        gurobi_time_limit=60,
        gurobi_num_threads=1,
        seed=0,
        tradeoff_hyperparam=1e-2,
    ):
        self.solver_name = solver
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.use_crb = use_crb
        self.check_dual_const = check_dual_const
        self.obj_eps = obj_eps
        self.obj_check = obj_check
        self.gurobi_time_limit = gurobi_time_limit
        self.gurobi_num_threads = gurobi_num_threads
        self.seed = seed
        self.tradeoff_hyperparam = tradeoff_hyperparam

        self.learners = []
        self.weights = []
        self.objective_values_ = []
        self.solve_times_ = []
        self.train_accuracies_ = []
        self.env = Env(params={"LogFile": ""})

    def __del__(self):
        if hasattr(self, "env"):
            try:
                self.env.dispose()
            except Exception:
                pass

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
        self.solver = get_solver(self.solver_name)
        X, y = np.asarray(X), np.asarray(y)
        self.learners = []
        self.weights = None
        beta = 0.0

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Only -1/+1 labels are supported.")

        sample_weights = np.ones(len(y)) / len(y)
        prev_obj = float("inf")
        pred_matrix = []

        for it in range(self.max_iter):
            if self.base_estimator is not None:
                clf = clone(self.base_estimator)
            else:
                clf = DecisionTreeClassifier(max_depth=self.max_depth)

            clf.fit(X, y, sample_weight=sample_weights)
            preds = create_predictions(clf, X, self.use_crb)
            pred_matrix.append(preds)

            dual_sum = np.dot(sample_weights * y, preds)
            if dual_sum <= beta and self.check_dual_const:
                logger.warning("Dual constraint not satisfied. Stopping.")
                break

            alpha, beta, optim_weights, objval, solve_time = self.solver.solve(
                predictions=pred_matrix,
                y_train=y,
                hyperparam=self.tradeoff_hyperparam,
                time_limit=self.gurobi_time_limit,
                num_threads=self.gurobi_num_threads,
                seed=self.seed,
                env=None,
            )

            self.objective_values_.append(objval)
            self.solve_times_.append(solve_time)

            train_preds = np.sign(np.dot(optim_weights, np.array(pred_matrix)))
            acc = np.mean(train_preds == y)
            self.train_accuracies_.append(acc)

            if alpha is None or beta is None:
                logger.warning("Solver failed. Stopping.")
                break

            sample_weights = alpha
            self.learners.append(clf)
            self.weights = optim_weights

            z_diff = prev_obj - objval
            if (it + 1) % self.obj_check == 0 and z_diff <= self.obj_eps:
                logger.info(
                    f"Stopping at iteration {it + 1}: z_diff={z_diff:.4f}"
                )
                break

            prev_obj = objval

        self.n_iter_ = len(self.learners)
        self.classes_ = np.unique(y)
        return self

    def fit_weights_only(self, X, y, learners):
        """
        Fit only the weights for a fixed ensemble of pre-trained learners.

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
        self.solver = get_solver(self.solver_name)

        X, y = np.asarray(X), np.asarray(y)
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Only -1/+1 labels are supported.")

        self.learners = learners
        pred_matrix = [create_predictions(clf, X, self.use_crb) for clf in learners]

        alpha, beta, optim_weights, objval, solve_time = self.solver.solve(
            predictions=pred_matrix,
            y_train=y,
            hyperparam=self.tradeoff_hyperparam,
            time_limit=self.gurobi_time_limit,
            num_threads=self.gurobi_num_threads,
            seed=self.seed,
            env=None,
        )

        if optim_weights is None:
            raise RuntimeError("Solver failed to reweight the ensemble.")

        self.weights = optim_weights
        self.objective_values_ = [objval]
        self.solve_times_ = [solve_time]

        # Compute and store training accuracy
        train_preds = np.sign(np.dot(optim_weights, np.array(pred_matrix)))
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
        return np.sign(np.dot(self.weights, pred_matrix))

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
            raise RuntimeError("Model must be fitted before computing margins.")

        y = np.asarray(y)
        pred_matrix = np.array([create_predictions(clf, X, self.use_crb) for clf in self.learners])
        aggregated = np.dot(self.weights, pred_matrix)
        return y * aggregated

