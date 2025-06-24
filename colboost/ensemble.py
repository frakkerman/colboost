import logging
import numpy as np
from tqdm import trange
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from colboost.solvers import get_solver
from colboost.utils.predictions import create_predictions


logger = logging.getLogger("colboost.ensemble")

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using column generation and LP-based solvers like LPBoost.
    """

    def __init__(
        self,
        solver="lp_boost",
        base_estimator=None,
        max_depth=1,
        max_iter=100,
        use_crb=False,
        check_dual_const=False,
        early_stopping=False,
        obj_eps=1e-4,
        obj_check=5,
        gurobi_time_limit=60,
        gurobi_num_threads=1,
        seed=0,
        tradeoff_hyperparam=1e-2,
    ):
        self.solver = solver
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.use_crb = use_crb
        self.check_dual_const = check_dual_const
        self.obj_eps = obj_eps
        self.obj_check = obj_check
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
        X, y = self._validate_inputs(X, y)
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
            self._validate_base_learner(clf)

            clf.fit(X, y, sample_weight=sample_weights)
            preds = create_predictions(clf, X, self.use_crb)
            pred_matrix.append(preds)

            dual_sum = np.dot(sample_weights * y, preds)
            if dual_sum <= beta and self.check_dual_const:
                logger.warning("Dual constraint not satisfied. Stopping.")
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
                logger.warning("Solver failed. Stopping.")
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

            if self.early_stopping and len(self.train_accuracies_) >= 2 * self.obj_check:
                recent_avg = np.mean(self.train_accuracies_[-self.obj_check:])
                prev_avg = np.mean(self.train_accuracies_[-2 * self.obj_check:-self.obj_check])
                delta_acc = recent_avg - prev_avg

                if delta_acc < self.obj_eps:
                    logger.info(
                        f"Early stopping at iteration {it + 1}: Δacc={delta_acc:.6f} < obj_eps={self.obj_eps}"
                    )
                    break

            progress.set_postfix({
                "train acc": f"{acc:.3f}",
            })

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
        self._validate_base_learner(learners[0])
        X, y = self._validate_inputs(X, y)

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

    @property
    def model_name_(self):
        if hasattr(self.solver, "__class__"):
            return self.solver.__class__.__name__
        return str(self.solver)

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

    def _validate_base_learner(self, learner):
        if not hasattr(learner, "fit") or not callable(learner.fit):
            raise TypeError(
                f"Base learner must implement `fit()`, got: {type(learner).__name__}"
            )
        if not hasattr(learner, "predict") or not callable(learner.predict):
            raise TypeError(
                f"Base learner must implement `predict()`, got: {type(learner).__name__}"
            )
        if self.use_crb and (
            not hasattr(learner, "predict_proba")
            or not callable(learner.predict_proba)
        ):
            logger.warning(
                f"Learner {type(learner).__name__} has no `predict_proba`; using `predict()` instead."
            )

    def _validate_inputs(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X should be 2D (got shape {X.shape})")
        if y.ndim != 1:
            raise ValueError(f"y should be 1D (got shape {y.shape})")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples (got {len(X)} and {len(y)})"
            )
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"X must be numeric (got dtype {X.dtype})")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("X and y must not contain NaN values")
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Only -1/+1 labels are supported.")

        return X, y
