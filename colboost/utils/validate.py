from sklearn.utils.validation import has_fit_parameter
import numpy as np


def validate_base_learner(learner, use_crb):
    if not hasattr(learner, "fit") or not callable(learner.fit):
        raise TypeError(
            f"Base learner must implement `fit()`, got: {type(learner).__name__}"
        )
    if not hasattr(learner, "predict") or not callable(learner.predict):
        raise TypeError(
            f"Base learner must implement `predict()`, got: {type(learner).__name__}"
        )
    if not has_fit_parameter(learner, "sample_weight"):
        raise TypeError(
            f"Base learner {type(learner).__name__} must support `fit(X, y, sample_weight=...)`"
        )
    if use_crb and (
        not hasattr(learner, "predict_proba")
        or not callable(learner.predict_proba)
    ):
        raise TypeError(
            f"Learner {type(learner).__name__} has no `predict_proba`, set `use_crb=False`"
        )


def validate_inputs(X, y):
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
