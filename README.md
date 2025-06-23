# colboost: LP-Based Boosting with Column Generation

**colboost** is a Python library for training ensemble classifiers using linear programming (LP) based boosting methods such as LPBoost. Each iteration fits a weak learner and solves an LP to determine optimal ensemble weights. The implementation is compatible with scikit-learn and supports any scikit-learn-compatible base learner.

## Installation

This project requires the Gurobi solver. Free academic licenses are available:

https://www.gurobi.com/academia/academic-program-and-licenses/

To install:

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
python -m pip install -i https://pypi.gurobi.com gurobipy
```

To verify the installation, in the root execute:

```bash
pytest
```

## Example

```python
from sklearn.datasets import make_classification
from colboost.ensemble import EnsembleClassifier

X, y = make_classification(n_samples=200, n_features=20, random_state=0)
y = 2 * y - 1  # Convert labels from {0, 1} to {-1, +1}

clf = EnsembleClassifier(solver="lp_boost", max_iter=50)
clf.fit(X, y)
print("Training accuracy:", clf.score(X, y))
```