# colboost: LP-Based Boosting with Column Generation

**colboost** is a Python library for training ensemble classifiers using linear programming (LP) based boosting methods such as LPBoost. Each iteration fits a weak learner and solves a mathematical program to determine optimal ensemble weights. The implementation is compatible with scikit-learn and supports any scikit-learn-compatible base learner. Currently, the library only supports binary classification.

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

## Example 1: fitting an ensemble

```python
from sklearn.datasets import make_classification
from colboost.ensemble import EnsembleClassifier

# Create a synthetic binary classification problem
X, y = make_classification(n_samples=200, n_features=20, random_state=0)
y = 2 * y - 1  # Convert labels from {0, 1} to {-1, +1}

# Train an LPBoost-based ensemble
model = EnsembleClassifier(solver="nm_boost", max_iter=50)
model.fit(X, y)
print("Training accuracy:", model.score(X, y))

# Obtain margin values y * f(x)
margins = model.compute_margins(X, y)
print("First 5 margins:", margins[:5])

```

## Example 2: Reweighting an existing ensemble

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from colboost.ensemble import EnsembleClassifier

# Generate data
X, y = make_classification(n_samples=200, n_features=20, random_state=42)
y = 2 * y - 1  # Convert labels to {-1, +1}

# Train AdaBoost with sklearn
ada = AdaBoostClassifier(n_estimators=10, random_state=0)
ada.fit(X, y)

# Reweight AdaBoost base estimators using colboost
model = EnsembleClassifier(solver="nm_boost")
model.reweight_ensemble(X, y, learners=ada.estimators_)

print("Training accuracy after reweighting:", model.score(X, y))
```

## Inspecting model attributes after training

```python
# assuming 'model' is the fitted colboost model
print("Learners:", model.learners) 
print("Weights:", model.weights) 
print("Objective values:", model.objective_values_)
print("Solve times:", model.solve_times_)    
print("Training accuracy per iter:", model.train_accuracies_)
print("Number of iterations:", model.n_iter_)

# compute margin distribution
margins = model.compute_margins(X, y)
print("First 5 margins (y * f(x)):", margins[:5])
```