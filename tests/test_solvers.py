import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from gurobipy import Env

from colboost.solvers.cg_boost import CGBoost
from colboost.solvers.lp_boost import LPBoost
from colboost.solvers.md_boost import MDBoost
from colboost.solvers.erlp_boost import ERLPBoost
from colboost.solvers.qrlp_boost import QRLPBoost
from colboost.solvers.nm_boost import NMBoost


def create_predictions(clf, X):
    return clf.predict(X)


@pytest.fixture
def dataset_and_preds():
    X, y = make_classification(n_samples=20, n_features=4, random_state=42)
    y = 2 * y - 1  # Convert to -1/+1
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = create_predictions(clf, X)
    return preds, y


@pytest.fixture
def gurobi_env():
    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def test_cg_boost(dataset_and_preds, gurobi_env):
    preds, y = dataset_and_preds
    solver = CGBoost()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=gurobi_env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=1.0,
    )
    assert weights is not None
    assert len(weights) == 1
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0

def test_lp_boost(dataset_and_preds, gurobi_env):
    preds, y = dataset_and_preds
    solver = LPBoost()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=gurobi_env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=0.1,
    )
    assert weights is not None
    assert len(weights) == 1
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0


def test_md_boost(dataset_and_preds, gurobi_env):
    preds, y = dataset_and_preds
    solver = MDBoost()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=gurobi_env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=1.0,
    )
    assert weights is not None
    assert len(weights) == 1
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0


def test_erlp_boost(dataset_and_preds, gurobi_env):
    preds, y = dataset_and_preds
    solver = ERLPBoost()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=gurobi_env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=10.0,
    )
    assert weights is not None
    assert len(weights) == 1
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0


def test_qrlp_boost(dataset_and_preds, gurobi_env):
    preds, y = dataset_and_preds
    solver = QRLPBoost()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=gurobi_env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=10.0,
    )
    assert weights is not None
    assert len(weights) == 1
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0


def test_nm_boost(dataset_and_preds, gurobi_env):
    preds, y = dataset_and_preds
    solver = NMBoost()
    alpha, beta, weights, obj_val, solve_time = solver.solve(
        predictions=[preds],
        y_train=y,
        env=gurobi_env,
        time_limit=10,
        num_threads=1,
        seed=0,
        hyperparam=1.0,
    )
    assert weights is not None
    assert len(weights) == 1
    assert weights[0] >= 0
    assert obj_val > 0
    assert solve_time >= 0
