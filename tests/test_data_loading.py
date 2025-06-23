import os
import pandas as pd
import numpy as np
from colboost.ensemble import EnsembleClassifier

data_dir = os.path.join(os.path.dirname(__file__), "data_for_tests")

def test_label_values_in_csv():
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, fname))
            y = df.iloc[:, -1].values
            assert set(np.unique(y)).issubset({-1, 0, 1})

def test_fit_on_all_csvs():
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, fname))
            X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
            if set(np.unique(y)) == {0, 1}:
                y = 2 * y - 1
            model = EnsembleClassifier(max_iter=10)
            model.fit(X, y)
            assert model.score(X, y) > 0.5
