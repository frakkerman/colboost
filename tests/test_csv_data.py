import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from colboost.ensemble import EnsembleClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_for_tests")


def test_label_values_in_csv():
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, filename))
        y = df.iloc[:, -1].values
        assert set(np.unique(y)).issubset({-1, 1, 0}), (
            f"{filename} has invalid labels"
        )


def test_fit_on_all_csvs():
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue

        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Convert y to -1/+1 if needed
        if set(np.unique(y)) == {0, 1}:
            y = 2 * y - 1

        model = EnsembleClassifier(max_iter=10)
        model.fit(X, y)
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        print(f"{filename}: Accuracy = {acc:.2f}")
        assert len(model.learners) > 0
        assert acc > 0.5
