from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from colboost.ensemble import EnsembleClassifier
from sklearn.datasets import make_classification


def test_adaboost_fit_weights():
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    y = 2 * y - 1
    ada = AdaBoostClassifier(n_estimators=5).fit(X, (y + 1) // 2)
    model = EnsembleClassifier()
    model.reweight_ensemble(X, y, learners=ada.estimators_)
    assert len(model.learners) == 5
    assert all(w >= 0 for w in model.weights)


def test_random_forest_fit_weights():
    X, y = make_classification(n_samples=50, n_features=5, random_state=1)
    y = 2 * y - 1
    rf = RandomForestClassifier(n_estimators=4).fit(X, (y + 1) // 2)
    model = EnsembleClassifier()
    model.reweight_ensemble(X, y, rf.estimators_)
    assert len(model.weights) == 4
