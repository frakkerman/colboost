def create_predictions(learner, x, use_crb=False):
    if use_crb and hasattr(learner, "predict_proba"):
        proba = learner.predict_proba(x)
        return proba[:, 1] - proba[:, 0]
    else:
        return learner.predict(x)
