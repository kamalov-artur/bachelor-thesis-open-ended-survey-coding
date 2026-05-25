import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class BinaryRelevanceClassifier(BaseEstimator, ClassifierMixin):
    """Independent binary classifier per label with constant-label safeguards."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, x, y):
        y = np.asarray(y)
        self.n_labels_ = y.shape[1]
        self.estimators_ = []
        self.constants_ = np.full(self.n_labels_, -1, dtype=int)
        for j in range(self.n_labels_):
            values = np.unique(y[:, j])
            if len(values) == 1:
                self.estimators_.append(None)
                self.constants_[j] = int(values[0])
                continue
            estimator = clone(self.base_estimator)
            estimator.fit(x, y[:, j])
            self.estimators_.append(estimator)
        return self

    def predict(self, x):
        scores = self.decision_scores(x)
        return (scores >= 0.5).astype(int)

    def decision_scores(self, x):
        cols = []
        for j, estimator in enumerate(self.estimators_):
            if estimator is None:
                cols.append(np.full(_num_rows(x), float(self.constants_[j])))
            elif hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(x)
                cols.append(proba[:, 1] if proba.ndim == 2 else proba)
            elif hasattr(estimator, 'decision_function'):
                raw = estimator.decision_function(x)
                cols.append(1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40))))
            else:
                cols.append(estimator.predict(x).astype(float))
        return np.column_stack(cols)

def _num_rows(x):
    return x.shape[0] if hasattr(x, 'shape') else len(x)
