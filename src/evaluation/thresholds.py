import numpy as np
from evaluation.metrics import multilabel_metrics

def apply_thresholds(scores, thresholds, force_at_least_one_label=True):
    y_pred = (scores >= thresholds.reshape(1, -1)).astype(int)
    if force_at_least_one_label:
        empty = np.where(y_pred.sum(axis=1) == 0)[0]
        if len(empty):
            y_pred[empty, np.argmax(scores[empty], axis=1)] = 1
    return y_pred

def tune_thresholds(y_true, scores, grid, fallback_threshold=0.5, force_at_least_one_label=True):
    thresholds = np.full(scores.shape[1], fallback_threshold, dtype=float)
    for j in range(scores.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        best_threshold = fallback_threshold
        best_score = -1.0
        for threshold in grid:
            candidate = thresholds.copy()
            candidate[j] = threshold
            pred = apply_thresholds(scores, candidate, force_at_least_one_label)
            score = multilabel_metrics(y_true, pred)['samples_f1']
            if score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[j] = best_threshold
    return thresholds
