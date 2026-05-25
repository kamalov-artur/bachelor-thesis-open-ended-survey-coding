import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

def multilabel_metrics(y_true, y_pred):
    return {
        'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'label_cardinality_true': float(y_true.sum(axis=1).mean()),
        'label_cardinality_pred': float(y_pred.sum(axis=1).mean()),
    }

def per_label_metrics(y_true, y_pred, labels):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    rows = []
    for i, label in enumerate(labels):
        rows.append({
            'label': label,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        })
    return rows

def sample_f1_by_row(y_true, y_pred):
    intersection = np.logical_and(y_true == 1, y_pred == 1).sum(axis=1)
    denom = y_true.sum(axis=1) + y_pred.sum(axis=1)
    return np.divide(2 * intersection, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
