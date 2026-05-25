import numpy as np
import pandas as pd
from evaluation.metrics import per_label_metrics
from paths import ARTIFACTS_DIR
from utils.io import write_csv, write_json, write_npz


def save_model_outputs(bundle, model_family, model_name, y_true, y_pred, y_score, thresholds, metrics, extra=None, prediction_family=None):
    prediction_family = prediction_family or model_family
    pred_path = ARTIFACTS_DIR / 'predictions' / prediction_family / bundle.name / f'{model_name}.npz'
    save_prediction_files(
        pred_path,
        ARTIFACTS_DIR / 'metrics' / 'per_label' / model_family / bundle.name / f'{model_name}.csv',
        y_true,
        y_pred,
        y_score,
        thresholds,
        bundle.label_names,
    )
    run_info = {
        'dataset': bundle.name,
        'question_type': bundle.question_type,
        'model_family': model_family,
        'model': model_name,
        'prediction_path': str(pred_path.relative_to(ARTIFACTS_DIR)),
        **(extra or {}),
        **metrics,
    }
    write_json(ARTIFACTS_DIR / 'metrics' / 'runs' / model_family / bundle.name / f'{model_name}.json', run_info)
    return run_info


def save_prediction_files(pred_path, per_label_path, y_true, y_pred, y_score, thresholds, label_names):
    write_npz(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        thresholds=thresholds,
        label_names=np.array(label_names),
    )
    write_csv(per_label_path, pd.DataFrame(per_label_metrics(y_true, y_pred, label_names)))
