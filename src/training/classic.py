import pickle
import pandas as pd
from config import load_project_config
from data.loader import load_dataset
from evaluation.metrics import multilabel_metrics
from evaluation.thresholds import apply_thresholds, tune_thresholds
from models.classic_ml import build_classic_model, enabled_classic_models
from paths import ARTIFACTS_DIR
from training.artifacts import save_model_outputs
from training.split import make_validation_split
from utils.io import ensure_parent, write_csv

def train_classic_models(datasets=None, models=None):
    cfg = load_project_config()
    classic_cfg = cfg['classic_ml']
    dataset_names = datasets or list(cfg['datasets']['datasets'].keys())
    model_names = models or enabled_classic_models(classic_cfg)
    rows = []
    for dataset_name in dataset_names:
        bundle = load_dataset(dataset_name)
        for model_name in model_names:
            rows.append(train_one_classic(bundle.name, model_name))
    result = pd.DataFrame(rows)
    write_csv(ARTIFACTS_DIR / 'metrics' / 'classic_ml_runs.csv', result)
    return result

def train_one_classic(dataset_name, model_name):
    cfg = load_project_config()
    exp = cfg['experiment']
    bundle = load_dataset(dataset_name)
    texts = bundle.x_train['verbatim'].tolist()
    y = bundle.y_train.to_numpy(dtype=int)
    x_fit, x_val, y_fit, y_val = make_validation_split(texts, y, exp['validation_size'], exp['random_seed'])
    model_cfg = cfg['classic_ml']['models'][model_name]
    if model_name == 'gradient_boosting' and model_cfg.get('implementation') == 'catboost':
        model, thresholds, tuning_rows, best_params = _tune_catboost_boosting(dataset_name=dataset_name, x_fit=x_fit, y_fit=y_fit, x_val=x_val, y_val=y_val, cfg=cfg)
        write_csv(ARTIFACTS_DIR / 'metrics' / 'tuning' / 'classic_ml' / dataset_name / 'gradient_boosting_catboost.csv', pd.DataFrame(tuning_rows))
    else:
        model = build_classic_model(model_name, cfg['classic_ml'], exp['random_seed'])
        model.fit(x_fit, y_fit)
        val_scores = model.named_steps['clf'].decision_scores(model[:-1].transform(x_val))
        thresholds = tune_thresholds(y_val, val_scores, grid=exp['thresholds']['grid'], fallback_threshold=exp['thresholds']['fallback_threshold'], force_at_least_one_label=exp['thresholds']['force_at_least_one_label'])
        best_params = None
    test_texts = bundle.x_test['verbatim'].tolist()
    test_scores = model.named_steps['clf'].decision_scores(model[:-1].transform(test_texts))
    y_pred = apply_thresholds(test_scores, thresholds, force_at_least_one_label=exp['thresholds']['force_at_least_one_label'])
    y_true = bundle.y_test.to_numpy(dtype=int)
    metrics = multilabel_metrics(y_true, y_pred)
    model_dir = ARTIFACTS_DIR / 'models' / 'classic_ml' / dataset_name
    ensure_parent(model_dir / f'{model_name}.pkl')
    with (model_dir / f'{model_name}.pkl').open('wb') as fh:
        pickle.dump({'model': model, 'thresholds': thresholds, 'labels': bundle.label_names}, fh)
    return save_model_outputs(
        bundle=bundle,
        model_family='classic_ml',
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_score=test_scores,
        thresholds=thresholds,
        metrics=metrics,
        extra={'implementation': model_cfg.get('implementation', 'sklearn'), 'best_params': best_params},
    )

def _tune_catboost_boosting(dataset_name, x_fit, y_fit, x_val, y_val, cfg):
    exp = cfg['experiment']
    model_cfg = cfg['classic_ml']['models']['gradient_boosting']
    best_score = -1.0
    best_model = None
    best_thresholds = None
    best_params = None
    rows = []
    for idx, params in enumerate(model_cfg['search_space'], start=1):
        model = build_classic_model('gradient_boosting', cfg['classic_ml'], exp['random_seed'], model_params_override=params)
        model.fit(x_fit, y_fit)
        val_scores = model.named_steps['clf'].decision_scores(model[:-1].transform(x_val))
        thresholds = tune_thresholds(y_val, val_scores, grid=exp['thresholds']['grid'], fallback_threshold=exp['thresholds']['fallback_threshold'], force_at_least_one_label=exp['thresholds']['force_at_least_one_label'])
        val_pred = apply_thresholds(val_scores, thresholds, force_at_least_one_label=exp['thresholds']['force_at_least_one_label'])
        metrics = multilabel_metrics(y_val, val_pred)
        row = {
            'dataset': dataset_name,
            'candidate': idx,
            **params,
            **{f'val_{key}': value for key, value in metrics.items()},
        }
        rows.append(row)
        if metrics[model_cfg['tune_metric']] > best_score:
            best_score = metrics[model_cfg['tune_metric']]
            best_model = model
            best_thresholds = thresholds
            best_params = dict(params)
    return (best_model, best_thresholds, rows, best_params)
