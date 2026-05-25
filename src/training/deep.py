import pickle
import pandas as pd
from config import load_project_config
from data.loader import load_dataset
from evaluation.metrics import multilabel_metrics
from evaluation.thresholds import apply_thresholds, tune_thresholds
from models.deep_learning import predict_torch_scores, train_torch_text_model
from paths import ARTIFACTS_DIR
from training.artifacts import save_model_outputs
from training.split import make_validation_split
from utils.io import ensure_parent, write_csv

def enabled_deep_models(cfg):
    return [name for name, model_cfg in cfg['models'].items() if model_cfg.get('enabled', False) and name in {'rnn', 'cnn'}]

def train_deep_models(datasets=None, models=None):
    cfg = load_project_config()
    dataset_names = datasets or list(cfg['datasets']['datasets'].keys())
    model_names = models or enabled_deep_models(cfg['deep_learning'])
    rows = []
    for dataset_name in dataset_names:
        for model_name in model_names:
            rows.append(train_one_torch_model(dataset_name, model_name))
    result = pd.DataFrame(rows)
    write_csv(ARTIFACTS_DIR / 'metrics' / 'deep_learning_runs.csv', result)
    return result

def train_one_torch_model(dataset_name, model_name):
    cfg = load_project_config()
    deep_cfg = cfg['deep_learning']
    bundle, result, y_true, y_pred, test_scores, thresholds, metrics = fit_torch_dataset(
        dataset_name,
        model_name,
        _model_params(deep_cfg['models'][model_name]),
        cfg,
    )
    model_dir = ARTIFACTS_DIR / 'models' / 'deep_learning' / dataset_name
    ensure_parent(model_dir / f'{model_name}.pkl')
    with (model_dir / f'{model_name}.pkl').open('wb') as fh:
        pickle.dump({'state_dict': result.model.state_dict(), 'vocab': result.vocab, 'thresholds': thresholds, 'labels': bundle.label_names, 'model_name': model_name}, fh)
    return save_model_outputs(
        bundle=bundle,
        model_family='deep_learning',
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_score=test_scores,
        thresholds=thresholds,
        metrics=metrics,
    )

def fit_torch_dataset(dataset_name, model_name, model_cfg, cfg=None):
    cfg = cfg or load_project_config()
    exp = cfg['experiment']
    deep_cfg = cfg['deep_learning']
    bundle = load_dataset(dataset_name)
    texts = bundle.x_train['verbatim'].tolist()
    y = bundle.y_train.to_numpy(dtype=int)
    x_fit, x_val, y_fit, y_val = make_validation_split(texts, y, exp['validation_size'], exp['random_seed'])
    result = train_torch_text_model(
        model_name=model_name,
        train_texts=x_fit,
        y_train=y_fit,
        val_texts=x_val,
        y_val=y_val,
        model_cfg=model_cfg,
        training_cfg=deep_cfg['training'],
        random_seed=exp['random_seed'],
    )
    thresholds = tune_thresholds(
        y_val,
        result.val_scores,
        grid=exp['thresholds']['grid'],
        fallback_threshold=exp['thresholds']['fallback_threshold'],
        force_at_least_one_label=exp['thresholds']['force_at_least_one_label'],
    )
    test_scores = predict_torch_scores(
        result.model,
        bundle.x_test['verbatim'].tolist(),
        result.vocab,
        deep_cfg['training']['max_tokens'],
        deep_cfg['training']['batch_size'],
    )
    y_true = bundle.y_test.to_numpy(dtype=int)
    y_pred = apply_thresholds(test_scores, thresholds, force_at_least_one_label=exp['thresholds']['force_at_least_one_label'])
    metrics = multilabel_metrics(y_true, y_pred)
    return (bundle, result, y_true, y_pred, test_scores, thresholds, metrics)

def _model_params(model_cfg):
    return {key: value for key, value in model_cfg.items() if key != 'enabled'}
