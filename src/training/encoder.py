import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from config import load_yaml
from data.loader import load_dataset
from evaluation.metrics import multilabel_metrics
from evaluation.thresholds import apply_thresholds, tune_thresholds
from models.encoder_transformer import WeightedTransformerEncoder, make_encoder_loader, predict_encoder_scores
from paths import ARTIFACTS_DIR
from training.artifacts import save_model_outputs
from training.split import make_validation_split
from utils.io import ensure_parent, write_csv

def train_encoder_models(datasets=None):
    cfg = load_yaml('models/encoder.yaml')
    dataset_names = datasets or list(load_yaml('datasets.yaml')['datasets'].keys())
    rows = [train_one_encoder(dataset_name, cfg) for dataset_name in dataset_names]
    result = pd.DataFrame(rows)
    write_csv(ARTIFACTS_DIR / 'metrics' / 'encoder_runs.csv', result)
    return result

def train_one_encoder(dataset_name, cfg):
    bundle = load_dataset(dataset_name)
    exp_cfg = load_yaml('experiment.yaml')
    seed = int(cfg['training']['random_state'])
    _set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_files_only = bool(cfg['model'].get('local_files_only', False))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_name'], local_files_only=local_files_only)
    train_texts = bundle.x_train['verbatim'].tolist()
    y = bundle.y_train.to_numpy(dtype=np.float32)
    x_fit, x_val, y_fit, y_val = make_validation_split(train_texts, y, exp_cfg['validation_size'], exp_cfg['random_seed'])
    pos_weight = _make_pos_weight(y_fit, float(cfg['training']['pos_weight_cap'])) if cfg['training'].get('use_pos_weight') else None
    grid_results = []
    best = {'score': -1.0, 'model': None, 'params': None, 'metrics': None, 'thresholds': None}
    primary_metric = cfg['training'].get('primary_metric', 'samples_f1')
    for params in _iter_grid(cfg['grid']):
        _set_seed(seed)
        data = {
            'train_loader': make_encoder_loader(x_fit, y_fit, tokenizer, int(cfg['model']['max_len']), int(cfg['training']['batch_size']), shuffle=True),
            'val_loader': make_encoder_loader(x_val, y_val, tokenizer, int(cfg['model']['max_len']), int(cfg['training']['batch_size']), shuffle=False),
            'y_val': y_val,
            'pos_weight': pos_weight,
        }
        model, val_metrics, thresholds = _train_one_grid(params, data, cfg, len(bundle.label_names), device)
        grid_results.append({
            **params,
            **{f'val_{key}': value for key, value in val_metrics.items()},
        })
        if val_metrics.get(primary_metric, -1.0) > best['score']:
            best.update(score=val_metrics[primary_metric], model=model, params=params, metrics=val_metrics, thresholds=thresholds)
    thresholds = np.asarray(best['thresholds'], dtype=np.float32)
    test_loader = make_encoder_loader(bundle.x_test['verbatim'].tolist(), bundle.y_test.to_numpy(dtype=np.float32), tokenizer, int(cfg['model']['max_len']), int(cfg['training']['batch_size']), shuffle=False)
    scores = predict_encoder_scores(best['model'], test_loader, device)
    y_true = bundle.y_test.to_numpy(dtype=int)
    y_pred = apply_thresholds(scores, thresholds, force_at_least_one_label=exp_cfg['thresholds']['force_at_least_one_label'])
    metrics = multilabel_metrics(y_true, y_pred)
    write_csv(ARTIFACTS_DIR / 'metrics' / 'tuning' / 'encoder' / dataset_name / 'grid_results.csv', pd.DataFrame(grid_results))
    model_dir = ARTIFACTS_DIR / 'models' / 'encoder' / dataset_name
    ensure_parent(model_dir / 'best_model.pt')
    torch.save(best['model'].state_dict(), model_dir / 'best_model.pt')
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    return save_model_outputs(
        bundle=bundle,
        model_family='encoder',
        model_name='encoder',
        y_true=y_true,
        y_pred=y_pred,
        y_score=scores,
        thresholds=thresholds,
        metrics=metrics,
        extra={'base_model': cfg['model']['model_name'], 'best_params': best['params'], 'best_val_metrics': best['metrics']},
        prediction_family='deep_learning',
    )

def _train_one_grid(params, data, cfg, n_labels, device):
    pos_weight = data['pos_weight'].to(device) if data.get('pos_weight') is not None else None
    model = WeightedTransformerEncoder(model_name=cfg['model']['model_name'], n_labels=n_labels, dropout=float(params.get('dropout', cfg['model']['dropout'])), encoder_trainable=bool(params['encoder_trainable']), pos_weight=pos_weight, local_files_only=bool(cfg['model'].get('local_files_only', False))).to(device)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=float(params.get('learning_rate', cfg['training']['learning_rate'])), weight_decay=float(cfg['training']['weight_decay']))
    total_steps = max(1, len(data['train_loader']) * int(cfg['training']['epochs']))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=total_steps)
    best_state = None
    best_metrics = {}
    best_thresholds = np.full(n_labels, float(cfg['training']['threshold']), dtype=np.float32)
    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        model.train()
        epoch_loss = 0.0
        for batch in data['train_loader']:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item()) * len(labels)
        val_scores = predict_encoder_scores(model, data['val_loader'], device)
        thresholds = _tune_encoder_thresholds(data['y_val'], val_scores, cfg, params['threshold_strategy'])
        val_pred = apply_thresholds(val_scores, thresholds, force_at_least_one_label=True)
        metrics = multilabel_metrics(data['y_val'], val_pred)
        metrics['epoch'] = epoch
        metrics['train_loss'] = epoch_loss / len(data['train_loader'].dataset)
        score = metrics[cfg['training'].get('primary_metric', 'samples_f1')]
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_thresholds = thresholds
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(cfg['training']['patience']):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_metrics, best_thresholds)

def _iter_grid(grid):
    keys = list(grid.keys())
    rows = [{}]
    for key in keys:
        rows = [{**row, key: value} for row in rows for value in grid[key]]
    return rows

def _make_pos_weight(y_train, cap):
    positives = y_train.sum(axis=0)
    negatives = y_train.shape[0] - positives
    weights = negatives / np.maximum(positives, 1)
    return torch.tensor(np.minimum(weights, cap), dtype=torch.float32)

def _tune_encoder_thresholds(y_true, scores, cfg, strategy):
    training_cfg = cfg['training']
    grid = np.arange(float(training_cfg['threshold_min']), float(training_cfg['threshold_max']) + 1e-09, float(training_cfg['threshold_step'])).round(6).tolist()
    fallback = float(training_cfg['threshold'])
    if strategy == 'per_label':
        return tune_thresholds(y_true, scores, grid=grid, fallback_threshold=fallback, force_at_least_one_label=True)
    if strategy == 'global':
        best_threshold = fallback
        best_score = -1.0
        for threshold in grid:
            thresholds = np.full(scores.shape[1], threshold, dtype=float)
            pred = apply_thresholds(scores, thresholds, force_at_least_one_label=True)
            score = multilabel_metrics(y_true, pred)['samples_f1']
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return np.full(scores.shape[1], best_threshold, dtype=float)
    return np.full(scores.shape[1], fallback, dtype=float)

def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
