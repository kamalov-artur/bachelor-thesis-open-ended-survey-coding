import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import load_yaml
from data.loader import load_dataset
from decoder.codebook import codebook_label_columns, format_codebook, load_codebook
from decoder.gigachat_client import GigaChatClient, gigachat_config_from_dict
from decoder.prompts import build_messages, parse_prediction_json
from evaluation.metrics import multilabel_metrics
from paths import ARTIFACTS_DIR, CONFIG_DIR
from utils.io import write_csv, write_json

def run_decoder_models(datasets=None, modes=None):
    cfg = load_yaml('models/decoder_gigachat.yaml')
    dataset_names = datasets or list(load_yaml('datasets.yaml')['datasets'].keys())
    rows = []
    for dataset_name in dataset_names:
        rows.extend(run_one_decoder_dataset(dataset_name, cfg, modes=modes))
    result = pd.DataFrame(rows)
    write_csv(ARTIFACTS_DIR / 'metrics' / 'decoder_gigachat_runs.csv', result)
    return result

def run_one_decoder_dataset(dataset_name, cfg, modes=None):
    bundle = load_dataset(dataset_name)
    train_df = pd.concat([bundle.x_train.reset_index(drop=True), bundle.y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([bundle.x_test.reset_index(drop=True), bundle.y_test.reset_index(drop=True)], axis=1)
    codebook_path = _resolve_codebook_path(cfg['codebooks'][dataset_name])
    codebook = load_codebook(codebook_path)
    label_cols = codebook_label_columns(codebook, bundle.label_names)
    questions = codebook.get('questions') or [codebook.get('title', f'{dataset_name} open-ended response')]
    codebook_text = format_codebook(codebook, label_cols)
    out_dir = ARTIFACTS_DIR / 'decoder_gigachat' / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / 'train_pool.csv', index=False)
    test_df.to_csv(out_dir / 'test.csv', index=False)
    write_json(out_dir / 'label_columns.json', {'labels': label_cols})
    client = GigaChatClient(gigachat_config_from_dict(cfg['gigachat']))
    summaries = []
    for mode in modes or cfg['inference']['modes']:
        completed = _load_completed_mode(out_dir / mode)
        if completed is not None:
            summaries.append(completed)
            continue
        summary = _run_mode(mode=mode, cfg=cfg, client=client, train_df=train_df, test_df=test_df, label_cols=label_cols, questions=questions, codebook_text=codebook_text, out_dir=out_dir)
        summary['question_type'] = bundle.question_type
        summaries.append(summary)
    write_json(out_dir / 'summary.json', {'dataset': dataset_name, 'summaries': summaries})
    return summaries

def _run_mode(*, mode, cfg, client, train_df, test_df, label_cols, questions, codebook_text, out_dir):
    batch_size = int(cfg['inference']['batch_size'])
    rng = random.Random(int(cfg['split']['random_state']))
    all_train_examples = _frame_to_examples(train_df, label_cols)
    icl_examples = rng.sample(all_train_examples, k=min(int(cfg['inference']['icl_examples']), len(all_train_examples)))
    vectorizer, train_matrix = _build_retriever(train_df)
    test_items = [{'case_id': str(row['caseID']), 'text': str(row['verbatim'])} for _, row in test_df.iterrows()]
    predictions = {}
    raw_dir = out_dir / mode / 'raw_responses'
    raw_dir.mkdir(parents=True, exist_ok=True)
    for batch_idx, batch_items in enumerate(_batched(test_items, batch_size), start=1):
        expected_ids = [item['case_id'] for item in batch_items]
        raw_path = raw_dir / f'batch_{batch_idx:04d}.txt'
        if raw_path.exists():
            predictions.update(parse_prediction_json(raw_path.read_text(encoding='utf-8'), expected_ids, set(label_cols), require_all=False))
            continue
        examples = _examples_for_mode(mode, batch_items, train_df, label_cols, vectorizer, train_matrix, icl_examples, int(cfg['inference']['retrieval_k']))
        messages = build_messages(mode=mode, questions=questions, codebook_text=codebook_text, label_cols=label_cols, batch_items=batch_items, examples=examples)
        content = client.chat(messages)
        raw_path.write_text(content, encoding='utf-8')
        predictions.update(parse_prediction_json(content, expected_ids, set(label_cols), require_all=False))
        time.sleep(float(cfg['inference'].get('retry_sleep_seconds', 0.2)))
    metrics = _save_decoder_predictions(out_dir / mode / 'test_predictions.csv', test_df, label_cols, predictions)
    summary = {
        'dataset': out_dir.name,
        'model_family': 'decoder',
        'model': f'gigachat_{mode}',
        'mode': mode,
        'n_test': len(test_df),
        'batch_size': batch_size,
        **metrics,
    }
    write_json(out_dir / mode / 'metrics.json', summary)
    return summary

def _save_decoder_predictions(path, test_df, label_cols, predictions):
    y_true = test_df[label_cols].to_numpy(dtype=int)
    y_pred = np.zeros_like(y_true)
    for row_idx, row in test_df.reset_index(drop=True).iterrows():
        pred_codes = set(predictions.get(str(row['caseID']), []))
        for col_idx, label in enumerate(label_cols):
            y_pred[row_idx, col_idx] = int(label in pred_codes)
    parts = [test_df[['caseID', 'verbatim']].reset_index(drop=True)]
    for col_idx, label in enumerate(label_cols):
        parts.append(pd.DataFrame({f'pred_{label}': y_pred[:, col_idx], f'true_{label}': y_true[:, col_idx]}))
    write_csv(path, pd.concat(parts, axis=1))
    return multilabel_metrics(y_true, y_pred)

def _examples_for_mode(mode, batch_items, train_df, label_cols, vectorizer, train_matrix, icl_examples, retrieval_k):
    if mode == 'zero_shot_codebook':
        return None
    if mode == 'in_context_100_codebook':
        return icl_examples
    if mode == 'retrieval_few_shot_codebook':
        query = ' '.join((item['text'] for item in batch_items))
        sims = cosine_similarity(vectorizer.transform([query]), train_matrix).ravel()
        top_idx = np.argsort(-sims)[:retrieval_k]
        return _frame_to_examples(train_df.iloc[top_idx], label_cols)
    return []

def _frame_to_examples(df, label_cols):
    return [{'text': str(row['verbatim']), 'labels': [label for label in label_cols if int(row[label]) == 1]} for _, row in df.iterrows()]

def _build_retriever(train_df):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(train_df['verbatim'].astype(str))
    return (vectorizer, matrix)

def _batched(items, batch_size):
    return [items[idx:idx + batch_size] for idx in range(0, len(items), batch_size)]

def _load_completed_mode(mode_dir):
    metrics_path = mode_dir / 'metrics.json'
    predictions_path = mode_dir / 'test_predictions.csv'
    if metrics_path.exists() and predictions_path.exists():
        return json.loads(metrics_path.read_text(encoding='utf-8'))
    return None

def _resolve_codebook_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = CONFIG_DIR / 'codebooks' / path.name
    if candidate.exists():
        return candidate
    return path
