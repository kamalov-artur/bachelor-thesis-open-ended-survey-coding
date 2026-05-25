import pickle
import pandas as pd
from config import load_project_config
from paths import ARTIFACTS_DIR
from training.artifacts import save_prediction_files
from training.deep import _model_params, fit_torch_dataset
from utils.io import ensure_parent, write_csv, write_json

EMBEDDING_PRESETS = {
    'glove': {
        'type': 'glove_text',
        'path': 'data/embeddings/glove.6B.50d.txt',
        'freeze': False,
    },
    'word2vec': {
        'type': 'word2vec_binary',
        'path': 'data/embeddings/GoogleNews-vectors-negative300.bin',
        'freeze': False,
    },
}

def train_deep_embedding_experiment(embedding, datasets=None, models=None):
    cfg = load_project_config()
    dataset_names = datasets or list(cfg['datasets']['datasets'].keys())
    model_names = models or ['rnn', 'cnn']
    rows = []
    for dataset_name in dataset_names:
        for model_name in model_names:
            rows.append(train_one_embedding_model(dataset_name, model_name, embedding))
    result = pd.DataFrame(rows)
    write_csv(ARTIFACTS_DIR / 'experimental' / 'deep_embeddings' / embedding / 'runs.csv', result)
    return result

def train_one_embedding_model(dataset_name, model_name, embedding):
    cfg = load_project_config()
    deep_cfg = cfg['deep_learning']
    model_cfg = _model_params(deep_cfg['models'][model_name])
    model_cfg['pretrained_embedding'] = EMBEDDING_PRESETS[embedding]
    bundle, result, y_true, y_pred, test_scores, thresholds, metrics = fit_torch_dataset(dataset_name, model_name, model_cfg, cfg)
    root = ARTIFACTS_DIR / 'experimental' / 'deep_embeddings' / embedding
    model_dir = root / 'models' / dataset_name
    ensure_parent(model_dir / f'{model_name}.pkl')
    with (model_dir / f'{model_name}.pkl').open('wb') as fh:
        pickle.dump({'state_dict': result.model.state_dict(), 'vocab': result.vocab, 'thresholds': thresholds, 'labels': bundle.label_names, 'model_name': model_name, 'embedding': embedding}, fh)
    pred_path = root / 'predictions' / dataset_name / f'{model_name}.npz'
    save_prediction_files(pred_path, root / 'per_label' / dataset_name / f'{model_name}.csv', y_true, y_pred, test_scores, thresholds, bundle.label_names)
    run_info = {
        'dataset': dataset_name,
        'question_type': bundle.question_type,
        'model_family': 'deep_learning',
        'model': f'{model_name}_{embedding}',
        'base_model': model_name,
        'embedding': embedding,
        'prediction_path': str(pred_path.relative_to(ARTIFACTS_DIR)),
        **metrics,
    }
    write_json(root / 'runs' / dataset_name / f'{model_name}.json', run_info)
    return run_info
