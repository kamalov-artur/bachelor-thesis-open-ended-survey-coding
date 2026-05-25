from pathlib import Path
import yaml
from paths import CONFIG_DIR

def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)

def load_project_config():
    return {
        'datasets': load_yaml('datasets.yaml'),
        'experiment': load_yaml('experiment.yaml'),
        'classic_ml': load_yaml('models/classic_ml.yaml'),
        'deep_learning': load_yaml('models/deep_learning.yaml'),
        'encoder': load_yaml('models/encoder.yaml'),
        'decoder_gigachat': load_yaml('models/decoder_gigachat.yaml'),
    }
