from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from config import load_yaml
from paths import PROJECT_ROOT


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    topic: str
    question_type: str
    variables: list
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame

    @property
    def label_names(self):
        return list(self.y_train.columns)

def dataset_names():
    cfg = load_yaml('datasets.yaml')
    return list(cfg['datasets'].keys())

def load_dataset(name):
    cfg = load_yaml('datasets.yaml')
    data_root = PROJECT_ROOT / cfg['data_root']
    meta = cfg['datasets'][name]
    x_train = _read_x(data_root / f'{name}_Xtrain.csv')
    x_test = _read_x(data_root / f'{name}_Xtest.csv')
    y_train = _read_y(data_root / f'{name}_ytrain.csv')
    y_test = _read_y(data_root / f'{name}_ytest.csv')
    _validate_bundle(name, x_train, y_train, x_test, y_test)
    return DatasetBundle(name=name, topic=meta['topic'], question_type=meta['question_type'], variables=list(meta['variables']), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

def load_all_datasets(names=None):
    return [load_dataset(name) for name in names or dataset_names()]

def _read_x(path):
    df = pd.read_csv(path)
    df = df[['caseID', 'verbatim']].copy()
    df['verbatim'] = df['verbatim'].fillna('').astype(str)
    return df

def _read_y(path):
    df = pd.read_csv(path)
    return df.astype(int)

def _validate_bundle(name, x_train, y_train, x_test, y_test):
    if len(x_train) != len(y_train):
        raise ValueError(f'{name}: Xtrain/ytrain length mismatch')
    if len(x_test) != len(y_test):
        raise ValueError(f'{name}: Xtest/ytest length mismatch')
    if list(y_train.columns) != list(y_test.columns):
        raise ValueError(f'{name}: train/test label columns differ')
