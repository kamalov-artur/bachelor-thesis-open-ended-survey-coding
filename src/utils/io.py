import json
from pathlib import Path
import numpy as np
import pandas as pd

def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)

def write_json(path, payload):
    ensure_parent(path)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

def read_json(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)

def write_csv(path, df):
    ensure_parent(path)
    df.to_csv(path, index=False)

def write_npz(path, **arrays):
    ensure_parent(path)
    np.savez_compressed(path, **arrays)
