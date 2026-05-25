from pathlib import Path
import yaml

def load_codebook(path):
    with Path(path).open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)

def codebook_label_columns(codebook, dataset_label_cols):
    codes = set((str(code) for code in codebook['codes']))
    return [label for label in dataset_label_cols if label in codes]

def format_codebook(codebook, label_cols):
    lines = []
    for code in label_cols:
        lines.append(f'{code}: {codebook['codes'][code]}')
    return '\n'.join(lines)
