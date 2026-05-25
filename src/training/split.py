import numpy as np
from sklearn.model_selection import train_test_split

def make_validation_split(texts, y, validation_size, random_seed):
    cardinality = y.sum(axis=1)
    stratify = None
    unique, counts = np.unique(cardinality, return_counts=True)
    if len(unique) > 1 and counts.min() >= 2:
        stratify = cardinality
    return train_test_split(texts, y, test_size=validation_size, random_state=random_seed, stratify=stratify)
